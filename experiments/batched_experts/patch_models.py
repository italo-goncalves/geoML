"""One-off patch: the expert-by-expert methods on VGPNetwork."""
p = "geoml/models.py"
s = open(p, encoding="utf-8").read()
anchor = "    def _predict(self, newdata, n_sim, include_noise, where,\n"
assert s.count(anchor) == 1
methods = '''    # ------------------------------------------------------------------ #
    # expert by expert
    # ------------------------------------------------------------------ #
    def _expert_gp_nodes(self):
        """The GP nodes an expert-by-expert pass computes, and their root,
        after refusing a model it cannot handle: directional data, leaves
        on several roots, a GP node that is not a `BasicGP`, or one that
        reads anything but the input or another `BasicGP`."""
        network = _latent.network
        if self.directional_data is not None:
            raise ValueError("training and predicting by expert does not "
                             "take directional data")
        roots = {id(leaf.root) for leaf in self.leaves}
        root = self.leaves[0].root
        if len(roots) != 1 or root is None:
            raise ValueError("training and predicting by expert needs every "
                             "leaf on one root, whose experts it visits")
        gp = [n for n in self._nodes() if isinstance(n, network._GPNode)]
        for node in gp:
            if type(node) is not network.BasicGP:
                raise ValueError("training and predicting by expert takes "
                                 "BasicGP nodes only, and %s is a %s"
                                 % (node.name, type(node).__name__))
            parent = node.parent
            if not (parent is root or type(parent) is network.BasicGP):
                raise ValueError("%s reads %s; a GP node must read the input "
                                 "or another BasicGP" % (node.name,
                                                         parent.name))
        return gp, root

    def _transformed(self, container, rows=None):
        """The locations of `container` (every row of a block's fan-out) in
        the input's transformed space, the space the kernels measure."""
        _, root = self._expert_gp_nodes()
        rows = _np.arange(container.n_data) if rows is None else rows
        coords, _ = container.get_batched_coordinates(rows)
        out = []
        for band in self.options.batch_index(
                len(coords), batch_size=self.options.prediction_batch_size):
            x = _tf.constant(coords[band], _tf.float64)
            out.append(_np.asarray(root.propagate(x)[0]))
        return _np.concatenate(out, axis=0)

    def expert_weights(self, container: "_data._SpatialData | None" = None
                       ) -> _types.FloatArray:
        """Each location's weight for each expert, as the model blends them.

        One sweep, an expert at a time, so that memory holds one expert's
        computation whatever their number: at every location the expert's
        explained variance gives its raw weight, the raw weights are
        normalized over the experts as the prediction normalizes them, and
        averaged over the outputs of every GP node that reads the input.

        Parameters
        ----------
        container
            The locations, the training data by default.

        Returns
        -------
        ndarray
            Of shape `(n, n_experts)`, rows summing to one.
        """
        gp, root = self._expert_gp_nodes()
        first = [n for n in gp if n.parent is root]
        container = self.data if container is None else container
        n_rows = container.n_data
        n_experts = root.n_experts
        n_out = sum(n.size for n in first)
        raw = _np.zeros([n_rows, n_experts, n_out])
        bands = self.options.batch_index(
            n_rows, batch_size=self.options.prediction_batch_size)
        for j in range(n_experts):
            with _latent.expert_subset((j,)):
                for node in first:
                    node.refresh(self.options.jitter)
                for band in bands:
                    rows = _np.arange(n_rows)[band]
                    coords, _ = container.get_batched_coordinates(rows)
                    variance, _ = container.get_batched_variance(rows)
                    x, x_var = root.propagate(
                        _tf.constant(coords, _tf.float64),
                        _tf.constant(variance, _tf.float64))
                    column = 0
                    for node in first:
                        _, var = node.interpolate(x, x_var)
                        var = _np.asarray(var).T
                        raw[band, j, column:column + node.size] = \\
                            (1 - var) / (var + 1e-6) + 1e-6
                        column += node.size
        weights = raw / raw.sum(axis=1, keepdims=True)
        return weights.mean(axis=2)

    def _expert_plan(self, coverage):
        """The weight table of the training data, and from it each expert's
        batch distribution, active set and shares of the KL terms."""
        table = self.expert_weights()
        measured = _np.any(self.has_value > 0, axis=1)
        w = table * measured[:, None]
        total = w.sum(axis=0)
        overlap = w.T @ w
        n_experts = table.shape[1]
        subsets, dropped = [], []
        for j in range(n_experts):
            order = _np.argsort(-overlap[j])
            share = _np.cumsum(overlap[j, order]) / overlap[j].sum()
            n = int(_np.searchsorted(share, coverage)) + 1
            keep = set(order[:n].tolist()) | {j}
            subsets.append(tuple(sorted(keep)))
            dropped.append(1.0 - overlap[j, sorted(keep)].sum()
                           / overlap[j].sum())
        # each expert's KL split among the batches it is active on, by the
        # weight it carries there, so an epoch counts it once
        carried = _np.zeros_like(overlap)
        for j, subset in enumerate(subsets):
            carried[j, list(subset)] = overlap[j, list(subset)]
        shares = carried / carried.sum(axis=0, keepdims=True)
        return dict(table=table, measured=measured, total=total,
                    subsets=subsets, dropped=_np.asarray(dropped),
                    shares=shares)

    def train_by_expert(self, epochs: int = 10,
                        batch_size: "int | None" = None,
                        coverage: float = 0.99,
                        global_update: str = "batch",
                        weights_every: int = 1) -> "dict[str, _Any]":
        """Train an expert at a time, so memory does not grow with their
        number.

        Each epoch visits the experts in a random order. For expert `j` a
        batch of data rows is drawn with probability proportional to their
        weight for `j`, and the experts active on it -- those holding
        `coverage` of the weight the batch carries -- are the only ones
        computed; their own parameters take a step on the batch. The
        batch's bound is the data term `W_j` times its mean
        log-likelihood, `W_j` the expert's total weight, so the experts'
        terms add up to the whole data term over an epoch, less each active
        expert's share of its KL divergence, the shares adding to one KL per
        expert over an epoch, and a `1 / n_experts` share of the priors.

        Parameters
        ----------
        epochs
            Number of passes over the experts.
        batch_size
            Rows per batch, `options.training_batch_size` by default.
        coverage
            The share of a batch's expert weight its active experts hold;
            the rest are left out of its blend.
        global_update
            `"batch"` steps the parameters every expert shares on each
            batch; `"epoch"` adds their gradients up over the epoch and
            takes one step.
        weights_every
            Epochs between sweeps of the weight table.

        Returns
        -------
        dict
            The run's record: the bound per epoch, seconds per epoch, the
            active sets and the weight they drop.
        """
        if global_update not in ("batch", "epoch"):
            raise ValueError("global_update is 'batch' or 'epoch', got %r"
                             % (global_update,))
        gp, root = self._expert_gp_nodes()
        n_experts = root.n_experts
        batch_size = batch_size or self.options.training_batch_size
        state = self.__dict__.get("_by_expert")
        if state is None:
            local = [[] for _ in range(n_experts)]
            for node in gp:
                for k in range(n_experts):
                    for name in ("alpha_white_%d", "delta_%d", "bias_%d"):
                        parameter = node.parameters[name % k]
                        if not parameter.fixed:
                            local[k].append(parameter.variable)
            held = {id(v) for vs in local for v in vs}
            shared = [v for v in self.get_unfixed_variables()
                      if id(v) not in held]

            def adam():
                return _tf.keras.optimizers.Adam(
                    _tf.keras.optimizers.schedules.ExponentialDecay(
                        1e-2, 1, 0.999), amsgrad=True)

            optimizers = [adam() for _ in range(n_experts)]
            for opt, variables in zip(optimizers, local):
                opt.build(variables)
            shared_optimizer = adam()
            shared_optimizer.build(shared)
            state = dict(local=local, shared=shared, optimizers=optimizers,
                         shared_optimizer=shared_optimizer, steps={},
                         rng=_np.random.default_rng(self.options.seed))
            self._by_expert = state
        local, shared = state["local"], state["shared"]
        optimizers = state["optimizers"]
        shared_optimizer = state["shared_optimizer"]
        rng = state["rng"]
        others = [n for n in self._nodes()
                  if not isinstance(n, _latent.network._GPNode)]

        def step_for(subset):
            key = (subset, global_update)
            if key in state["steps"]:
                return state["steps"][key]
            variables = [v for k in subset for v in local[k]]
            counts = [len(local[k]) for k in subset]

            @_tf.function(reduce_retracing=True)
            def step(x, y, has_value, x_var, scale, shares):
                with _tf.GradientTape() as tape:
                    self._refresh(self.options.jitter)
                    data = self._data_log_lik(
                        x, y, has_value, [{} for _ in self.variables],
                        x_var=x_var, samples=self.options.training_samples,
                        seed=self.options.seed)
                    kl = _tf.constant(0.0, _tf.float64)
                    for node in gp:
                        kl = kl + _tf.reduce_sum(
                            _tf.stack(node.expert_kl_terms()) * shares)
                    for node in others:
                        kl = kl + node.kl_divergence() / n_experts
                    bound = scale * data - kl + self.log_prior() / n_experts
                grads = tape.gradient(-bound, variables + shared)
                grads = [_tf.zeros_like(v) if g is None else g
                         for g, v in zip(grads, variables + shared)]
                start = 0
                for k, count in zip(subset, counts):
                    optimizers[k].apply_gradients(zip(
                        grads[start:start + count],
                        variables[start:start + count]))
                    start += count
                shared_grads = grads[len(variables):]
                if global_update == "batch":
                    shared_optimizer.apply_gradients(zip(shared_grads, shared))
                return bound, shared_grads

            state["steps"][key] = step
            return step

        record = dict(bound=[], seconds=[], subsets=None, dropped=None)
        plan = None
        rows = None
        with _latent.propagation_rule(self.options.expert_propagation):
            for epoch in range(epochs):
                start_time = _time.time()
                if plan is None or epoch % weights_every == 0:
                    plan = self._expert_plan(coverage)
                    rows = _np.flatnonzero(plan["measured"])
                    record["subsets"] = plan["subsets"]
                    record["dropped"] = plan["dropped"]
                total = _tf.constant(0.0, _tf.float64)
                summed = None
                for j in rng.permutation(n_experts):
                    p = plan["table"][rows, j] / plan["table"][rows, j].sum()
                    idx = rows[rng.choice(len(rows), batch_size, p=p)]
                    subset = plan["subsets"][j]
                    shares = _tf.constant(plan["shares"][j, list(subset)],
                                          _tf.float64)
                    scale = _tf.constant(plan["total"][j] / batch_size,
                                         _tf.float64)
                    with _latent.expert_subset(subset):
                        bound, grads = step_for(subset)(
                            _tf.constant(self.data.coordinates[idx],
                                         _tf.float64),
                            _tf.constant(self.y[idx], _tf.float64),
                            _tf.constant(self.has_value[idx], _tf.float64),
                            _tf.constant(
                                self.data.get_batched_variance(idx)[0],
                                _tf.float64),
                            scale, shares)
                    for pr in self._all_parameters:
                        pr.refresh()
                    total = total + bound
                    if global_update == "epoch":
                        summed = grads if summed is None else \\
                            [a + b for a, b in zip(summed, grads)]
                if global_update == "epoch":
                    shared_optimizer.apply_gradients(zip(summed, shared))
                    for pr in self._all_parameters:
                        pr.refresh()
                record["bound"].append(float(total))
                record["seconds"].append(_time.time() - start_time)
                self.training_log.append(float(total))
                if self.options.verbose:
                    print("\\rEpoch %d | bound: %s" % (epoch + 1, float(total)),
                          end="")
        if self.options.verbose:
            print("\\n")
        return record

    def predict_by_expert(self, newdata: "_data._SpatialData",
                          n_sim: "int | None" = None,
                          coverage: float = 0.99, neighbours: int = 8,
                          include_noise: bool = True) -> "dict[str, _Any]":
        """Predict with only the experts active at each location.

        The training data's expert weights are carried to the locations by
        an inverse-distance average of the `neighbours` nearest data, in the
        input's transformed space; a location's active experts are those
        holding `coverage` of its weight, a block's the union over its
        sub-blocks; the locations are grouped by their active experts and
        each group predicted with only those, the rest left out of the
        blend. A location's group depends on the location alone, so the
        answer does not depend on the batching.

        Parameters
        ----------
        newdata
            The locations to predict. Modified in place, as by `predict`.
        n_sim
            Number of realizations, as for `predict`.
        coverage
            The share of a location's expert weight its experts must hold.
        neighbours
            Data averaged into a location's weights.
        include_noise
            As for `predict`.

        Returns
        -------
        dict
            The groups: their active experts and sizes, and the weight left
            out at each location.
        """
        from scipy.spatial import cKDTree
        table = self.expert_weights()
        tree = cKDTree(self._transformed(self.data))
        rows_per = newdata.rows_per_location
        targets = self._transformed(newdata)
        distance, nearest = tree.query(targets, k=neighbours)
        inverse = 1.0 / (distance ** 2 + 1e-12)
        weights = _np.einsum("rn,rnj->rj", inverse, table[nearest]) \\
            / inverse.sum(axis=1, keepdims=True)
        weights = weights.reshape([newdata.n_data, rows_per, -1])

        groups, left_out = {}, _np.zeros(newdata.n_data)
        for loc in range(newdata.n_data):
            keep = set()
            for row in weights[loc]:
                order = _np.argsort(-row)
                n = int(_np.searchsorted(_np.cumsum(row[order]),
                                         coverage)) + 1
                keep |= set(order[:n].tolist())
            subset = tuple(sorted(keep))
            groups.setdefault(subset, []).append(loc)
            left_out[loc] = 1.0 - weights[loc][:, subset].sum(axis=1).max()

        n_sim = _simulation_count(newdata, self.variables, n_sim, keep=False)
        for subset, locs in sorted(groups.items(), key=lambda g: -len(g[1])):
            with _latent.expert_subset(subset):
                self._predict(newdata, n_sim, include_noise,
                              _np.asarray(locs), check_measurements=False)
        return dict(subsets=list(groups), sizes=[len(v) for v in
                                                 groups.values()],
                    left_out=left_out)

'''
s = s.replace(anchor, methods + anchor)
open(p, "w", encoding="utf-8").write(s)
print("patched")
