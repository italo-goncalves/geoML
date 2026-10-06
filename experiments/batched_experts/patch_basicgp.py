"""One-off patch: BasicGP's per-expert loops over the active experts."""
p = "geoml/latent/network.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:200]
    s = s.replace(old, new)


# ---------------------------------------------------------------- refresh
start = s.index(
    "            eye = tuple(_tf.eye(n, dtype=_tf.float64) "
    "for n in self.root.n_ip)\n")
end = s.index("    def cache_prediction_state(self):\n"
              "        super().cache_prediction_state()\n"
              "        self.alpha = self._cache_tuple(\"alpha\", self.alpha)")
new_refresh = '''            # every expert, or the subset an expert-by-expert pass asks for;
            # the tuples below hold the active experts in that order
            ids = _active_experts(self.root.n_experts)
            ips, ipvs = self._parent_points(ids)

            eye = tuple(_tf.eye(self.root.n_ip[i], dtype=_tf.float64)
                        for i in ids)

            cov = tuple(
                    self.covariance_matrix(ip, ip, ip_var, ip_var) + e * jitter
                    for ip, ip_var, e in zip(ips, ipvs, eye)
            )
            chol = tuple(_tf.linalg.cholesky(mat) for mat in cov)
            cov_inv = tuple(_tf.linalg.cholesky_solve(mat, e) for mat, e in zip(chol, eye))

            self.cov = cov
            self.cov_chol = chol
            self.cov_inv = cov_inv

            # posterior
            eye = tuple(_tf.tile(e[None, :, :], [self.size, 1, 1]) for e in eye)
            delta = tuple(self.parameters[f"delta_{i}"].get_value() for i in ids)
            delta_diag = tuple(_tf.linalg.diag(d) for d in delta)
            self.cov_smooth = tuple(mat[None, :, :] + d for mat, d in zip(self.cov, delta_diag))
            self.cov_smooth_chol = tuple(
                _tf.linalg.cholesky(mat + e * jitter)
                for mat, e in zip(self.cov_smooth, eye)
            )
            self.cov_smooth_inv = tuple(
                _tf.linalg.cholesky_solve(mat, e)
                for mat, e in zip(self.cov_smooth_chol, eye)
            )
            # the square root the simulations draw through, with
            # chol_r chol_r^T = K^-1 - (K+D)^-1, in the whitened form
            # L^-T chol(W) with W = (I + L^T D^-1 L)^-1 rather than as the
            # Cholesky of the difference itself: that difference cancels
            # catastrophically once K is ill-conditioned -- inducing points
            # 0.03 apart behind a fault displacement put K^-1 at 1e9 against
            # (K+D)^-1 at 1e3, and the Cholesky came back NaN in graph mode
            # while it passed eagerly, so every simulation and the prediction
            # built on them was NaN -- where W has its eigenvalues in (0, 1]
            # whatever K does
            self.chol_r = tuple(
                self._whitened_root(chol, d, e)
                for chol, d, e in zip(self.cov_chol, delta, eye)
            )

            # inducing points
            alpha_white = tuple(self.parameters[f"alpha_white_{i}"].get_value() for i in ids)
            means = tuple(
                _tf.einsum("ab,sbc->sac", mat, vec)
                for mat, vec in zip(self.cov_chol, alpha_white)
            )
            self.alpha = tuple(
                _tf.einsum("ab,sbc->sac", mat, vec)
                for mat, vec in zip(self.cov_inv, means)
            )

            # inducing points, for whatever is built on top of this node.
            # Under the default rule every expert's set is predicted from
            # every other and combined by precision weighting -- the one
            # quadratic step in the network; with a terminal node it is pure
            # waste, since nothing ever reads the result. Under
            # `GPOptions(expert_propagation="independent")` each expert
            # speaks for its own set alone: duplicated points in overlapping
            # sets are then free to disagree (measured at several latent
            # standard deviations), and the data-side weighting in
            # `interpolate` arbitrates. That trades the consensus for O(K)
            # cost -- measured 6.3x training and 8x prediction at 40 experts,
            # with quality within a few percent either way. Under an expert
            # subset the sets are those of the active experts, in their
            # order, and only the active experts are consulted.
            self._point_experts = None if _EXPERT_SUBSET is None else ids
            if len(self.children) > 0:
                bias = [self.parameters[f'bias_{i}'].get_value() for i in ids]

                self.inducing_points = []
                self.inducing_points_variance = []
                if _EXPERT_PROPAGATION == "independent":
                    for p in range(len(ids)):
                        ip_i = ips[p]
                        ipv_i = ipvs[p]
                        cov = self.covariance_matrix(ip_i, ip_i, ipv_i, ipv_i)
                        mean = _tf.einsum(
                            "ab,sbc->sac", cov, self.alpha[p]) + bias[p]
                        pred_var = 1.0 - _tf.reduce_sum(
                            _tf.einsum("ab,sbc->sac", cov,
                                       self.cov_smooth_inv[p])
                            * cov[None, :, :],
                            axis=2, keepdims=False
                        )
                        self.inducing_points.append(
                            _tf.transpose(mean[:, :, 0]))
                        self.inducing_points_variance.append(
                            _tf.transpose(pred_var))
                else:
                    for p in range(len(ids)):
                        ip_i = ips[p]
                        ipv_i = ipvs[p]
                        means = []
                        pred_vars = []
                        for q in range(len(ids)):
                            ip_j = ips[q]
                            ipv_j = ipvs[q]
                            cov = self.covariance_matrix(ip_i, ip_j, ipv_i, ipv_j)
                            means.append(_tf.einsum("ab,sbc->sac", cov, self.alpha[q]) + bias[q])
                            pred_vars.append(
                                1.0 - _tf.reduce_sum(
                                    _tf.einsum("ab,sbc->sac", cov, self.cov_smooth_inv[q]) * cov[None, :, :],
                                    axis=2, keepdims=False
                                )
                            )
                        means = _tf.stack(means, axis=0)  # [n_experts, n_latent, n_data, 1]
                        pred_vars = _tf.stack(pred_vars, axis=0)  # [n_experts, n_latent, n_data]
                        weights = _GPNode.get_expert_weights(pred_vars)
                        self.inducing_points.append(
                            _tf.transpose(_tf.reduce_sum(means[:, :, :, 0] * weights, axis=0))
                        )
                        self.inducing_points_variance.append(
                            _tf.transpose(_tf.reduce_sum(pred_vars * weights, axis=0))
                        )

    def _parent_points(self, ids):
        """The parent's inducing points and their variances for the experts
        `ids`, in that order. An input holds every expert's; a GP node
        refreshed under a subset holds the active experts' only, in the
        order of `_point_experts`."""
        held = getattr(self.parent, "_point_experts", None)
        positions = ids if held is None else [held.index(i) for i in ids]
        ips = [self.parent.inducing_points[p] for p in positions]
        ipvs = [self.parent.inducing_points_variance[p] for p in positions]
        return ips, ipvs

'''
s = s[:start] + new_refresh + s[end:]

# ------------------------------------------------- cache_prediction_state
sub('''    def cache_prediction_state(self):
        super().cache_prediction_state()
        self.alpha = self._cache_tuple("alpha", self.alpha)
        self.cov_inv = self._cache_tuple("cov_inv", self.cov_inv)
        self.cov_smooth_inv = self._cache_tuple(
            "cov_smooth_inv", self.cov_smooth_inv)
        self.chol_r = self._cache_tuple("chol_r", self.chol_r)

    def _moments(self, x, x_var=None):
        with _tf.name_scope("basic_interpolation"):
            cov_cross = [
                self.covariance_matrix(x, ip, x_var, ip_var)
                for ip, ip_var in zip(self.parent.inducing_points, self.parent.inducing_points_variance)
            ]

            bias = [self.parameters[f'bias_{i}'].get_value() for i in range(self.root.n_experts)]''',
'''    def cache_prediction_state(self):
        if _EXPERT_SUBSET is None:
            super().cache_prediction_state()
            self.alpha = self._cache_tuple("alpha", self.alpha)
            self.cov_inv = self._cache_tuple("cov_inv", self.cov_inv)
            self.cov_smooth_inv = self._cache_tuple(
                "cov_smooth_inv", self.cov_smooth_inv)
            self.chol_r = self._cache_tuple("chol_r", self.chol_r)
            return
        # under a subset the tuples hold the active experts only, so each
        # snapshot is named after its expert rather than its position, and
        # a Variable keeps the one shape its expert gives it
        ids = _active_experts(self.root.n_experts)

        def by_expert(name, values):
            return tuple(self._state_var("%s_%d" % (name, i), v)
                         for i, v in zip(ids, values))

        if self.inducing_points is not None:
            self.inducing_points = by_expert(
                "inducing_points", self.inducing_points)
        if self.inducing_points_variance is not None:
            self.inducing_points_variance = by_expert(
                "inducing_points_variance", self.inducing_points_variance)
        self.alpha = by_expert("alpha", self.alpha)
        self.cov_inv = by_expert("cov_inv", self.cov_inv)
        self.cov_smooth_inv = by_expert("cov_smooth_inv", self.cov_smooth_inv)
        self.chol_r = by_expert("chol_r", self.chol_r)

    def _moments(self, x, x_var=None):
        with _tf.name_scope("basic_interpolation"):
            ids = _active_experts(self.root.n_experts)
            ips, ipvs = self._parent_points(ids)
            cov_cross = [
                self.covariance_matrix(x, ip, x_var, ip_var)
                for ip, ip_var in zip(ips, ipvs)
            ]

            bias = [self.parameters[f'bias_{i}'].get_value() for i in ids]''')

# ----------------------------------------------------------- kl_divergence
sub('''    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            all_kl = []
            for i in range(self.root.n_experts):
                delta = self.parameters[f"delta_{i}"].get_value()
                alpha_white = self.parameters[f"alpha_white_{i}"].get_value()

                tr = _tf.reduce_sum(self.cov_smooth_inv[i] * self.cov[i][None, :, :])
                fit = _tf.reduce_sum(alpha_white**2)
                det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                    _tf.linalg.diag_part(self.cov_smooth_chol[i])))
                det_2 = _tf.reduce_sum(_tf.math.log(delta))
                kl = 0.5 * (- tr + fit + det_1 - det_2)

                all_kl.append(kl)

            return _tf.add_n(all_kl)
''',
'''    def kl_divergence(self):
        with _tf.name_scope("basic_KL_divergence"):
            return _tf.add_n(self.expert_kl_terms())

    def expert_kl_terms(self):
        """Each active expert's KL divergence, in the order of the active
        experts -- what `kl_divergence` adds up."""
        all_kl = []
        for p, i in enumerate(_active_experts(self.root.n_experts)):
            delta = self.parameters[f"delta_{i}"].get_value()
            alpha_white = self.parameters[f"alpha_white_{i}"].get_value()

            tr = _tf.reduce_sum(self.cov_smooth_inv[p] * self.cov[p][None, :, :])
            fit = _tf.reduce_sum(alpha_white**2)
            det_1 = 2 * _tf.reduce_sum(_tf.math.log(
                _tf.linalg.diag_part(self.cov_smooth_chol[p])))
            det_2 = _tf.reduce_sum(_tf.math.log(delta))
            kl = 0.5 * (- tr + fit + det_1 - det_2)

            all_kl.append(kl)
        return all_kl
''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
