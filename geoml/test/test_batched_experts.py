"""Training and predicting an expert at a time (prototype, branch
`batched-experts`).

The point of contact is `latent.expert_subset`: with every expert in it,
nothing changes. Its batched twin, `latent.network.expert_slots`, computes
the experts in a fixed number of slots so that one trace serves every set.
`VGPNetwork.train_by_expert` and `predict_by_expert` are built on them. The memory and accuracy gates are experiments
(`experiments/batched_experts/`); these tests pin the plumbing.
"""
import numpy as np
import tensorflow as tf
import pytest

import geoml
import geoml.latent as latent
import geoml.likelihood as lk
import geoml.transform as tr
import geoml.warping as wp


def _model(n_experts=4, deep=False, kernel=None):
    geoml.set_seed(1234)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (400, 2))
    y = np.sin(x[:, 0] / 15) + np.cos(x[:, 1] / 20) + 0.1 * rng.normal(
        size=400)
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", y)
    ip = geoml.data.inducing.from_kmeans(data, 40 * n_experts, seed=0)
    experts = geoml.data.inducing.experts(ip, n_experts, seed=0)
    root = latent.BasicInput(experts, transform=tr.Isotropic(20.0))
    leaf = latent.BasicGP(root, size=1, kernel=kernel)
    if deep:
        leaf = latent.BasicGP(leaf, size=1)
    return geoml.models.VGPNetwork(
        data, "v", lk.Gaussian(wp.ZScore(1)), leaf,
        options=geoml.models.GPOptions(verbose=False,
                                       training_batch_size=100))


def _targets(n=50):
    g = np.linspace(2, 98, n)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


@pytest.mark.parametrize("deep", [False, True])
def test_every_expert_in_the_subset_changes_nothing(deep):
    a, b = _model(deep=deep), _model(deep=deep)
    a.train_full(5)
    with latent.expert_subset(range(4)):
        b.train_full(5)
    np.testing.assert_allclose(a.training_log, b.training_log, rtol=1e-12)
    pa = geoml.data.PointData.from_array(_targets(10), ["X", "Y"])
    pb = geoml.data.PointData.from_array(_targets(10), ["X", "Y"])
    a.predict(pa, n_sim=5)
    with latent.expert_subset(range(4)):
        b.predict(pb, n_sim=5)
    np.testing.assert_allclose(pa.values("v/prediction"),
                               pb.values("v/prediction"), rtol=1e-10)


def test_the_weights_of_a_location_sum_to_one():
    m = _model()
    table = m.expert_weights()
    assert table.shape == (400, 4)
    np.testing.assert_allclose(table.sum(axis=1), 1.0)
    # each location leans on the expert whose ground it is in
    assert np.median(table.max(axis=1)) > 0.5


def test_training_by_expert_moves_every_expert_and_raises_the_bound():
    m = _model()
    leaf = m.leaves[0]
    before = [np.asarray(leaf.parameters["alpha_white_%d" % k].get_value())
              for k in range(4)]
    ranges = float(np.ravel(leaf.parameters["ranges"].get_value())[0])
    record = m.train_by_expert(6, batch_size=100)
    for k in range(4):
        after = np.asarray(leaf.parameters["alpha_white_%d" % k].get_value())
        assert not np.allclose(before[k], after)
    assert float(np.ravel(leaf.parameters["ranges"].get_value())[0]) \
        != ranges
    assert record["bound"][-1] > record["bound"][0]
    assert all(j in s for j, s in enumerate(record["subsets"]))


def test_the_shared_parameters_can_wait_for_the_epoch():
    m = _model()
    record = m.train_by_expert(4, batch_size=100, global_update="epoch")
    assert record["bound"][-1] > record["bound"][0]
    with pytest.raises(ValueError, match="'round' or 'epoch'"):
        m.train_by_expert(1, global_update="sometimes")


def test_prediction_by_expert_with_full_coverage_is_the_prediction():
    m = _model()
    m.train_full(10)
    a = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    b = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    m.predict(a, n_sim=5)
    # bit for bit by set; in slots the same to rounding
    info = m.predict_by_expert(b, n_sim=5, coverage=1.0, slots=False)
    assert len(info["subsets"]) == 1
    np.testing.assert_array_equal(a.values("v/prediction"),
                                  b.values("v/prediction"))
    np.testing.assert_array_equal(a.variables["v"].get_simulations(),
                                  b.variables["v"].get_simulations())
    c = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    m.predict_by_expert(c, n_sim=5, coverage=1.0)
    np.testing.assert_allclose(a.values("v/prediction"),
                               c.values("v/prediction"), rtol=1e-10)


def test_prediction_by_expert_truncates_little_and_ignores_batching():
    m = _model()
    m.train_full(10)
    a = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    b = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    m.predict(a, n_sim=5)
    info = m.predict_by_expert(b, n_sim=5, coverage=0.99, pack=False)
    assert len(info["subsets"]) > 1
    assert np.max(info["left_out"]) < 0.02
    diff = np.abs(np.asarray(a.values("v/prediction"))
                  - np.asarray(b.values("v/prediction")))
    assert np.max(diff) < 0.05
    # a location's group is its own: half the locations alone give the
    # same answer at those locations, to rounding (the batches differ in
    # shape, and a matrix product's blocking with them)
    c = geoml.data.PointData.from_array(_targets()[::2], ["X", "Y"])
    m.predict_by_expert(c, n_sim=5, coverage=0.99, pack=False)
    np.testing.assert_allclose(c.values("v/prediction"),
                               np.asarray(b.values("v/prediction"))[::2],
                               rtol=1e-12, atol=1e-14)


def test_blocks_take_the_union_of_their_sub_blocks():
    m = _model()
    m.train_full(10)
    blocks = geoml.data.Blocks2D(start=[5.0, 5.0], n=[10, 10],
                                 step=[10.0, 10.0], discretization=[2, 2])
    info = m.predict_by_expert(blocks, n_sim=5, coverage=0.99)
    assert sum(info["sizes"]) == 100
    assert np.all(np.isfinite(blocks.values("v/prediction")))


@pytest.mark.parametrize("node", ["AdditiveGP", "UncertainInputGP"])
def test_what_has_no_path_by_expert_is_refused(node):
    data = _two_variables()
    geoml.set_seed(1234)
    leaf = getattr(latent, node)(_input(data, 2), size=1)
    m = geoml.models.VGPNetwork(
        data, "v", lk.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False))
    with pytest.raises(ValueError, match="does not take"):
        m.train_by_expert(1)


def _deep_model(n_experts=4, propagation="independent"):
    """A GP reading the coordinates and the GP below them."""
    geoml.set_seed(1234)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (400, 2))
    y = np.sin(x[:, 0] / 15) + np.cos(x[:, 1] / 20) + 0.1 * rng.normal(
        size=400)
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable("v", y)
    ip = geoml.data.inducing.from_kmeans(data, 40 * n_experts, seed=0)
    experts = geoml.data.inducing.experts(ip, n_experts, seed=0)
    root = latent.BasicInput(experts, transform=tr.Isotropic(20.0))
    first = latent.BasicGP(root, size=1)
    leaf = latent.BasicGP(latent.Concatenate(root, first), size=1)
    return geoml.models.VGPNetwork(
        data, "v", lk.Gaussian(wp.ZScore(1)), leaf,
        options=geoml.models.GPOptions(verbose=False,
                                       training_batch_size=100,
                                       expert_propagation=propagation))


def _local_values(m):
    return np.concatenate([
        np.ravel(node.parameters["%s_%d" % (name, k)].get_value())
        for node in m.leaves[0].get_unique_parents() + [m.leaves[0]]
        if isinstance(node, latent.BasicGP)
        for k in range(node.root.n_experts)
        for name in ("alpha_white", "delta", "bias")])


@pytest.mark.parametrize("build", [_model, _deep_model])
def test_the_slots_train_as_the_sets_do(build):
    a, b = build(), build()
    # one sweep of the weights, so that both keep the sets it forms
    ra = a.train_by_expert(3, batch_size=50, visits=2, slots=False,
                           weights_every=10, sampling="replacement")
    rb = b.train_by_expert(3, batch_size=50, visits=2, slots=True,
                           weights_every=10, decay="steps",
                           sampling="replacement")
    np.testing.assert_allclose(ra["bound"], rb["bound"], rtol=1e-8)
    np.testing.assert_allclose(_local_values(a), _local_values(b),
                               rtol=1e-6, atol=1e-9)
    # every set a trace of its own on one path, one for them all on the
    # other
    assert ra["traces"] == len(set(ra["subsets"])) > 1
    assert rb["traces"] == 1


def test_a_round_steps_the_shared_parameters_once_a_round():
    for update, steps in (("batch", 12), ("round", 3), ("epoch", 1)):
        m = _model()
        m.train_by_expert(1, batch_size=50, visits=3, global_update=update,
                          sampling="replacement")
        assert int(m._by_expert["shared_optimizer"].iterations) == steps
    with pytest.raises(ValueError, match="'round'"):
        m.train_by_expert(1, global_update="sometimes")


@pytest.mark.parametrize("propagation", ["independent", "consensus"])
def test_concatenated_coordinates_predict_in_slots_as_the_model_does(
        propagation):
    m = _deep_model(propagation=propagation)
    m.train_full(10)
    a = geoml.data.PointData.from_array(_targets(20), ["X", "Y"])
    b = geoml.data.PointData.from_array(_targets(20), ["X", "Y"])
    m.predict(a, n_sim=5)
    info = m.predict_by_expert(b, n_sim=5, coverage=1.0)
    assert info["slots"] == 4 and len(info["subsets"]) == 1
    np.testing.assert_allclose(a.values("v/prediction"),
                               b.values("v/prediction"), rtol=1e-9)
    np.testing.assert_allclose(a.variables["v"].get_simulations(),
                               b.variables["v"].get_simulations(),
                               rtol=1e-8, atol=1e-10)


def test_concatenated_coordinates_keep_the_second_layer_local():
    m = _deep_model(n_experts=6)
    m.train_full(20)
    table = m.expert_weights()
    ordered = -np.sort(-table, axis=1)
    needed = (np.cumsum(ordered, axis=1) < 0.99).sum(axis=1) + 1
    assert needed.mean() < 4
    record = m.train_by_expert(2, batch_size=50)
    # the batches differ in size, and the second shape relaxes the trace
    assert record["traces"] <= 2
    assert np.all(np.isfinite(record["bound"]))


def test_prediction_in_slots_compiles_with_xla():
    """The slots' ids are a Variable the traced prediction reads; an int32
    one sat in host memory, which XLA on a GPU could not reach. (Only a
    GPU shows it; on a CPU this checks the compiled answer.)"""
    m = _model()
    m.train_full(10)
    a = geoml.data.PointData.from_array(_targets(20), ["X", "Y"])
    b = geoml.data.PointData.from_array(_targets(20), ["X", "Y"])
    m.predict_by_expert(a, n_sim=5)
    m.options.jit_predict = True
    m.predict_by_expert(b, n_sim=5)
    # the latent moments: XLA draws its own normals from the same seed, so
    # the realizations and what is read off them differ
    np.testing.assert_allclose(a.values("v/latent_mean"),
                               b.values("v/latent_mean"), atol=1e-8)


def test_prediction_in_slots_is_prediction_by_set():
    m = _model()
    m.train_full(10)
    a = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    b = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    ia = m.predict_by_expert(a, n_sim=5, slots=False, grouping="exact")
    ib = m.predict_by_expert(b, n_sim=5, slots=True, pack=False)
    assert ia["subsets"] == ib["subsets"]
    np.testing.assert_allclose(a.values("v/prediction"),
                               b.values("v/prediction"), rtol=1e-10)
    np.testing.assert_allclose(a.variables["v"].get_simulations(),
                               b.variables["v"].get_simulations(),
                               rtol=1e-9, atol=1e-12)
    # packed, the groups merge wherever their experts fit the slots, and
    # each location only gains experts
    c = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    ic = m.predict_by_expert(c, n_sim=5)
    assert len(ic["subsets"]) < len(ib["subsets"])
    assert all(len(s) <= ic["slots"] for s in ic["subsets"])
    diff = np.abs(np.asarray(a.values("v/prediction"))
                  - np.asarray(c.values("v/prediction")))
    assert np.max(diff) < 0.05


def test_prediction_by_expert_takes_where_and_resumes():
    m = _model()
    m.train_full(10)
    whole = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    m.predict_by_expert(whole, n_sim=5, pack=False)
    parts = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    first = np.arange(parts.n_data) < parts.n_data // 3
    info = m.predict_by_expert(parts, n_sim=5, where=first, pack=False)
    assert np.all(np.isnan(info["left_out"][~first]))
    assert np.sum(parts.unpredicted()) == np.sum(~first)
    m.predict_by_expert(parts, n_sim=5, where=parts.unpredicted(),
                        pack=False)
    np.testing.assert_allclose(parts.values("v/prediction"),
                               whole.values("v/prediction"),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("sampling", ["assignment", "replacement"])
def test_training_by_expert_reports_and_can_be_cancelled(sampling):
    m = _model()
    seen = []
    with geoml.progress(seen.append):
        m.train_by_expert(2, batch_size=50, visits=2, sampling=sampling)
    batches = [e for e in seen if e.task == "train"]
    assert [e.done for e in batches] == list(range(1, len(batches) + 1))
    # an assignment's epochs hold as many batches as the experts' rows
    # make, so it reports no total
    total = 16 if sampling == "replacement" else None
    assert total is None or len(batches) == total
    assert all(e.total == total and e.unit == "batch" for e in batches)
    assert all(np.isfinite(e.bound) for e in batches)

    class Stop(Exception):
        pass

    def stop_at_five(event):
        if event.task == "train" and event.done == 5:
            raise Stop

    m = _model()
    before = _local_values(m)
    with pytest.raises(Stop):
        with geoml.progress(stop_at_five):
            m.train_by_expert(2, batch_size=50, visits=2, sampling=sampling)
    # what the five steps did is in the parameters
    assert not np.allclose(before, _local_values(m))


def test_training_by_expert_stops_once_the_bound_settles():
    m = _model()
    m.options.training_tolerance = 0.5
    record = m.train_by_expert(40, batch_size=50)
    assert 0 < len(record["bound"]) < 40
    assert m.train_by_expert(5, batch_size=50)["bound"] == []


def test_cross_validation_by_expert():
    m = _model()
    m.train_full(10)
    labels = (m.data.coordinates[:, 0] > 50).astype(int)
    m.data.add_metadata("fold", labels)
    oof, scores = geoml.models.cross_validate(
        m, method="by_expert", epochs=2,
        expert_options=dict(batch_size=50), n_sim=5)
    assert np.all(np.isfinite(oof.values("v/prediction")))
    assert np.all(np.isfinite(scores["rmse"]))


def test_refinement_by_expert():
    geoml.set_seed(7)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 80.0, (200, 3))
    point = geoml.data.PointData.from_array(coords)
    point.add_continuous_variable("V", coords[:, 2] / 40.0 - 1.0)
    point.variables["V"].set_cutoffs([0.0])
    ip = geoml.data.inducing.experts(
        geoml.data.inducing.from_kmeans(point, 80, seed=0), 3, seed=0)
    root = latent.BasicInput(ip, transform=tr.Isotropic(40))
    model = geoml.models.VGPNetwork(
        point, "V", lk.Gaussian(), latent.BasicGP(root, size=1),
        options=geoml.models.GPOptions(verbose=False,
                                       prediction_batch_size=200))
    model.train_full(max_iter=10)
    blocks = geoml.data.BlockSet3D([0, 0, 0], [4, 4, 4], [20.0] * 3,
                                   discretization=(2, 2, 2), max_levels=2)
    a = geoml.models.refine(model, blocks, n_sim=4)
    b = geoml.models.refine(model, blocks, n_sim=4, by_expert=True)
    assert b.is_full() and not np.any(b.unpredicted())
    assert abs(a.n_data - b.n_data) <= 0.1 * a.n_data
    # with every expert in the slots, each pass is the model's prediction,
    # and so are the cuts
    c = geoml.models.refine(model, blocks, n_sim=4, by_expert=True,
                            expert_options={"coverage": 1.0})
    assert c.n_data == a.n_data
    np.testing.assert_allclose(c.values("V/prediction"),
                               a.values("V/prediction"), atol=1e-6)


def test_the_paths_take_turns_on_one_model():
    """A trace leaves its symbolic state on the nodes; the next trace of the
    other kind must not pick it up."""
    m = _deep_model()
    m.train_by_expert(1, batch_size=50)
    a = geoml.data.PointData.from_array(_targets(10), ["X", "Y"])
    m.predict(a, n_sim=3)
    m.predict_by_expert(a, n_sim=3)
    m.train_svi(1)
    m.predict_by_expert(a, n_sim=3)
    m.train_by_expert(1, batch_size=50)
    m.predict(a, n_sim=3)
    assert np.all(np.isfinite(a.values("v/prediction")))


def test_fewer_slots_keep_each_location_s_leading_experts():
    m = _model()
    m.train_full(10)
    a = geoml.data.PointData.from_array(_targets(), ["X", "Y"])
    info = m.predict_by_expert(a, n_sim=3, slots=2)
    assert info["slots"] == 2
    assert all(len(s) <= 2 for s in info["subsets"])
    # what the cap leaves out is said, location by location
    assert np.nanmax(info["left_out"]) > 0.01
    assert np.all(np.isfinite(a.values("v/prediction")))


def test_the_experts_rate_decays_on_the_epoch_clock_in_slots():
    m = _model()
    record = m.train_by_expert(2, batch_size=50, visits=2, decay="epochs")
    assert np.all(np.isfinite(record["bound"]))
    assert m._by_expert["batches"] == 16
    with pytest.raises(ValueError, match="needs slots"):
        _model().train_by_expert(1, slots=False, decay="epochs")


# --------------------------------------------------------------------------- #
# the nodes a network may hold
# --------------------------------------------------------------------------- #
def _two_variables(n=300):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, (n, 2))
    data = geoml.data.PointData.from_array(x, ["X", "Y"])
    data.add_continuous_variable(
        "v", np.sin(x[:, 0] / 15) + np.cos(x[:, 1] / 20)
        + 0.1 * rng.normal(size=n))
    data.add_continuous_variable(
        "w", np.cos(x[:, 0] / 25) - np.sin(x[:, 1] / 10)
        + 0.1 * rng.normal(size=n))
    return data


def _input(data, n_experts, kind=latent.BasicInput, seed=0):
    ip = geoml.data.inducing.from_kmeans(data, 30 * n_experts, seed=seed)
    return kind(geoml.data.inducing.experts(ip, n_experts, seed=0),
                transform=tr.Isotropic(20.0))


def _gp(parent, size=1):
    return latent.BasicGP(parent, size=size)


# each builds the leaves for "v" (and "w" where there are two)
NETWORKS = {
    # below a GP node
    "linear": lambda r: _gp(latent.Linear(r, size=2)),
    "select": lambda r: _gp(latent.SelectInput(r, [0])),
    "walk": lambda r: _gp(latent.GPWalk(_gp(r, 2), n_steps=5)),
    "bias": lambda r: _gp(latent.Bias(r)),
    "add": lambda r: _gp(latent.Concatenate(r, latent.Add(_gp(r), _gp(r)))),
    "combination": lambda r: _gp(latent.Concatenate(
        r, latent.LinearCombination(_gp(r), _gp(r)))),
    "scale": lambda r: _gp(latent.Concatenate(r, latent.Scale(_gp(r)))),
    "concatenated_linear": lambda r: _gp(latent.Concatenate(
        latent.Linear(r, size=2), _gp(r))),
    "multi_structure": lambda r: latent.MultiStructureGP(
        r, size=1, n_structures=2),
    # above the GP nodes
    "add_above": lambda r: latent.Add(_gp(r), _gp(r)),
    "multiply_above": lambda r: latent.Multiply(_gp(r), _gp(r)),
    "product_above": lambda r: latent.ProductOfExperts(_gp(r), _gp(r)),
    "exponentiation_above": lambda r: latent.Exponentiation(_gp(r)),
    "combination_above": lambda r: latent.LinearCombination(_gp(r), _gp(r)),
    "linear_above": lambda r: latent.Linear(_gp(r, 2), size=1),
    "select_above": lambda r: latent.SelectInput(_gp(r, 2), [1]),
    "scale_bias_above": lambda r: latent.Bias(latent.Scale(_gp(r))),
}


def _node_model(name):
    data = _two_variables()
    geoml.set_seed(1234)
    options = geoml.models.GPOptions(verbose=False, training_batch_size=100,
                                     expert_propagation="independent")
    if name in ("two_inputs", "stack"):
        first, second = _input(data, 2), _input(data, 3, seed=1)
        leaves = [_gp(first), _gp(second)]
        if name == "stack":
            leaves = latent.Stack(*leaves)
        return geoml.models.VGPNetwork(
            data, ["v", "w"], [lk.Gaussian(wp.ZScore(1)),
                               lk.Gaussian(wp.ZScore(1))],
            leaves, options=options)
    kind = latent.GaussianInput if name == "gaussian_input" \
        else latent.BasicInput
    root = _input(data, 4, kind)
    leaf = _gp(root) if name == "gaussian_input" else NETWORKS[name](root)
    return geoml.models.VGPNetwork(data, "v", lk.Gaussian(wp.ZScore(1)),
                                   leaf, options=options)


ALL_NETWORKS = sorted(NETWORKS) + ["gaussian_input", "two_inputs", "stack"]


@pytest.mark.parametrize("name", ALL_NETWORKS)
def test_a_network_predicts_by_expert_as_the_model_does(name):
    """Every expert in the slots is the model, whatever the nodes."""
    m = _node_model(name)
    m.train_full(5)
    a = geoml.data.PointData.from_array(_targets(8), ["X", "Y"])
    b = geoml.data.PointData.from_array(_targets(8), ["X", "Y"])
    m.predict(a, n_sim=4)
    info = m.predict_by_expert(b, n_sim=4, coverage=1.0)
    assert len(info["subsets"]) == 1
    for v in m.variables:
        np.testing.assert_allclose(a.values(v + "/prediction"),
                                   b.values(v + "/prediction"),
                                   rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(a.variables[v].get_simulations(),
                                   b.variables[v].get_simulations(),
                                   rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", ALL_NETWORKS)
def test_a_network_trains_by_expert_in_one_trace(name):
    m = _node_model(name)
    gp = [n for n in m._nodes() if isinstance(n, latent.BasicGP)]
    before = [np.asarray(n.parameters["alpha_white_0"].get_value())
              for n in gp]
    record = m.train_by_expert(2, batch_size=50)
    assert np.all(np.isfinite(record["bound"]))
    assert record["traces"] <= 2
    for node, value in zip(gp, before):
        assert not np.allclose(
            value, np.asarray(node.parameters["alpha_white_0"].get_value()))
    c = geoml.data.PointData.from_array(_targets(8), ["X", "Y"])
    m.predict_by_expert(c, n_sim=3)
    for v in m.variables:
        assert np.all(np.isfinite(c.values(v + "/prediction")))


@pytest.mark.parametrize("name", ["walk", "add", "two_inputs"])
def test_a_network_trains_in_slots_as_by_set(name):
    a, b = _node_model(name), _node_model(name)
    ra = a.train_by_expert(2, batch_size=50, visits=2, slots=False,
                           weights_every=10)
    rb = b.train_by_expert(2, batch_size=50, visits=2, slots=True,
                           weights_every=10, decay="steps")
    np.testing.assert_allclose(ra["bound"], rb["bound"], rtol=1e-8)
    assert rb["traces"] <= 2


def test_two_inputs_keep_a_set_and_a_slot_count_each():
    m = _node_model("two_inputs")
    table = m.expert_weights()
    assert table.shape == (300, 5)
    np.testing.assert_allclose(table[:, :2].sum(axis=1), 1.0)
    np.testing.assert_allclose(table[:, 2:].sum(axis=1), 1.0)
    record = m.train_by_expert(1, batch_size=50)
    assert len(record["subsets"]) == 5
    assert all(len(sets) == 2 for sets in record["subsets"])
    c = geoml.data.PointData.from_array(_targets(8), ["X", "Y"])
    info = m.predict_by_expert(c, n_sim=3)
    assert len(info["slots"]) == 2
    assert all(len(key) == 2 for key in info["subsets"])


# --------------------------------------------------------------------------- #
# a partition of the rows: every row once an epoch, the batch's expert alone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("quotas", ["equal", "weight"])
@pytest.mark.parametrize("name", ["one_input", "two_inputs"])
def test_a_partition_reads_every_row_once_an_epoch(name, quotas):
    m = _model() if name == "one_input" else _node_model(name)
    table = m.expert_weights()
    rows = np.arange(m.data.n_data)
    blocks = m._expert_blocks()
    order = np.random.default_rng(0).permutation(table.shape[1])
    batches = geoml.models.VGPNetwork._partition(
        table, rows, order, np.random.default_rng(1), 0.99, blocks, quotas)
    drawn = np.concatenate([b["rows"] for b in batches])
    np.testing.assert_array_equal(np.sort(drawn), rows)
    sizes = np.asarray([len(b["rows"]) for b in batches])
    if quotas == "equal":
        assert sizes.max() - sizes.min() <= 1
    else:
        mass = table.sum(axis=0)[order]
        exact = mass / mass.sum() * len(rows)
        assert np.all(np.abs(sizes - exact) < 1)
        assert sizes.max() - sizes.min() > 1
    for b in batches:
        for (_, start, count), s in zip(blocks, b["sets"]):
            if start <= b["expert"] < start + count:
                assert b["expert"] - start in s


@pytest.mark.parametrize("name", ["one_input", "two_inputs"])
def test_an_assignment_reads_every_row_once_an_epoch(name):
    m = _model() if name == "one_input" else _node_model(name)
    table = m.expert_weights()
    rows = np.arange(m.data.n_data)
    blocks = m._expert_blocks()
    target = len(rows) / (2 * table.shape[1])
    batches = geoml.models.VGPNetwork._assignment(
        table, rows, np.random.default_rng(1), 0.99, blocks, target)
    drawn = np.concatenate([b["rows"] for b in batches])
    np.testing.assert_array_equal(np.sort(drawn), rows)
    # a crowded expert's rows split into batches of about the target, its
    # KL shared out among them
    assert max(len(b["rows"]) for b in batches) <= 1.5 * target + 1
    kl = np.zeros(table.shape[1])
    for b in batches:
        kl[b["expert"]] += b["kl"]
        for (_, start, count), s in zip(blocks, b["sets"]):
            if start <= b["expert"] < start + count:
                assert b["expert"] - start in s
    np.testing.assert_allclose(kl[kl > 0], 1.0)


def test_an_assignment_lands_a_row_with_its_weight():
    rng = np.random.default_rng(0)
    table = rng.dirichlet(np.ones(4), size=30)
    rows = np.arange(30)
    seen = np.zeros_like(table)
    for _ in range(4000):
        for b in geoml.models.VGPNetwork._assignment(
                table, rows, rng, 0.99, [(None, 0, 4)], 30.0):
            seen[b["rows"], b["expert"]] += 1
    np.testing.assert_allclose(seen / 4000, table, atol=0.035)


def test_an_assignment_has_no_rounds():
    with pytest.raises(ValueError, match="no rounds"):
        _model().train_by_expert(1, sampling="assignment",
                                 global_update="round")


@pytest.mark.parametrize("sampling, quotas, stepping", [
    ("partition", "equal", "own"), ("partition", "weight", "own"),
    ("assignment", "equal", "own"), ("partition", "equal", "active"),
    ("assignment", "equal", "active")])
def test_an_epoch_of_a_partition_adds_up_to_the_bound(sampling, quotas,
                                                      stepping):
    """At fixed parameters, every expert active: each row counts once, each
    expert's KL once, the rest and the priors once. (The reference is built
    from its pieces: `_training_elbo` keeps the trace `train_full` made,
    which counts the priors of parameters unfixed at the time.)"""
    m = _model()
    m.train_full(5)
    for parameter in m._all_parameters:
        parameter.fix()
    rows = np.arange(m.data.n_data)
    m._refresh(m.options.jitter)
    data = float(m._data_log_lik(
        tf.constant(m.data.coordinates, tf.float64),
        tf.constant(m.y, tf.float64), tf.constant(m.has_value, tf.float64),
        [{} for _ in m.variables],
        x_var=tf.constant(m.data.get_batched_variance(rows)[0], tf.float64),
        samples=m.options.training_samples, seed=m.options.seed))
    kl = float(tf.add_n([node.kl_divergence() for node in m._nodes()]))
    full = data - kl + float(m.log_prior())
    record = m.train_by_expert(1, coverage=1.0, sampling=sampling,
                               quotas=quotas, stepping=stepping)
    np.testing.assert_allclose(record["bound"][0], full, rtol=1e-9)


@pytest.mark.parametrize("sampling", ["partition", "assignment"])
def test_a_partition_steps_only_the_batch_s_own_expert(sampling):
    m = _model()
    leaf = m.leaves[0]

    def values():
        return [np.asarray(leaf.parameters["alpha_white_%d" % k].get_value())
                for k in range(4)]

    class Stop(Exception):
        pass

    def stop(event):
        if event.task == "train" and event.done == 1:
            raise Stop

    before = values()
    with pytest.raises(Stop):
        with geoml.progress(stop):
            m.train_by_expert(1, sampling=sampling, stepping="own")
    moved = [k for k, (a, b) in enumerate(zip(before, values()))
             if not np.allclose(a, b)]
    assert len(moved) == 1


def test_an_assignment_steps_every_active_expert():
    m = _model()
    leaf = m.leaves[0]

    def values():
        return [np.asarray(leaf.parameters["alpha_white_%d" % k].get_value())
                for k in range(4)]

    class Stop(Exception):
        pass

    def stop(event):
        if event.task == "train" and event.done == 1:
            raise Stop

    before = values()
    with pytest.raises(Stop):
        with geoml.progress(stop):
            m.train_by_expert(1, coverage=1.0, sampling="assignment",
                              stepping="active")
    moved = [k for k, (a, b) in enumerate(zip(before, values()))
             if not np.allclose(a, b)]
    assert len(moved) == 4


@pytest.mark.parametrize("sampling, stepping", [
    ("partition", "own"), ("assignment", "own"), ("assignment", "active")])
def test_a_partition_trains_in_slots_as_by_set(sampling, stepping):
    a, b = _model(), _model()
    ra = a.train_by_expert(2, slots=False, weights_every=10,
                           sampling=sampling, stepping=stepping)
    rb = b.train_by_expert(2, slots=True, weights_every=10, decay="steps",
                           sampling=sampling, stepping=stepping)
    np.testing.assert_allclose(ra["bound"], rb["bound"], rtol=1e-8)
    assert rb["partition"][0]["cover_mean"] > 0.95
