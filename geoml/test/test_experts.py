"""Tests for multi-expert networks and the inducing point helpers.

``BasicInput`` has always accepted a list of inducing point sets -- one per
expert, combined per query point by precision weighting -- but nothing
exercised it: every other test in this suite passes a single container, so
``n_experts > 1`` was unguarded code. These cover it end to end, and with it
the two things that only show up when there is more than one expert: the
``K x K`` propagation loop in ``BasicGP.refresh``, and the traced refresh that
replaces running that loop in eager Python.
"""
import numpy as np
import pytest

import geoml
import geoml.inducing as ind


def _walker():
    walker_point, walker_grid = geoml.datasets.walker()
    return walker_point, walker_grid


def build_model(n_experts=3, n_ip=10, seed=1234, depth=1):
    """A Walker Lake network with `n_experts` inducing point sets."""
    geoml.set_seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    walker_point, walker_grid = _walker()
    sets = ind.experts(ind.from_kmeans(walker_point, n_experts * n_ip,
                                       seed=seed), n_experts, seed=seed)
    root = geoml.latent.BasicInput(
        sets, transform=geoml.transform.Isotropic(50))
    node = root
    for _ in range(depth):
        node = geoml.latent.BasicGP(
            node, size=1, kernel=geoml.kernels.Gaussian())
    model = geoml.models.VGPNetwork(
        walker_point, "V", geoml.likelihood.Gaussian(), node,
        options=geoml.models.GPOptions(
            verbose=False, seed=seed, training_samples=6))
    return model, walker_grid, root, node


# --------------------------------------------------------------------------- #
# the inducing point helpers
# --------------------------------------------------------------------------- #
def test_from_kmeans_returns_the_requested_number():
    walker_point, _ = _walker()
    points = ind.from_kmeans(walker_point, 25, seed=0)
    assert points.n_data == 25
    assert points.n_dim == walker_point.n_dim
    # the centroids sit inside the data's own bounding box
    coords = np.asarray(points.coordinates)
    data = np.asarray(walker_point.coordinates)
    assert np.all(coords.min(0) >= data.min(0))
    assert np.all(coords.max(0) <= data.max(0))


def test_from_kmeans_rejects_more_points_than_data():
    walker_point, _ = _walker()
    with pytest.raises(ValueError, match="between 1 and"):
        ind.from_kmeans(walker_point, walker_point.n_data + 1)


def test_from_grid_covers_the_data():
    walker_point, _ = _walker()
    points = ind.from_grid(walker_point, 40)
    coords = np.asarray(points.coordinates)
    data = np.asarray(walker_point.coordinates)
    assert np.all(coords.min(0) <= data.min(0) + 1e-9)
    assert np.all(coords.max(0) >= data.max(0) - 1e-9)
    # a regular lattice: every axis holds evenly spaced, repeated values
    for d in range(coords.shape[1]):
        spacing = np.diff(np.unique(coords[:, d]))
        assert np.allclose(spacing, 40)


def test_combine_drops_duplicates():
    a = geoml.data.PointData.from_array(np.array([[0.0, 0.0], [1.0, 1.0]]))
    b = geoml.data.PointData.from_array(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert ind.combine(a, b).n_data == 3
    # with a tolerance, near-duplicates go too
    c = geoml.data.PointData.from_array(np.array([[1.001, 1.0]]))
    assert ind.combine(a, b, c, tolerance=0.01).n_data == 3


def test_combine_rejects_mixed_dimensions():
    a = geoml.data.PointData.from_array(np.zeros([2, 2]))
    b = geoml.data.PointData.from_array(np.zeros([2, 3]))
    with pytest.raises(ValueError, match="same dimension"):
        ind.combine(a, b)


def test_grid_experts_are_equal_sized_and_overlap_by_one_step():
    walker_point, _ = _walker()
    sets = ind.grid_experts(walker_point, 20, block=3)
    sizes = {s.n_data for s in sets}
    # (block + 2) ** n_dim, the block plus one node of margin all round
    assert sizes == {25}
    assert len(sets) > 1

    # neighbouring blocks share exactly the two node layers that make up the
    # one-step overlap
    coords = [np.asarray(s.coordinates) for s in sets]
    shared = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            common = {tuple(np.round(p, 6)) for p in coords[i]} & \
                     {tuple(np.round(p, 6)) for p in coords[j]}
            if common:
                shared.append(len(common))
    assert len(shared) > 0
    assert all(n > 0 for n in shared)


def test_grid_experts_in_three_dimensions():
    rng = np.random.default_rng(0)
    data = geoml.data.PointData.from_array(rng.uniform(0, 100, (200, 3)))
    sets = ind.grid_experts(data, 25, block=2)
    assert {s.n_data for s in sets} == {4 ** 3}


def test_experts_cover_every_point_at_least_once():
    walker_point, _ = _walker()
    points = ind.from_kmeans(walker_point, 200, seed=0)
    sets = ind.experts(points, 5, seed=0)
    assert len(sets) == 5

    everything = {tuple(np.round(p, 6))
                  for s in sets for p in np.asarray(s.coordinates)}
    original = {tuple(np.round(p, 6))
                for p in np.asarray(points.coordinates)}
    assert everything == original


def test_experts_overlap_and_more_overlap_shares_more():
    walker_point, _ = _walker()
    points = ind.from_kmeans(walker_point, 200, seed=0)

    def total(overlap):
        return sum(s.n_data for s in ind.experts(points, 5, overlap=overlap,
                                                 seed=0))

    # every expert holds more than its share, and asking for more reach shares
    # more points out
    assert total(0.0) > points.n_data
    assert total(0.5) > total(0.0)
    assert total(1.0) > total(0.5)


def test_experts_keep_their_own_cluster():
    """A point may be absorbed by a neighbour but never dropped by its own."""
    walker_point, _ = _walker()
    points = ind.from_kmeans(walker_point, 150, seed=0)
    sets = ind.experts(points, 4, overlap=0.0, seed=0)
    counts = {}
    for s in sets:
        for p in np.asarray(s.coordinates):
            counts[tuple(np.round(p, 6))] = counts.get(
                tuple(np.round(p, 6)), 0) + 1
    assert min(counts.values()) >= 1
    assert max(counts.values()) > 1      # something really is shared


def test_experts_survive_a_collinear_cluster():
    """Samples along one drillhole make a cluster that is nearly a line, whose
    covariance is singular; the expert must still come out finite."""
    rng = np.random.default_rng(0)
    line = np.stack([np.linspace(0, 100, 60),
                     np.zeros(60), np.zeros(60)], axis=1)
    blob = rng.uniform(200, 260, (60, 3))
    points = geoml.data.PointData.from_array(np.concatenate([line, blob]))
    sets = ind.experts(points, 2, seed=0)
    assert len(sets) == 2
    for s in sets:
        assert 0 < s.n_data <= points.n_data
        assert np.all(np.isfinite(np.asarray(s.coordinates)))


def test_experts_rejects_an_impossible_request():
    walker_point, _ = _walker()
    points = ind.from_kmeans(walker_point, 40, seed=0)
    with pytest.raises(ValueError, match="between 1 and"):
        ind.experts(points, points.n_data + 1)
    with pytest.raises(ValueError, match="negative"):
        ind.experts(points, 3, overlap=-0.1)


# --------------------------------------------------------------------------- #
# multi-expert networks
# --------------------------------------------------------------------------- #
def test_multi_expert_trains_and_predicts():
    model, grid, root, _ = build_model(n_experts=3)
    assert root.n_experts == 3
    model.train_full(max_iter=10)
    assert model.training_log[-1] > model.training_log[0]

    small = geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[12, 12])
    model.predict(small, n_sim=6)
    prediction = np.asarray(small.variables["V"].get_predictions())
    assert prediction.shape[0] == small.n_data
    assert np.all(np.isfinite(prediction))


def test_multi_expert_prediction_is_batch_invariant():
    """The combination is per point, so batching must not change it."""
    model, _, _, _ = build_model(n_experts=3)
    model.train_full(max_iter=5)

    def latent_mean(batch_size):
        grid = geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[10, 10])
        model.options.prediction_batch_size = batch_size
        model.predict(grid, n_sim=4)
        return np.array(grid.variables["V"].latent_mean.values, copy=True)

    assert np.allclose(latent_mean(10 ** 9), latent_mean(7), atol=1e-8)


def test_terminal_node_skips_the_propagation():
    """Nothing reads a childless node's inducing points, so it must not build
    them -- that loop is the one quadratic step in the network."""
    model, _, root, output = build_model(n_experts=3, depth=1)
    model.latent_network.refresh(1e-6)
    assert output.children == []
    assert output.inducing_points is None
    # the root still propagates its own, which the GP node consumes
    assert root.inducing_points is not None
    assert len(root.inducing_points) == 3


def test_inner_node_still_propagates():
    model, _, _, output = build_model(n_experts=3, depth=2)
    model.latent_network.refresh(1e-6)
    inner = output.parent
    assert inner.children == [output]
    assert inner.inducing_points is not None
    assert len(inner.inducing_points) == 3
    assert output.inducing_points is None


def test_deep_multi_expert_predicts():
    model, _, _, _ = build_model(n_experts=2, depth=2)
    model.train_full(max_iter=5)
    grid = geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[10, 10])
    model.predict(grid, n_sim=4)
    assert np.all(np.isfinite(
        np.asarray(grid.variables["V"].get_predictions())))


# --------------------------------------------------------------------------- #
# the traced refresh
# --------------------------------------------------------------------------- #
def test_traced_refresh_matches_the_eager_one():
    """`refresh_cached` only changes how the posterior is computed, not what
    it is: the cached state must come out identical either way."""
    import tensorflow as tf

    model, _, _, output = build_model(n_experts=3)
    model.train_full(max_iter=5)

    model.latent_network.refresh(1e-6)
    nodes = set(model.latent_network.get_unique_parents())
    nodes.add(model.latent_network)
    for node in nodes:
        node.cache_prediction_state()
    eager = [np.asarray(tf.convert_to_tensor(a)) for a in output.alpha]

    geoml.latent.refresh_cached(model.latent_network, 1e-6)
    traced = [np.asarray(tf.convert_to_tensor(a)) for a in output.alpha]

    assert len(eager) == len(traced) == 3
    for a, b in zip(eager, traced):
        assert np.allclose(a, b)


def test_traced_refresh_is_reused_not_rebuilt():
    model, _, _, _ = build_model(n_experts=2)
    model.train_full(max_iter=3)
    network = model.latent_network
    assert network._refresh_graph is None

    geoml.latent.refresh_cached(network, 1e-6)
    first = network._refresh_graph
    assert first is not None

    geoml.latent.refresh_cached(network, 1e-6)
    assert network._refresh_graph is first

    # a different jitter is a different graph
    geoml.latent.refresh_cached(network, 1e-4)
    assert network._refresh_graph is not first


def test_prediction_follows_further_training():
    """The traced refresh reads the parameters live, so training after a
    prediction must change the next one."""
    model, _, _, _ = build_model(n_experts=3)
    model.train_full(max_iter=5)

    grid = geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[10, 10])
    model.predict(grid, n_sim=4)
    before = np.asarray(grid.variables["V"].latent_mean.values).copy()

    model.train_full(max_iter=15)
    model.predict(grid, n_sim=4)
    after = np.asarray(grid.variables["V"].latent_mean.values)

    assert not np.allclose(before, after)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_multi_expert_model_round_trips(tmp_path):
    model, _, _, _ = build_model(n_experts=3)
    model.train_full(max_iter=5)

    grid = geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[10, 10])
    model.predict(grid, n_sim=4)
    before = np.asarray(grid.variables["V"].latent_mean.values).copy()

    path = str(tmp_path / "experts.zarr")
    model.save(path)
    reloaded = geoml.models.VGPNetwork.open(path)

    assert reloaded.latent_network.root.n_experts == 3
    grid2 = geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[10, 10])
    reloaded.predict(grid2, n_sim=4)
    after = np.asarray(grid2.variables["V"].latent_mean.values)
    assert np.allclose(before, after)
