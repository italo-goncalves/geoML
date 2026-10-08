"""What a node inside the tree says, written into a container.

`VGPNetwork.predict_node` stops the model's prediction at one node and writes
its moments and realizations as a `LatentVariable`. The gate: a node's
realizations, carried through the operations above it, give the leaf's --
realization s of a part is the one realization s of the whole was built
from, the seed shifts of `Add` and `Multiply` replayed along the path.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.latent as gl


def _model(leaf_of):
    """A small Walker Lake model whose leaf `leaf_of(root)` builds."""
    import tensorflow as tf
    geoml.set_seed(1234)
    tf.random.set_seed(1234)
    point, _ = geoml.datasets.walker()
    inducing = geoml.data.inducing.from_kmeans(point, 20, seed=0)
    root = gl.BasicInput(inducing, transform=geoml.transform.Isotropic(50))
    leaf, parts = leaf_of(root)
    model = geoml.models.VGPNetwork(
        point, "V", geoml.likelihood.Gaussian(), leaf,
        options=geoml.models.GPOptions(verbose=False, training_samples=6))
    model.train_full(5)
    return model, point, root, leaf, parts


def _gp(parent, size=1):
    return gl.BasicGP(parent, size=size, kernel=geoml.kernels.Gaussian())


def _sum_of_three(root):
    a, b = _gp(root), _gp(root)
    # c sits at parent 1 of an Add at parent 1 of an Add: seed + 2
    c = gl.Scale(_gp(root))
    return gl.Add(a, gl.Add(b, c)), (a, b, c)


def _grid():
    return geoml.data.Grid2D(start=[1, 1], end=[256, 291], n=[9, 9])


def _sims(variable):
    return np.stack([np.asarray(part.simulations)
                     for part in variable.components.values()], axis=1)


@pytest.fixture
def float64_realizations():
    """The seed gates compare realizations to 1e-10, which is a check of
    the draws and not of how they are stored."""
    geoml.set_realization_dtype("float64")
    yield
    geoml.set_realization_dtype("float32")


def test_the_parts_realizations_add_up_to_the_leaf_s(float64_realizations):
    model, _, _, leaf, (a, b, c) = _model(_sum_of_three)
    grid = _grid()
    for node, name in ((leaf, "leaf"), (a, "a"), (b, "b"), (c, "c")):
        model.predict_node(node, grid, n_sim=5, name=name)
    whole = _sims(grid.variables["leaf"])
    parts = sum(_sims(grid.variables[n]) for n in ("a", "b", "c"))
    np.testing.assert_allclose(parts, whole, atol=1e-10)
    # and the seed matters: a part drawn at the leaf's own seed is another
    # realization than the one the leaf was built from
    assert not np.allclose(_sims(grid.variables["b"]),
                           _sims(grid.variables["c"]))


def test_a_product_s_realizations_are_its_parents_multiplied(
        float64_realizations):
    def product(root):
        a, b = _gp(root), _gp(root)
        return gl.Multiply(a, b), (a, b)
    model, _, _, leaf, (a, b) = _model(product)
    grid = _grid()
    for node, name in ((leaf, "leaf"), (a, "a"), (b, "b")):
        model.predict_node(node, grid, n_sim=4, name=name)
    np.testing.assert_allclose(
        _sims(grid.variables["a"]) * _sims(grid.variables["b"]),
        _sims(grid.variables["leaf"]), atol=1e-10)


def test_the_leaf_s_mean_is_the_model_s_latent_mean():
    model, _, _, leaf, _ = _model(_sum_of_three)
    grid = _grid()
    model.predict(grid, n_sim=3)
    model.predict_node(leaf, grid, n_sim=3, name="leaf")
    np.testing.assert_allclose(
        grid.values("leaf/0/latent_mean"), grid.values("V/latent_mean"),
        atol=1e-10)
    np.testing.assert_allclose(
        grid.values("leaf/0/latent_variance"),
        grid.values("V/latent_variance"), atol=1e-10)


def test_a_node_s_prediction_does_not_depend_on_the_batching():
    model, _, _, _, (a, _, _) = _model(_sum_of_three)

    def run(batch_size):
        model.options.prediction_batch_size = batch_size
        grid = _grid()
        model.predict_node(a, grid, n_sim=3, name="a")
        return grid.values("a/0/latent_mean"), _sims(grid.variables["a"])

    mean_1, sims_1 = run(10 ** 6)
    mean_2, sims_2 = run(7)
    np.testing.assert_allclose(mean_1, mean_2, atol=1e-10)
    np.testing.assert_allclose(sims_1, sims_2, atol=1e-10)


def test_finishing_what_is_unpredicted_gives_the_whole():
    model, _, _, _, (a, _, _) = _model(_sum_of_three)
    whole = _grid()
    model.predict_node(a, whole, n_sim=3, name="a")

    parts = _grid()
    first = np.arange(parts.n_data) < 30
    model.predict_node(a, parts, n_sim=3, name="a", where=first)
    left = parts.unpredicted("a")
    np.testing.assert_array_equal(left, ~first)
    model.predict_node(a, parts, name="a", where=left)
    assert not parts.unpredicted("a").any()
    # the same draws; the batch's shape moves the rounding, no more
    np.testing.assert_allclose(_sims(parts.variables["a"]),
                               _sims(whole.variables["a"]), atol=1e-12)

    with pytest.raises(ValueError, match="realization"):
        model.predict_node(a, parts, n_sim=4, name="a", where=first)


def test_every_output_is_a_part_named_as_asked():
    def wide(root):
        gp = _gp(root, size=3)
        return gl.Linear(gp, size=1), (gp,)
    model, _, _, _, (gp,) = _model(wide)
    grid = _grid()
    variable = model.predict_node(gp, grid, n_sim=2, labels=["u", "v", "w"])
    assert variable is grid.variables[gp.name]
    assert variable.labels == ["u", "v", "w"]
    assert _sims(variable).shape == (grid.n_data, 3, 2)
    assert np.all(np.isfinite(grid.values(gp.name + "/w/latent_mean")))
    assert np.all(grid.values(gp.name + "/w/latent_variance") >= 0)
    with pytest.raises(ValueError, match="label"):
        model.predict_node(gp, _grid(), labels=["u", "v"])


def test_a_walk_says_where_it_moved_the_coordinates():
    def walked(root):
        walk = gl.GPWalk(_gp(root, size=2))
        return _gp(walk), (walk,)
    model, _, _, _, (walk,) = _model(walked)
    grid = _grid()
    variable = model.predict_node(walk, grid, n_sim=2, labels=["x", "y"])
    assert np.all(np.isfinite(variable.get_predictions()))
    assert variable.get_predictions().shape == (grid.n_data, 2)


def test_what_cannot_be_predicted_is_refused():
    model, point, root, _, (a, _, _) = _model(_sum_of_three)
    with pytest.raises(ValueError, match="input"):
        model.predict_node(root, _grid())
    with pytest.raises(ValueError, match="not a node"):
        model.predict_node(_gp(root), _grid())
    blocks = geoml.data.Blocks2D(start=[1, 1], end=[256, 291], n=[5, 5])
    with pytest.raises(ValueError, match="block"):
        model.predict_node(a, blocks)
    grid = _grid()
    model.predict(grid, n_sim=2)
    with pytest.raises(ValueError, match="already"):
        model.predict_node(a, grid, name="V")


def test_a_model_does_not_train_on_a_node_s_prediction():
    model, point, _, _, (a, _, _) = _model(_sum_of_three)
    model.predict_node(a, point, n_sim=2, name="a")
    with pytest.raises(TypeError, match="latent node"):
        geoml.models.VGPNetwork(
            point, "a", geoml.likelihood.Gaussian(), _gp(gl.BasicInput(
                geoml.data.inducing.from_kmeans(point, 10, seed=0))))


def test_a_node_s_prediction_survives_the_store_and_a_subset(tmp_path):
    model, _, _, _, (a, _, _) = _model(_sum_of_three)
    target = geoml.data.PointData(
        pd.DataFrame(np.random.default_rng(0).uniform(10, 250, (30, 2)),
                     columns=["X", "Y"]), ["X", "Y"])
    model.predict_node(a, target, n_sim=3, name="a")
    mean = target.values("a/0/latent_mean").copy()
    sims = _sims(target.variables["a"])

    path = str(tmp_path / "a.zarr")
    target.to_zarr(path)
    opened = geoml.data.PointData.open(path)
    assert isinstance(opened.variables["a"], geoml.data.LatentVariable)
    np.testing.assert_array_equal(opened.values("a/0/latent_mean"), mean)
    np.testing.assert_array_equal(_sims(opened.variables["a"]), sims)

    mask = np.arange(30) % 3 == 0
    np.testing.assert_array_equal(
        opened[mask].values("a/0/latent_mean"), mean[mask])
    frame = target.as_data_frame()
    assert "a_0_latent_mean" in frame.columns
