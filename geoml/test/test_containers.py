"""Batched-write smoke tests across the container types.

Every container feeds coordinates to the model and receives predictions written
back in batches through ``variable.update(idx, ...)``. These tests confirm the
Zarr/NumPy storage refactor preserves that contract for each container shape,
including the ``Blocks`` discretization path: with discretization enabled a
block fans out into several sub-points for the model, but the prediction written
back is still one row per block.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


def _synth_point_model(n_dim, seed=1234, n_train=40, n_ip=6, max_iter=5):
    """A minimal continuous VGP on random points in ``n_dim`` dimensions."""
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cols = [f"c{i}" for i in range(n_dim)]
    coords = np.random.uniform(0, 100, (n_train, n_dim))
    values = np.sin(coords[:, 0] / 25.0)
    df = pd.DataFrame(coords, columns=cols)
    point = geoml.data.PointData(df, cols)
    point.add_continuous_variable("v", values)

    ind_df = pd.DataFrame(np.random.uniform(0, 100, (n_ip, n_dim)), columns=cols)
    inducing = geoml.data.PointData(ind_df, cols)
    root = geoml.latent.BasicInput(inducing, transform=geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(verbose=False, seed=seed, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), gp, options=options)
    model.train_full(max_iter=max_iter)
    return model


def _predict_mean(model, container, n_sim=4, batch_size=None):
    if batch_size is not None:
        model.options.prediction_batch_size = batch_size
    model.predict(container, n_sim=n_sim)
    return np.asarray(container.variables["v"].latent_mean.values)


# --------------------------------------------------------------------------- #
# standard containers (base batched-write path)
# --------------------------------------------------------------------------- #
def test_predict_grid1d():
    model = _synth_point_model(1)
    grid = geoml.data.Grid1D(start=0, n=30, step=3)
    mean = _predict_mean(model, grid)
    assert mean.shape == (grid.n_data,)
    assert np.all(np.isfinite(mean))


def test_predict_grid3d():
    model = _synth_point_model(3)
    grid = geoml.data.Grid3D(start=[0, 0, 0], n=[6, 6, 6], step=[18, 18, 18])
    mean = _predict_mean(model, grid)
    assert mean.shape == (grid.n_data,)
    assert np.all(np.isfinite(mean))


def test_predict_gridnd():
    model = _synth_point_model(4)
    grid = geoml.data.GridND(start=[0, 0, 0, 0], n=[4, 4, 4, 4], step=[30, 30, 30, 30])
    mean = _predict_mean(model, grid)
    assert mean.shape == (grid.n_data,)
    assert np.all(np.isfinite(mean))


def test_predict_section3d():
    model = _synth_point_model(3)
    section = geoml.data.Section3D(
        center=[50, 50, 50], azimuth=30, dip=20,
        width=80, height=60, n_x=12, n_y=10)
    mean = _predict_mean(model, section)
    assert mean.shape == (section.n_data,)
    assert np.all(np.isfinite(mean))


def test_predict_rotated_grid3d():
    model = _synth_point_model(3)
    grid = geoml.data.RotatedGrid3D(
        start=[0, 0, 0], n=[6, 6, 6], step=[15, 15, 15], azimuth=25, dip=15)
    mean = _predict_mean(model, grid)
    assert mean.shape == (grid.n_data,)
    assert np.all(np.isfinite(mean))


def test_predict_directional_data():
    model = _synth_point_model(2)
    n = 20
    df = pd.DataFrame({
        "c0": np.random.uniform(0, 100, n), "c1": np.random.uniform(0, 100, n),
        "dx": np.ones(n), "dy": np.zeros(n)})
    dirs = geoml.data.DirectionalData(df, ["c0", "c1"], ["dx", "dy"])
    mean = _predict_mean(model, dirs)
    assert mean.shape == (dirs.n_data,)
    assert np.all(np.isfinite(mean))


# --------------------------------------------------------------------------- #
# Blocks: discretization fans out sub-points but returns one row per block
# --------------------------------------------------------------------------- #
def test_predict_blocks1d_discretization():
    model = _synth_point_model(1)
    blocks = geoml.data.Blocks1D(start=0, n=20, step=5, discretization=[4])
    mean = _predict_mean(model, blocks)
    assert mean.shape == (blocks.n_data,)          # one value per block, not x4
    assert np.all(np.isfinite(mean))


def test_predict_blocks2d_discretization_is_batch_invariant():
    model = _synth_point_model(2)
    blocks = geoml.data.Blocks2D(
        start=[0, 0], n=[6, 6], step=[15, 15], discretization=[3, 3])

    one = _predict_mean(model, blocks, batch_size=10 ** 9).copy()   # single batch
    many = _predict_mean(model, blocks, batch_size=5).copy()        # ~8 batches

    assert one.shape == (blocks.n_data,)           # 36 blocks, not 36*9 points
    assert np.all(np.isfinite(one))
    assert np.allclose(one, many, atol=1e-8)


def test_predict_blocks3d_discretization():
    model = _synth_point_model(3)
    blocks = geoml.data.Blocks3D(
        start=[0, 0, 0], n=[5, 5, 5], step=[18, 18, 18], discretization=[2, 2, 2])
    mean = _predict_mean(model, blocks)
    assert mean.shape == (blocks.n_data,)
    assert np.all(np.isfinite(mean))


def test_blocks_discretization_on_zarr_backend(monkeypatch):
    """Discretization + on-disk store + multi-batch must stay batch-invariant."""
    import geoml.storage as storage

    model = _synth_point_model(2)
    blocks = geoml.data.Blocks2D(
        start=[0, 0], n=[6, 6], step=[15, 15], discretization=[3, 3])

    ref = _predict_mean(model, blocks, batch_size=10 ** 9).copy()

    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    model2 = _synth_point_model(2)
    blocks2 = geoml.data.Blocks2D(
        start=[0, 0], n=[6, 6], step=[15, 15], discretization=[3, 3])
    got = _predict_mean(model2, blocks2, batch_size=5)

    assert blocks2.variables["v"].simulations.backend == "zarr"
    assert got.shape == (blocks2.n_data,)
    assert np.all(np.isfinite(got))
    assert np.allclose(ref, got, atol=1e-8)


# --------------------------------------------------------------------------- #
# variable types survive being copied to another container
# --------------------------------------------------------------------------- #
def _container_with_every_variable_type(n=12, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, 2))
    cols = ["X", "Y"]
    point = geoml.data.PointData(pd.DataFrame(coords, columns=cols), cols)

    rock = np.array(["granite", "basalt"] * (n // 2))
    point.add_continuous_variable("continuous", rng.normal(size=n))
    point.add_vector_variable("vector", ["a", "b"], rng.normal(size=(n, 2)))
    point.add_rock_type_variable("rock_type", labels=("granite", "basalt"),
                                 measurements_a=rock)
    point.add_categorical_variable("categorical", ("granite", "basalt"), rock)
    point.add_rock_type_variable("ordered", labels=("granite", "basalt"),
                                 measurements_a=rock, ordered=True)
    point.add_binary_variable("binary", labels=("basalt", "granite"),
                              measurements=rock)
    point.add_anomaly_variable("anomaly", "granite", measurements=rock)
    point.add_compositional_variable(
        "composition", ["s", "t"],
        np.tile([0.4, 0.6], (n, 1)))
    return point


@pytest.mark.parametrize("name", [
    "continuous", "vector", "rock_type", "categorical", "ordered", "binary",
    "anomaly", "composition"])
def test_copy_to_keeps_the_variable_subclass(name):
    """`predict` builds its output variable with `copy_to`, which used to
    downgrade subclasses to their base (a CategoricalVariable came out as a
    RockTypeVariable, an AnomalyVariable as a BinaryVariable)."""
    source = _container_with_every_variable_type()
    target = geoml.data.Grid2D(start=[0, 0], n=[4, 4], step=[10, 10])

    variable = source.variables[name]
    variable.copy_to(target)
    copied = target.variables[name]

    assert type(copied) is type(variable)
    assert copied.name == variable.name
    assert copied.length == variable.length
    assert copied.coordinates is target

    if hasattr(variable, "labels"):
        assert list(copied.labels) == list(variable.labels)
    if hasattr(variable, "components"):
        assert list(copied.components.keys()) == list(variable.components.keys())
