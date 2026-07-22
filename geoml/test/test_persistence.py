"""Round-trip tests for container persistence (to_zarr / open).

A predicted container is written to one on-disk Zarr store and reloaded into an
equivalent geoML container, with its ContinuousVariable arrays reopened on disk.
Reloaded values must match, the store must survive on both the NumPy and Zarr
backends, and the reloaded container must still be predictable into.
"""
import numpy as np
import pandas as pd
import pytest

import geoml


def _model_and_grid(seed=1234):
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cols = ["c0", "c1"]
    coords = np.random.uniform(0, 100, (40, 2))
    values = np.sin(coords[:, 0] / 25.0)
    df = pd.DataFrame(coords, columns=cols)
    point = geoml.data.PointData(df, cols)
    point.add_continuous_variable("v", values)
    ind = geoml.data.PointData(
        pd.DataFrame(np.random.uniform(0, 100, (6, 2)), columns=cols), cols)
    root = geoml.latent.BasicInput(ind, transform=geoml.transform.Isotropic(40))
    gp = geoml.latent.BasicGP(root, size=1, kernel=geoml.kernels.Gaussian())
    options = geoml.models.GPOptions(verbose=False, seed=seed, training_samples=8)
    model = geoml.models.VGPNetwork(
        point, "v", geoml.likelihood.Gaussian(), gp, options=options)
    model.train_full(max_iter=8)
    grid = geoml.data.Grid2D(start=[0, 0], n=[10, 10], step=[10, 10])
    return model, grid


def test_grid_roundtrip_values_match(tmp_path):
    model, grid = _model_and_grid()
    model.predict(grid, n_sim=5)
    grid.variables["v"].reset_quantiles([0.1, 0.9])

    mean = np.asarray(grid.variables["v"].latent_mean.values).copy()
    pred = np.asarray(grid.variables["v"].prediction.values).copy()
    sims = np.asarray(grid.variables["v"].simulations).copy()

    path = str(tmp_path / "run.zarr")
    grid.to_zarr(path)
    reloaded = geoml.data.Grid2D.open(path)

    assert isinstance(reloaded, geoml.data.Grid2D)
    assert reloaded.n_data == grid.n_data
    assert np.allclose(reloaded.grid_size, grid.grid_size)
    assert np.allclose(np.asarray(reloaded.variables["v"].latent_mean.values), mean)
    assert np.allclose(np.asarray(reloaded.variables["v"].prediction.values), pred)
    assert np.allclose(np.asarray(reloaded.variables["v"].simulations), sims)
    assert reloaded.variables["v"].simulations.backend == "zarr"       # on disk
    assert set(reloaded.variables["v"].quantiles.keys()) == {0.1, 0.9}


def test_roundtrip_survives_zarr_backend(tmp_path, monkeypatch):
    import geoml.storage as storage
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)   # force source on disk

    model, grid = _model_and_grid()
    model.predict(grid, n_sim=4)
    assert grid.variables["v"].prediction.values.backend == "zarr"
    mean = np.asarray(grid.variables["v"].latent_mean.values).copy()

    path = str(tmp_path / "run_zarr.zarr")
    grid.to_zarr(path)
    reloaded = geoml.data.Grid2D.open(path)
    assert np.allclose(np.asarray(reloaded.variables["v"].latent_mean.values), mean)


def test_pointdata_roundtrip_stores_coordinates(tmp_path):
    model, _ = _model_and_grid()
    target = geoml.data.PointData(
        pd.DataFrame(np.random.uniform(0, 100, (15, 2)), columns=["c0", "c1"]),
        ["c0", "c1"])
    model.predict(target, n_sim=3)
    coords = target.coordinates.copy()
    pred = np.asarray(target.variables["v"].prediction.values).copy()

    path = str(tmp_path / "pts.zarr")
    target.to_zarr(path)
    reloaded = geoml.data.PointData.open(path)

    assert np.allclose(reloaded.coordinates, coords)
    assert np.allclose(np.asarray(reloaded.variables["v"].prediction.values), pred)


def test_reloaded_container_is_predictable(tmp_path):
    """The reloaded container must accept a fresh prediction (arrays are r/w)."""
    model, grid = _model_and_grid()
    model.predict(grid, n_sim=4)

    path = str(tmp_path / "resume.zarr")
    grid.to_zarr(path)
    reloaded = geoml.data.Grid2D.open(path)

    model.predict(reloaded, n_sim=4)         # must not raise
    got = np.asarray(reloaded.variables["v"].latent_mean.values)
    assert np.all(np.isfinite(got))


def test_open_honours_stored_type_over_calling_class(tmp_path):
    model, grid = _model_and_grid()
    model.predict(grid, n_sim=3)
    path = str(tmp_path / "typed.zarr")
    grid.to_zarr(path)
    # Called on the base class, still returns the stored Grid2D.
    reloaded = geoml.data.PointData.open(path)
    assert isinstance(reloaded, geoml.data.Grid2D)


def test_unsupported_variable_type_raises(tmp_path):
    point = geoml.data.PointData(
        pd.DataFrame({"c0": [1.0, 2.0, 3.0], "c1": [3.0, 4.0, 5.0]}),
        ["c0", "c1"])

    class _Unsupported(geoml.data._Variable):
        pass

    point.variables["u"] = _Unsupported("u", point)
    with pytest.raises(NotImplementedError):
        point.to_zarr(str(tmp_path / "u.zarr"))
