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


def test_gaussian_data_roundtrip(tmp_path):
    coords = np.random.uniform(0, 100, (12, 2))
    variance = np.random.random((12, 2))
    gauss = geoml.data.GaussianData.from_array(coords, variance, ["c0", "c1"])
    gauss.add_continuous_variable("v", np.random.random(12))

    path = str(tmp_path / "gauss.zarr")
    gauss.to_zarr(path)
    reloaded = geoml.data.PointData.open(path)

    assert isinstance(reloaded, geoml.data.GaussianData)
    assert np.allclose(reloaded.coordinates, gauss.coordinates)
    assert np.allclose(reloaded.variance, variance)
    assert np.allclose(
        np.asarray(reloaded.variables["v"].measurements.values),
        np.asarray(gauss.variables["v"].measurements.values))


def test_directional_data_roundtrip_and_predict(tmp_path):
    model, _ = _model_and_grid()
    n = 15
    df = pd.DataFrame({
        "c0": np.random.uniform(0, 100, n),
        "c1": np.random.uniform(0, 100, n),
        "dx": np.ones(n), "dy": np.zeros(n)})
    dirs = geoml.data.DirectionalData(df, ["c0", "c1"], ["dx", "dy"])
    model.predict(dirs, n_sim=3)
    mean = np.asarray(dirs.variables["v"].latent_mean.values).copy()

    path = str(tmp_path / "dirs.zarr")
    dirs.to_zarr(path)
    reloaded = geoml.data.PointData.open(path)

    assert isinstance(reloaded, geoml.data.DirectionalData)
    assert reloaded.direction_labels == ["dx", "dy"]
    assert np.allclose(reloaded.directions, dirs.directions)
    assert np.allclose(
        np.asarray(reloaded.variables["v"].latent_mean.values), mean)

    model.predict(reloaded, n_sim=3)     # resume-predict must not raise
    assert np.all(np.isfinite(
        np.asarray(reloaded.variables["v"].latent_mean.values)))


def test_section3d_roundtrip(tmp_path):
    section = geoml.data.Section3D(
        center=[50, 50, 50], azimuth=30, dip=20,
        width=80, height=60, n_x=8, n_y=6)
    section.add_continuous_variable("v")
    section.variables["v"].prediction.values[:] = \
        np.random.random(section.n_data)
    pred = np.asarray(section.variables["v"].prediction.values).copy()

    path = str(tmp_path / "section.zarr")
    section.to_zarr(path)
    reloaded = geoml.data.PointData.open(path)

    assert isinstance(reloaded, geoml.data.Section3D)
    assert reloaded.grid_shape == [8, 6]
    assert np.allclose(reloaded.coordinates, section.coordinates)
    assert np.allclose(
        np.asarray(reloaded.variables["v"].prediction.values), pred)
    reloaded.as_pyvista()                # grid_shape-dependent path works


def test_surface3d_roundtrip(tmp_path):
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    normals = np.array([[0.0, 0, 1], [0, 0, 1]])
    surf = geoml.data.Surface3D(points, triangles, normals)
    surf.add_continuous_variable("v", np.random.random(4))

    path = str(tmp_path / "surf.zarr")
    surf.to_zarr(path)
    reloaded = geoml.data.PointData.open(path)

    assert isinstance(reloaded, geoml.data.Surface3D)
    assert np.allclose(reloaded.coordinates, points)
    assert np.array_equal(reloaded.triangles, triangles)
    assert np.allclose(reloaded.normals, normals)
    assert np.allclose(
        np.asarray(reloaded.variables["v"].measurements.values),
        np.asarray(surf.variables["v"].measurements.values))


def test_unsupported_variable_type_raises(tmp_path):
    point = geoml.data.PointData(
        pd.DataFrame({"c0": [1.0, 2.0, 3.0], "c1": [3.0, 4.0, 5.0]}),
        ["c0", "c1"])

    class _Unsupported(geoml.data._Variable):
        pass

    point.variables["u"] = _Unsupported("u", point)
    with pytest.raises(NotImplementedError):
        point.to_zarr(str(tmp_path / "u.zarr"))
