"""Coordinate/variance storage and the batched-variance contract.

Point coordinates and ``GaussianData`` variance live in ``ArrayStore``s, so they
spill to Zarr when large, like variable arrays do. ``get_batched_variance`` must
mirror ``get_batched_coordinates`` for every container: same number of rows, same
aggregation, and O(batch) work -- it used to rebuild the whole coordinate array
on every batch, which made prediction quadratic in the number of points.
"""
import numpy as np
import pandas as pd
import pytest

import geoml
import geoml.storage as storage

from test_containers import _synth_point_model


def _points(n=30, n_dim=3, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(n_dim)]
    df = pd.DataFrame(rng.uniform(0, 100, (n, n_dim)), columns=cols)
    return geoml.data.PointData(df, cols), df


# --------------------------------------------------------------------------- #
# batched variance mirrors batched coordinates
# --------------------------------------------------------------------------- #
def _containers():
    return [
        _points(25, 3)[0],
        geoml.data.Grid1D(start=0, n=40, step=2),
        geoml.data.Grid2D(start=[0, 0], n=[8, 9], step=[1, 1]),
        geoml.data.Grid3D(start=[0, 0, 0], n=[4, 5, 6], step=[1, 1, 1]),
        geoml.data.RotatedGrid3D(start=[0, 0, 0], n=[4, 4, 4], step=[2, 2, 2],
                                 azimuth=20, dip=10),
        geoml.data.Blocks2D(start=[0, 0], n=[5, 5], step=[1, 1],
                            discretization=[3, 2]),
        geoml.data.Blocks3D(start=[0, 0, 0], n=[3, 3, 3], step=[1, 1, 1],
                            discretization=[2, 2, 2]),
    ]


@pytest.mark.parametrize("container", _containers(),
                         ids=lambda c: type(c).__name__)
def test_batched_variance_matches_batched_coordinates(container):
    for index in (np.arange(7), None):
        coords, splits = container.get_batched_coordinates(index)
        var, var_splits = container.get_batched_variance(index)

        assert var.shape == np.asarray(coords).shape
        assert var_splits == splits
        assert not np.any(var)            # zero without an explicit variance


def test_batched_variance_accepts_boolean_mask():
    point, _ = _points(20, 2)
    mask = np.zeros(20, dtype=bool)
    mask[[1, 5, 9]] = True

    coords, _ = point.get_batched_coordinates(mask)
    var, _ = point.get_batched_variance(mask)
    assert var.shape == np.asarray(coords).shape == (3, 2)


def test_batched_variance_is_independent_of_container_size():
    """The whole point of the fix: cost must not scale with n_data."""
    import time

    batch = np.arange(5000)
    times = []
    for n in (30, 120):        # 27k vs 1.7M nodes
        grid = geoml.data.Grid3D(start=[0, 0, 0], n=[n, n, n], step=[1, 1, 1])
        grid.get_batched_variance(batch)                   # warm up
        start = time.perf_counter()
        for _ in range(20):
            grid.get_batched_variance(batch)
        times.append(time.perf_counter() - start)

    # a 64x larger grid must not cost meaningfully more per batch
    assert times[1] < 5 * times[0] + 1e-3


# --------------------------------------------------------------------------- #
# coordinates in a store
# --------------------------------------------------------------------------- #
def test_coordinates_are_stored_and_match_the_input():
    point, df = _points(30, 3)
    assert isinstance(point.coordinates, storage.ArrayStore)
    assert point.coordinates.backend == "numpy"       # small: stays in RAM
    assert np.allclose(np.asarray(point.coordinates), df.to_numpy())
    assert point.n_data == 30 and point.n_dim == 3


def test_large_coordinates_spill_to_zarr(monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    point, df = _points(30, 3)

    assert point.coordinates.backend == "zarr"
    assert np.allclose(np.asarray(point.coordinates), df.to_numpy())
    # bounding box is computed from the store without materializing it
    assert np.allclose(point.bounding_box.min[0], df.to_numpy().min(axis=0))
    assert np.allclose(point.bounding_box.max[0], df.to_numpy().max(axis=0))
    # and the usual ndarray-ish access still works
    assert point.coordinates[np.arange(4)].shape == (4, 3)
    assert point.coordinates[:, 1].shape == (30,)
    assert np.allclose(point.as_data_frame().to_numpy(), df.to_numpy())


def test_subset_of_stored_coordinates(monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    point, df = _points(30, 3)
    point.add_continuous_variable("v", np.arange(30.0))

    sub = point[np.arange(5, 15)]
    assert isinstance(sub.coordinates, storage.ArrayStore)
    assert sub.n_data == 10
    assert np.allclose(np.asarray(sub.coordinates), df.to_numpy()[5:15])
    assert np.allclose(np.asarray(sub.variables["v"].measurements.values),
                       np.arange(5, 15))


def test_predict_into_stored_point_cloud(monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    model = _synth_point_model(2)
    target, _ = _points(24, 2, seed=3)

    assert target.coordinates.backend == "zarr"
    model.predict(target, n_sim=3)
    got = np.asarray(target.variables["v"].latent_mean.values)
    assert got.shape == (24,) and np.all(np.isfinite(got))


# --------------------------------------------------------------------------- #
# GaussianData
# --------------------------------------------------------------------------- #
def _gaussian(n=20, n_dim=2, seed=1):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, (n, n_dim))
    variance = rng.uniform(0.1, 2.0, (n, n_dim))
    return geoml.data.GaussianData.from_array(coords, variance), coords, variance


def test_gaussian_data_variance_is_stored():
    gauss, coords, variance = _gaussian()
    assert isinstance(gauss.variance, storage.ArrayStore)
    assert np.allclose(np.asarray(gauss.variance), variance)
    assert np.allclose(np.asarray(gauss.coordinates), coords)


def test_gaussian_variance_spills_to_zarr(monkeypatch):
    monkeypatch.setattr(storage, "DEFAULT_THRESHOLD", 0)
    gauss, _, variance = _gaussian()
    assert gauss.variance.backend == "zarr"
    assert gauss.coordinates.backend == "zarr"
    assert np.allclose(np.asarray(gauss.variance), variance)


def test_gaussian_batched_variance_returns_the_batch():
    gauss, _, variance = _gaussian(20)
    batch = np.arange(3, 11)

    var, splits = gauss.get_batched_variance(batch)
    coords, _ = gauss.get_batched_coordinates(batch)
    assert splits is None
    assert np.allclose(var, variance[batch])
    assert var.shape == np.asarray(coords).shape

    full, _ = gauss.get_batched_variance()
    assert np.allclose(full, variance)


def test_gaussian_data_rejects_mismatched_variance():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        geoml.data.GaussianData.from_array(rng.uniform(size=(10, 2)),
                                           rng.uniform(size=(10, 3)))


def test_gaussian_subset_keeps_class_and_variance():
    """Subsetting used to fall back to PointData and drop the variance."""
    gauss, coords, variance = _gaussian(20)
    gauss.add_continuous_variable("v", np.arange(20.0))

    sub = gauss[np.arange(4, 12)]
    assert isinstance(sub, geoml.data.GaussianData)
    assert np.allclose(np.asarray(sub.variance), variance[4:12])
    assert np.allclose(np.asarray(sub.coordinates), coords[4:12])
    assert np.allclose(np.asarray(sub.variables["v"].measurements.values),
                       np.arange(4, 12))

    region = gauss.subset_region([0, 0], [50, 50])
    assert isinstance(region, geoml.data.GaussianData)
    assert region.variance.shape == (region.n_data, region.n_dim)


def test_gaussian_as_data_frame_includes_variance():
    gauss, coords, variance = _gaussian(12)
    df = gauss.as_data_frame()
    assert list(df.columns) == ["X", "Y", "X_var", "Y_var"]
    assert np.allclose(df[["X_var", "Y_var"]].to_numpy(), variance)


def test_predict_into_gaussian_data():
    """Predicting into a GaussianData target used to crash on unpacking."""
    model = _synth_point_model(2)
    gauss, _, _ = _gaussian(15, 2, seed=7)

    model.predict(gauss, n_sim=3)
    got = np.asarray(gauss.variables["v"].latent_mean.values)
    assert got.shape == (15,) and np.all(np.isfinite(got))


def test_predict_into_gaussian_data_is_batch_invariant():
    model = _synth_point_model(2)
    gauss, _, _ = _gaussian(15, 2, seed=7)

    model.options.prediction_batch_size = 10 ** 9
    model.predict(gauss, n_sim=3)
    one = np.array(gauss.variables["v"].latent_mean.values, copy=True)

    model.options.prediction_batch_size = 4
    model.predict(gauss, n_sim=3)
    many = np.array(gauss.variables["v"].latent_mean.values, copy=True)

    assert np.allclose(one, many, atol=1e-8)


def test_variance_handed_to_the_model_is_the_batch_variance():
    """Each batch must receive its own rows of the variance, not the whole array.

    (Whether the network then uses them is a separate matter: a network rooted
    at ``BasicInput`` currently drops ``x_var`` in ``propagate``.)
    """
    model = _synth_point_model(2)
    gauss, _, variance = _gaussian(15, 2, seed=7)

    seen = []
    original = model.predict_raw

    def spy(x, variable_inputs, x_var=None, **kwargs):
        seen.append(np.asarray(x_var))
        return original(x, variable_inputs, x_var=x_var, **kwargs)

    model.predict_raw = spy
    model.options.prediction_batch_size = 4
    model.predict(gauss, n_sim=3)

    assert len(seen) == 4                        # 15 points in batches of 4
    assert np.allclose(np.concatenate(seen, axis=0), variance)

    # a plain PointData still gets zeros of the right shape
    point, _ = _points(15, 2, seed=7)
    seen.clear()
    model.predict(point, n_sim=3)
    assert np.concatenate(seen, axis=0).shape == (15, 2)
    assert not np.any(np.concatenate(seen, axis=0))


def test_gaussian_data_roundtrip_keeps_variance_on_disk(tmp_path):
    gauss, coords, variance = _gaussian(12)
    gauss.add_continuous_variable("v", np.arange(12.0))

    path = str(tmp_path / "gauss.zarr")
    gauss.to_zarr(path)
    reloaded = geoml.data.PointData.open(path)

    assert isinstance(reloaded, geoml.data.GaussianData)
    assert reloaded.coordinates.backend == "zarr"
    assert reloaded.variance.backend == "zarr"
    assert np.allclose(np.asarray(reloaded.coordinates), coords)
    assert np.allclose(np.asarray(reloaded.variance), variance)
    assert np.allclose(np.asarray(reloaded.variables["v"].measurements.values),
                       np.arange(12))
