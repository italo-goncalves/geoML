"""Unit tests for the ArrayStore storage backend.

These prove that the NumPy- and Zarr-backed variants are interchangeable
through the ndarray-compatible surface that ``data.py`` relies on (region
reads/writes, ``numpy`` interop, reshaping), that large arrays spill to disk,
and that the batched-write and out-of-core reduction paths behave correctly.
"""
import numpy as np
import pytest

from geoml.storage import ArrayStore, DEFAULT_THRESHOLD


def test_numpy_backend_roundtrip():
    store = ArrayStore.from_numpy(np.arange(10, dtype=float))
    assert store.backend == "numpy"
    assert store.shape == (10,)
    store[2:5] = -1.0
    assert np.array_equal(store[2:5], [-1.0, -1.0, -1.0])
    assert np.asarray(store)[0] == 0.0


def test_zarr_backend_region_write_read(tmp_path):
    store = ArrayStore.allocate((100,), backend="zarr",
                                store=str(tmp_path / "a.zarr"))
    assert store.backend == "zarr"
    assert np.isnan(store[0])                      # fill value
    store[10:20] = np.arange(10, dtype=float)
    assert np.array_equal(store[10:20], np.arange(10, dtype=float))
    assert np.isnan(np.asarray(store)[50])         # untouched region


def test_backends_are_interchangeable(tmp_path):
    data = np.random.random((60, 4))
    npy = ArrayStore.allocate((60, 4), backend="numpy")
    zrr = ArrayStore.allocate((60, 4), backend="zarr",
                              store=str(tmp_path / "b.zarr"))
    for s in (npy, zrr):
        s[:20] = data[:20]
        s[20:] = data[20:]
    assert np.allclose(np.asarray(npy), data)
    assert np.allclose(np.asarray(npy), np.asarray(zrr))


def test_auto_backend_selects_by_size():
    small = ArrayStore.allocate((100,), backend="auto", threshold=DEFAULT_THRESHOLD)
    assert small.backend == "numpy"
    # force the boundary low so a small array is considered "large"
    big = ArrayStore.allocate((10_000,), backend="auto", threshold=1024)
    assert big.backend == "zarr"


def test_object_dtype_stays_numpy():
    store = ArrayStore.allocate((5,), dtype=object, backend="auto", threshold=1)
    assert store.backend == "numpy"


def test_reopen_persists(tmp_path):
    path = str(tmp_path / "persist.zarr")
    store = ArrayStore.allocate((50, 3), backend="zarr", store=path)
    store[10:20] = 7.0

    reopened = ArrayStore.open(path)
    assert reopened.shape == (50, 3)
    assert np.all(reopened[10:20] == 7.0)
    assert np.isnan(reopened[0, 0])


def test_batched_simulation_writes(tmp_path):
    """Mirrors the predict() write loop: (n_data, n_sim) filled by row-batch."""
    n_data, n_sim = 73, 8
    sims = ArrayStore.allocate((n_data, n_sim), backend="zarr",
                               store=str(tmp_path / "sims.zarr"))
    reference = np.random.random((n_data, n_sim))

    batch = 16
    for start in range(0, n_data, batch):
        idx = np.arange(start, min(start + batch, n_data))
        sims[idx, :] = reference[idx, :]

    assert np.allclose(np.asarray(sims), reference)


def test_numpy_compat_ops(tmp_path):
    store = ArrayStore.allocate((40,), backend="zarr",
                                store=str(tmp_path / "c.zarr"))
    store[:] = np.arange(40, dtype=float)

    assert len(store) == 40
    assert store.dtype == np.float64
    assert not np.any(np.isnan(store))             # ufunc via __array__
    assert np.reshape(store, (5, 8)).shape == (5, 8)
    col = store.copy()[:, None]                    # measurements pattern
    assert col.shape == (40, 1)

    obj = ArrayStore.from_numpy(np.array(["", "a", ""], dtype=object))
    assert not all(obj == "")                       # pyvista-fill pattern


def test_as_xarray_quantile_matches_numpy(tmp_path):
    reference = np.random.random((30, 12))
    sims = ArrayStore.allocate((30, 12), backend="zarr",
                               store=str(tmp_path / "d.zarr"))
    sims[:] = reference

    da = sims.as_xarray(dims=("point", "simulation"))
    q = da.quantile(0.5, dim="simulation").to_numpy()
    assert np.allclose(q, np.quantile(reference, 0.5, axis=1))


def test_temp_store_is_cleaned_up():
    store = ArrayStore.allocate((2000,), backend="zarr", threshold=1)
    import os
    path = store.store_path
    assert path is not None and os.path.exists(path)
    store.close()
    assert not os.path.exists(path)
