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


def test_scratch_consolidation_and_cleanup():
    """Arrays of the same owner share one scratch store, deleted with it."""
    import gc
    import os

    class Owner:
        pass

    owner = Owner()
    a = ArrayStore.allocate((50,), backend="zarr", owner=owner)
    b = ArrayStore.allocate((50, 3), backend="zarr", owner=owner)

    assert a.store_path == b.store_path            # one consolidated store
    root = a.store_path
    assert os.path.exists(root)

    a[:] = 1.0
    b[:] = 2.0
    assert np.all(np.asarray(a) == 1.0)
    assert np.all(np.asarray(b) == 2.0)

    other = Owner()
    c = ArrayStore.allocate((50,), backend="zarr", owner=other)
    assert c.store_path != root                    # per-owner isolation

    del a, b, owner
    gc.collect()
    assert not os.path.exists(root)                # gone with the owner
    assert os.path.exists(c.store_path)            # other owner unaffected


def test_row_bands_cover_every_row_once_and_hold_whole_chunks(tmp_path):
    reference = np.random.random((40, 16))
    sims = ArrayStore.allocate((40, 16), backend="zarr",
                               store=str(tmp_path / "b.zarr"),
                               chunks=(7, 16))
    sims[:] = reference

    bands = sims.row_bands()
    assert len(bands) == 6                              # 40 rows, 7 to a chunk
    assert [(b.start, b.stop) for b in bands][:2] == [(0, 7), (7, 14)]
    assert bands[-1].stop == 40                         # the short last one
    assert np.allclose(np.concatenate([sims[b] for b in bands]), reference)

    # a band is what a reduction across simulations reads at once, so it has to
    # hold every column of the rows it covers
    assert all(sims[b].shape[1] == 16 for b in bands)


def test_row_bands_of_a_numpy_store_are_a_single_band():
    """Data already in RAM has nothing to stream, so a caller written to band
    it pays nothing for being written that way."""
    store = ArrayStore.from_numpy(np.random.random((40, 16)))
    assert store.row_bands() == [slice(0, 40)]


def test_row_bands_can_be_asked_for_a_size(tmp_path):
    sims = ArrayStore.allocate((40, 16), backend="zarr",
                               store=str(tmp_path / "s.zarr"), chunks=(7, 16))
    assert len(sims.row_bands(rows=20)) == 2
    assert len(sims.row_bands(rows=1)) == 40
    assert sims.row_bands(rows=0) == sims.row_bands(rows=1)   # never empty


def test_row_quantiles_matches_numpy(tmp_path):
    reference = np.random.random((40, 16))
    sims = ArrayStore.allocate((40, 16), backend="zarr",
                               store=str(tmp_path / "q.zarr"),
                               chunks=(7, 16))     # multiple row-chunks
    sims[:] = reference

    got = sims.row_quantiles([0.1, 0.5, 0.9]).compute()
    want = np.quantile(reference, [0.1, 0.5, 0.9], axis=1).T
    assert got.shape == (40, 3)
    assert np.allclose(got, want)


def test_row_cdf_matches_numpy_and_inverts_quantiles(tmp_path):
    reference = np.random.random((40, 16))
    sims = ArrayStore.allocate((40, 16), backend="zarr",
                               store=str(tmp_path / "cdf.zarr"),
                               chunks=(7, 16))
    sims[:] = reference

    cutoffs = [0.25, 0.5, 0.75]
    got = sims.row_cdf(cutoffs).compute()
    want = np.stack([np.mean(reference <= c, axis=1) for c in cutoffs], axis=1)
    assert got.shape == (40, 3)
    assert np.allclose(got, want)
    assert np.all((got >= 0.0) & (got <= 1.0))

    # inverse relationship: at least a p-fraction of simulations lie at or
    # below the p-quantile of each row
    p = 0.5
    q = sims.row_quantiles([p]).compute()[:, 0]
    cdf_at_q = np.mean(reference <= q[:, None], axis=1)
    assert np.all(cdf_at_q >= p)


def test_store_columns_mixed_backends(tmp_path):
    from geoml.storage import store_columns

    reference = np.random.random((30, 8))
    sims = ArrayStore.allocate((30, 8), backend="zarr",
                               store=str(tmp_path / "src.zarr"))
    sims[:] = reference

    numpy_target = ArrayStore.allocate((30,), backend="numpy")
    zarr_target = ArrayStore.allocate((30,), backend="zarr",
                                      store=str(tmp_path / "tgt.zarr"))
    columns = sims.row_quantiles([0.25, 0.75])
    store_columns(columns, [numpy_target, zarr_target])

    want = np.quantile(reference, [0.25, 0.75], axis=1).T
    assert np.allclose(np.asarray(numpy_target), want[:, 0])
    assert np.allclose(np.asarray(zarr_target), want[:, 1])
