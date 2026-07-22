# geoML - machine learning models for geospatial data
# Copyright (C) 2019  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Array storage backend for data objects.

``ArrayStore`` wraps a single array that is held either in RAM (NumPy) or on
disk in chunks (Zarr), and exposes a small, NumPy-compatible surface so callers
can treat it like an ``ndarray`` regardless of where the data lives:

- region reads/writes: ``store[idx]`` and ``store[idx] = value`` touch only the
  affected chunks, which is what the batched prediction write path needs;
- full materialization on demand: ``numpy.asarray(store)`` (used implicitly by
  ``numpy`` ufuncs, ``reshape``, filters, ...);
- a lazy labelled view: :meth:`ArrayStore.as_xarray` (dask-backed for Zarr) for
  out-of-core reductions and export.

The backend is chosen by size: arrays whose in-RAM footprint would exceed
``DEFAULT_THRESHOLD`` spill to a chunked Zarr array, everything else stays in
NumPy. This module is deliberately independent of ``data.py`` so it can be
tested in isolation.
"""

__all__ = ["ArrayStore", "DEFAULT_THRESHOLD"]

import os as _os
import shutil as _shutil
import tempfile as _tempfile

import numpy as _np
import zarr as _zarr
import dask.array as _da
import xarray as _xr


# Arrays whose uncompressed in-RAM size would exceed this many bytes are stored
# on disk (Zarr) rather than in NumPy. Keeps ordinary point data in RAM while
# large grids / simulation cubes spill to disk.
DEFAULT_THRESHOLD = 50 * 1024 ** 2  # 50 MB

# Target uncompressed size of a single Zarr chunk. Chunking is done along the
# leading (data-location) axis only, so batched writes align to whole chunks.
_TARGET_CHUNK_BYTES = 8 * 1024 ** 2  # 8 MB


def _leading_chunk(shape, dtype):
    """Chunk shape that splits only axis 0, targeting ``_TARGET_CHUNK_BYTES``."""
    itemsize = _np.dtype(dtype).itemsize
    trailing = int(_np.prod(shape[1:])) if len(shape) > 1 else 1
    row_bytes = max(trailing * itemsize, 1)
    rows = max(1, _TARGET_CHUNK_BYTES // row_bytes)
    if shape[0] > 0:
        rows = min(rows, shape[0])
    return (int(rows),) + tuple(int(s) for s in shape[1:])


def _use_zarr(shape, dtype, threshold):
    """Whether an array of this shape/dtype should live on disk."""
    if _np.dtype(dtype) == object:
        # object arrays (categorical labels, ...) stay in NumPy; Zarr would need
        # a variable-length codec and they are small in practice.
        return False
    nbytes = int(_np.prod(shape)) * _np.dtype(dtype).itemsize
    return nbytes > threshold


class ArrayStore:
    """A single array backed by NumPy (in RAM) or Zarr (on disk, chunked)."""

    def __init__(self, array, backend, store_path=None, _tempdir=None):
        # Low-level constructor; prefer the ``from_numpy`` / ``allocate`` /
        # ``open`` factories.
        self._array = array
        self._backend = backend          # "numpy" or "zarr"
        self._store_path = store_path     # on-disk location, if any
        self._tempdir = _tempdir          # root to clean up when we own it

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_numpy(cls, values):
        """Wrap an existing array in a NumPy-backed store (no copy)."""
        return cls(_np.asarray(values), backend="numpy")

    @classmethod
    def allocate(cls, shape, dtype=float, fill_value=_np.nan, chunks=None,
                 backend="auto", store=None, threshold=None):
        """Create a new, filled array.

        Parameters
        ----------
        shape : tuple
            Full array shape; axis 0 is the data-location axis.
        dtype : data-type
        fill_value : scalar
            Initial value for every element (``nan`` by default).
        chunks : tuple, optional
            Zarr chunk shape. Defaults to splitting axis 0 only.
        backend : {"auto", "numpy", "zarr"}
            ``"auto"`` picks Zarr past ``threshold`` bytes, NumPy otherwise.
        store : str or zarr store, optional
            Where a Zarr array lives. If omitted, a temporary directory is
            created and cleaned up with this object.
        threshold : int
            Size in bytes above which ``"auto"`` chooses Zarr.
        """
        shape = tuple(int(s) for s in _np.atleast_1d(shape))
        if threshold is None:
            # Read at call time so the module-level default stays configurable.
            threshold = DEFAULT_THRESHOLD

        if backend == "auto":
            backend = "zarr" if _use_zarr(shape, dtype, threshold) else "numpy"

        if backend == "numpy":
            return cls(_np.full(shape, fill_value, dtype=dtype), backend="numpy")

        if backend != "zarr":
            raise ValueError(f"unknown backend '{backend}'")

        if chunks is None:
            chunks = _leading_chunk(shape, dtype)

        tempdir = None
        if store is None:
            tempdir = _tempfile.mkdtemp(prefix="geoml_zarr_")
            store = _os.path.join(tempdir, "array.zarr")
        store_path = store if isinstance(store, str) else getattr(store, "path", None)

        array = _zarr.create_array(
            store=store, shape=shape, chunks=chunks,
            dtype=_np.dtype(dtype), fill_value=fill_value)
        return cls(array, backend="zarr", store_path=store_path, _tempdir=tempdir)

    @classmethod
    def open(cls, path, mode="r+"):
        """Reopen an existing on-disk Zarr array."""
        return cls(_zarr.open_array(path, mode=mode), backend="zarr",
                   store_path=path)

    @classmethod
    def wrap_zarr(cls, zarr_array):
        """Wrap an already-open Zarr array (e.g. a child of a reopened group)."""
        return cls(zarr_array, backend="zarr",
                   store_path=getattr(zarr_array, "store_path", None))

    def write_into(self, group, name):
        """Stream this store into a new array ``name`` of an open Zarr group.

        The copy is chunk-by-chunk via dask, so a large on-disk source is never
        fully materialized. Returns the created Zarr array.
        """
        if self._backend == "zarr":
            chunks = self._array.chunks
        else:
            chunks = _leading_chunk(self.shape, self.dtype)
        fill = _np.nan if _np.issubdtype(_np.dtype(self.dtype), _np.floating) else 0
        target = group.create_array(
            name=name, shape=self.shape, chunks=chunks,
            dtype=_np.dtype(self.dtype), fill_value=fill)
        _da.store(self.as_dask(), target, lock=False)
        return target

    # ------------------------------------------------------------------ #
    # ndarray-compatible surface
    # ------------------------------------------------------------------ #
    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, item, value):
        self._array[item] = value

    def __array__(self, dtype=None, copy=None):
        if self._backend == "zarr":
            array = _np.asarray(self._array[...])
            made_copy = True
        else:
            array = _np.asarray(self._array)
            made_copy = False
        if dtype is not None and array.dtype != _np.dtype(dtype):
            array = array.astype(dtype)
            made_copy = True
        if copy and not made_copy:
            array = array.copy()
        return array

    @property
    def shape(self):
        return tuple(self._array.shape)

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def ndim(self):
        return len(self._array.shape)

    @property
    def size(self):
        return int(_np.prod(self._array.shape))

    def __len__(self):
        return int(self._array.shape[0])

    def copy(self):
        """Materialize to a fresh NumPy array."""
        return _np.array(self.__array__())

    def ravel(self):
        return self.__array__().ravel()

    def to_numpy(self):
        return self.__array__()

    def __eq__(self, other):
        return self.__array__() == other

    def __ne__(self, other):
        return self.__array__() != other

    __hash__ = None

    def __repr__(self):
        return f"ArrayStore(backend={self._backend!r}, shape={self.shape}, " \
               f"dtype={self.dtype})"

    def __copy__(self):
        # Copies are always independent NumPy-backed stores: sharing a Zarr
        # array (and its temp directory) across stores would risk double-free.
        return ArrayStore.from_numpy(self.__array__().copy())

    def __deepcopy__(self, memo):
        return ArrayStore.from_numpy(self.__array__().copy())

    # ------------------------------------------------------------------ #
    # labelled / lazy views
    # ------------------------------------------------------------------ #
    def as_dask(self):
        """A dask array view (lazy & chunked for Zarr, single-chunk for NumPy)."""
        if self._backend == "zarr":
            return _da.from_array(self._array, chunks=self._array.chunks)
        return _da.from_array(self._array, chunks=-1)

    def as_xarray(self, dims=None, coords=None, name=None):
        """A labelled ``xarray.DataArray`` over this store (dask-backed)."""
        return _xr.DataArray(self.as_dask(), dims=dims, coords=coords, name=name)

    # ------------------------------------------------------------------ #
    # backend info / lifecycle
    # ------------------------------------------------------------------ #
    @property
    def backend(self):
        return self._backend

    @property
    def store_path(self):
        return self._store_path

    def close(self):
        """Release the array and delete the temp store if we created it."""
        self._array = None
        if self._tempdir is not None and _os.path.isdir(self._tempdir):
            _shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
