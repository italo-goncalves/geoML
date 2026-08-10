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

__all__ = ["PointData",
           "Grid1D", "Grid2D", "Grid3D",
           "DirectionalData",
           "batch_index",
           "export_planes",
           "RotatedGrid3D", "Section3D",
           "Mesh3D", "Surface3D", "Solid3D", "DTM3D", "mesh3d",
           "Blocks1D", "Blocks2D", "Blocks3D"]


import tensorflow as _tf
import pandas as _pd
import numpy as _np
import copy as _copy
import collections as _col
import pyvista as _pv
import itertools as _iter
import sklearn.metrics as _skmetrics
# from astropy.coordinates.matrix_utilities import rotation_matrix

from skimage import measure as _measure
from skimage import filters as _filters

import ezdxf as _ezdxf
from ezdxf.render import MeshVertexMerger as _MeshVertexMerger

import geoml.interpolation as _gint
import geoml.plotly as _py
import geoml.tftools as _tftools
import geoml.metrics as _gmlmetrics
import geoml.geometry as _gmt
import geoml.storage as _storage

import dask.array as _da
import zarr as _zarr


class NoDataError(Exception):
    """Exception raised when a data object is empty."""
    pass


class NotGriddedDataError(Exception):
    """Exception raised when expecting a gridded data object."""
    pass


class NotClosedError(ValueError):
    """A mesh that does not bound a volume was asked to."""


class InconsistentMeshError(ValueError):
    """A closed mesh whose triangles disagree about which way is out."""


class NotSingleValuedError(ValueError):
    """A sheet that folds over was asked which of its heights to use."""


class MeshTypeError(ValueError):
    """Two meshes were combined in a way that means nothing."""


class DimensionMismatchError(Exception):
    """Exception raised when the dimensionality of objects does not match."""
    pass


def bounding_box(points):
    """
    Computes a point set's bounding box and its diagonal.

    Parameters
    ----------
    points : array
        A set of coordinates.

    Returns
    -------
    bbox : array-like
        Array with the box's minimum and maximum values in each direction.
    d : float
        The box's diagonal length.
    """
    if len(points.shape) < 2:
        points = _np.expand_dims(points, axis=0)
    bbox = _np.array([[_np.min(points[:, i]) for i in range(points.shape[1])],
                      [_np.max(points[:, i]) for i in range(points.shape[1])]])
    d = _np.sqrt(sum([
        _np.diff(bbox[:, i]) ** 2 for i in range(bbox.shape[1])]))
    d = _np.squeeze(d)
    return bbox, d


class BoundingBox(object):
    """
    An n-dimensional box.
    """
    def __init__(self, min_values, max_values):
        """
        An n-dimensional box.

        Parameters
        ----------
        min_values : array
            The box's minimum values in each direction.
        max_values : array
            The box's maximum values in each direction.
        """
        min_values = _np.array(min_values, ndmin=2)
        max_values = _np.array(max_values, ndmin=2)

        if min_values.shape != max_values.shape:
            raise ValueError("min_values and max_values must have the"
                             "same size")

        self._n_dim = min_values.shape[1]
        self._min = min_values
        self._max = max_values
        self._center = (max_values - min_values) / 2.0
        self._diagonal = _np.sqrt(_np.sum((max_values - min_values)**2))
        self.labels = None

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def center(self):
        return self._center

    def as_array(self):
        return _np.concatenate([self.min, self.max], axis=0)

    def as_data_frame(self):
        a = self.as_array()
        df = _pd.DataFrame(a, index=['min', 'max'])
        if self.labels:
            df.columns = self.labels
        return df

    def __repr__(self):
        return self.as_data_frame().__repr__()

    def overlaps_with(self, other):
        """
        Checks if box overlaps with another box.

        Parameters
        ----------
        other : BoundingBox
            The other box.

        Returns
        -------
        check : bool
            The checking result.
        """
        if other.n_dim != self.n_dim:
            raise ValueError("box dimensions mismatch")

        checks = []
        for min_A, max_A, min_B, max_B in zip(
                self.min[0], self.max[0], other.min[0], other.max[0]):
            checks.append(min_A > max_B)
            checks.append(min_B > max_A)
        return not any(checks)

    @classmethod
    def from_array(cls, array):
        """
        Builds bounding box from an array with minimum and maximum coordinates.

        Parameters
        ----------
        array : array
            Array with the minimum and maximum coordinates.

        Returns
        -------
        box : BoundingBox
            A box object.
        """
        if len(array.shape) < 2:
            array = _np.expand_dims(array, axis=0)
        min_values = _np.min(array, axis=0, keepdims=True)
        max_values = _np.max(array, axis=0, keepdims=True)
        return cls(min_values, max_values)

    def contains_points(self, array):
        """
        Checks if points are contained within the box.

        Parameters
        ----------
        array : array
            A set of coordinates.

        Returns
        -------
        check : bool
            True if all points are contained within the box.
        """
        if array.shape[1] != self.n_dim:
            raise ValueError("box and points dimensions mismatch")

        check_1 = array >= self.min
        check_2 = array <= self.max

        contains = _np.all(_np.concatenate([check_1, check_2], axis=1), axis=1)
        return contains


def _carry_rows(source, target, keep):
    """Copy the `keep` rows of one simulation store into the front of another.

    A band of the source at a time, so a store too large to hold is never
    asked for whole. The kept rows keep their order and land contiguously,
    which is what lets every write here be a plain slice.
    """
    index = _np.flatnonzero(keep)
    written = 0
    for band in source.row_bands():
        inside = index[(index >= band.start) & (index < band.stop)]
        if len(inside) == 0:
            continue
        chunk = _np.asarray(source[band])
        target[written:written + len(inside), :] = chunk[inside - band.start]
        written += len(inside)


def _filled(attribute):
    """Whether an attribute is there and holds anything worth exporting.

    Two ways it may not be: a component has no latent space of its own and
    says so with a `None`, and a column nothing ever wrote is all NaN. Asking
    both is what lets one `as_data_frame` serve a variable and a component
    alike, instead of a second copy that drifts a column at a time.
    """
    return attribute is not None and attribute._has_content()


def _export_label(prefix, name):
    """An exported column's name, carrying the owner's when there is one.

    A component is called `a`; on its own that says nothing about which assay
    it belongs to, and two variables may each hold one called `a`.
    """
    return name if prefix is None else prefix + " - " + name


PATH_SEP = "/"
METADATA_ROOT = "_metadata"


def _path_key(key):
    """A dict family's key as a path segment.

    `str` on a float is the shortest string that reads back as the same
    number, which is what makes `quantiles/1.5` and `quantiles/1.50` the same
    address without any tolerance being involved: both parse to `1.5`. A key
    that is not a number is its own segment.
    """
    try:
        return str(float(key))
    except (TypeError, ValueError):
        return str(key)


class VariablePath:
    """Where a piece of data sits inside a container.

    A container holds variables, a variable holds components or attributes,
    and an attribute holds one array per location. This names a place in that
    tree the way a file system names a file --
    ``assay/Zn/noise_variance`` -- so that one string can serve the lookup,
    the persistence key and the exported column name.

    Built from a string, from parts, or from another path; `/` composes, as it
    does for `pathlib`:

    >>> VariablePath("assay") / "Zn" / "prediction"
    VariablePath('assay/Zn/prediction')
    """
    __slots__ = ("parts",)

    def __init__(self, parts=()):
        if isinstance(parts, VariablePath):
            parts = parts.parts
        elif isinstance(parts, str):
            parts = tuple(p for p in parts.split(PATH_SEP) if p != "")
        else:
            parts = tuple(str(p) for p in parts)
        for part in parts:
            if PATH_SEP in part:
                raise ValueError(
                    "%r cannot be a path segment: %r is the separator"
                    % (part, PATH_SEP))
        self.parts = parts

    def __truediv__(self, other):
        return VariablePath(self.parts + VariablePath(other).parts)

    def __str__(self):
        return PATH_SEP.join(self.parts)

    def __repr__(self):
        return "VariablePath(%r)" % str(self)

    def __eq__(self, other):
        if isinstance(other, (str, tuple, list)):
            other = VariablePath(other)
        return isinstance(other, VariablePath) and self.parts == other.parts

    def __hash__(self):
        return hash(self.parts)

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return iter(self.parts)

    def __getitem__(self, item):
        return self.parts[item]

    @property
    def name(self):
        """The last segment, or `''` for the root."""
        return self.parts[-1] if self.parts else ""

    @property
    def parent(self):
        return VariablePath(self.parts[:-1])


class _TreeNode(object):
    """One traversal of the container tree, shared by every node in it.

    A container is a node, a variable is a node, a component is a node, and
    each of them holds *leaves* -- one array per location -- plus *attrs*, the
    facts describing the node itself. Everything that reads a container whole
    (persistence, the data frame, the pyvista exports, subsetting, carrying to
    a new container) is a fold over `leaves()`, so a column added in one place
    reaches all of them. Writing those lists out per class is what put the
    same bug in the code seven times.

    A subclass says what it holds by declaring `_ZARR_ATTRS` (scalar columns),
    `_DICT_FAMILIES` (columns keyed by a cut-off) and `_NODE_ATTRS` (the
    facts), and by answering `child_nodes`. Nothing else is per class.
    """
    _NODE_ATTRS = ()      # facts a rebuild must carry across; not arrays
    _DICT_FAMILIES = ()   # attribute families keyed by a cut-off
    _ZARR_ATTRS = ()

    @property
    def _node_name(self):
        return getattr(self, "name", "")

    def child_nodes(self):
        """The nodes directly beneath this one, by name."""
        return dict(getattr(self, "components", None) or {})

    def own_leaves(self):
        """`(parts, attribute)` for the columns this node owns directly.

        Not its children's, and not `simulations`: that one carries a
        realization axis, so it is a leaf of a different shape and is reached
        by name (`get`) rather than swept up by everything that walks columns.
        """
        for role in self._ZARR_ATTRS:
            attribute = getattr(self, role, None)
            if attribute is not None:
                yield (role,), attribute
        for family in self._DICT_FAMILIES:
            for key, attribute in (getattr(self, family, None) or {}).items():
                yield (family, _path_key(key)), attribute

    def node_attrs(self):
        """The facts describing this node, as `{name: value}`.

        Cut-offs and the like: not arrays, not rebuilt from anything else, and
        therefore invisible to any code that only knows about columns. Every
        method that builds a fresh variable has to carry them across, and the
        three that forgot are why they are enumerated here.
        """
        return {name: getattr(self, name, None) for name in self._NODE_ATTRS}

    def walk(self, prefix=None):
        """`(path, node)` for this node and every node beneath it."""
        prefix = VariablePath(self._node_name) if prefix is None \
            else VariablePath(prefix)
        yield prefix, self
        for name, child in self.child_nodes().items():
            for item in child.walk(prefix / name):
                yield item

    def leaves(self, prefix=None):
        """`(path, attribute)` for every column at or beneath this node."""
        for path, node in self.walk(prefix):
            for parts, attribute in node.own_leaves():
                yield path / parts, attribute

    def get(self, path, key=None):
        """The node or attribute at `path`, relative to this one.

        `container.get("assay/Zn/noise_variance")` is the column,
        `container.get("assay/Zn")` the component, `container.get("")` the
        container. A dict family takes its key as the next segment --
        `"assay/Zn/quantiles/1.5"` -- and `get(path, key)` is accepted as
        shorthand for the same thing.
        """
        path = VariablePath(path)
        if key is not None:
            path = path / _path_key(key)
        found = self._resolve(path)
        if found is None:
            # report from as deep as the path did reach: told that `au` holds
            # no `nope`, the reader knows where to look, where a list of the
            # container's variables only says the first segment was fine
            node = self._deepest(path)
            raise KeyError(
                "nothing at %r; %s holds %s"
                % (str(path),
                   "%r" % node._node_name if node._node_name
                   else "the container",
                   ", ".join(node._tree_names()) or "nothing"))
        return found

    def _deepest(self, path):
        """The last node `path` reaches before it stops matching."""
        node = self
        for segment in path:
            children = node.child_nodes()
            if segment not in children:
                return node
            node = children[segment]
        return node

    def values(self, path, key=None):
        """The array at `path`, decoded if it is coded."""
        found = self.get(path, key)
        if isinstance(found, _Attribute):
            return found.to_numpy()
        return _np.asarray(found)

    def _tree_names(self):
        """What could follow this node, for an error message."""
        names = sorted(self.child_nodes())
        names += sorted(str(VariablePath(parts))
                        for parts, _ in self.own_leaves())
        if getattr(self, "simulations", None) is not None:
            names.append("simulations")
        return names

    def _resolve(self, path):
        """The node or attribute at `path`, or None. Never raises."""
        if len(path) == 0:
            return self

        head, rest = path[0], VariablePath(path[1:])

        children = self.child_nodes()
        if head in children:
            return children[head]._resolve(rest)

        if head == "simulations" \
                and getattr(self, "simulations", None) is not None:
            if len(rest) == 0:
                return self.simulations
            try:
                return self.simulation(int(rest[0]))
            except (ValueError, IndexError, NoDataError):
                return None

        if len(rest) == 0:
            attribute = getattr(self, head, None)
            if isinstance(attribute, _Attribute):
                return attribute

        if len(rest) == 1 and head in self._DICT_FAMILIES:
            wanted = _path_key(rest[0])
            for key, attribute in (getattr(self, head, None) or {}).items():
                if _path_key(key) == wanted:
                    return attribute

        return None


def _selected_simulations(store, selection):
    """The simulations to export, read in a single pass over the store.

    Chunking splits only axis 0, so a chunk holds every simulation of a row
    band: reading one column at a time decompresses the whole array once per
    simulation. Taking the wanted columns together reads each chunk once.

    Parameters
    ----------
    store
        An (n_data, n_sim) `ArrayStore`, or None.
    selection
        `False` or `None` for no simulation, `True` for all of them, an `int`
        for the first n, or a sequence of indices.

    Returns
    -------
    A list of (index, values) pairs, in the order requested.
    """
    if store is None or selection is None or selection is False:
        return []

    n_sim = store.shape[1]
    if selection is True:
        index = list(range(n_sim))
    elif isinstance(selection, (int, _np.integer)):
        index = list(range(min(int(selection), n_sim)))
    else:
        index = [int(i) for i in selection]

    outside = [i for i in index if i < 0 or i >= n_sim]
    if len(outside) > 0:
        raise IndexError(
            "simulation(s) %s are not among the %d available"
            % (outside, n_sim))
    if len(index) == 0:
        return []

    values = _np.asarray(store.as_dask()[:, index])
    return list(zip(index, values.T))


def _code_dtype(n_labels):
    """Smallest integer type holding `n_labels` codes and the missing one.

    Signed, because -1 is what "not measured here" looks like.
    """
    return _np.min_scalar_type(-n_labels)


def _encode(values, labels):
    """Codes for `values` against `labels`; anything else is -1 (missing).

    A lookup rather than a `searchsorted`: a variable's labels are in the order
    the variable wants them (`BinaryVariable` puts the positive class first),
    not in sorted order.
    """
    codes = _np.full(len(values), -1, dtype=_code_dtype(len(labels)))
    for i, label in enumerate(labels):
        codes[values == label] = i
    return codes


class _Attribute(object):
    """A specific sequence of values, tied to data locations.

    One value per location, in an `ArrayStore` — NumPy in RAM or chunked Zarr
    on disk, chosen by size. Variables are built out of these (measurements,
    latent mean, predictions, ...), and so is a container's point-wise
    metadata.

    Text can be held as integer codes plus a `labels` list, built by
    :meth:`encoded`; `labels` is None for a plain numeric attribute. Codes are
    what makes text affordable: an object array of two million rock codes
    allocates 105 MB against 1.9 MB as `int8`, and `ArrayStore` cannot spill
    an object array to disk at all.
    """

    # An attribute is coded only when `encoded` says so; the rest are plain.
    labels = None

    def __init__(self, coordinates, values=None, dtype=None):
        self.coordinates = coordinates

        if dtype is None:
            dtype = float

        if values is None:
            # Lazily allocated and NaN-filled; the backend (NumPy in RAM
            # vs. chunked Zarr on disk) is chosen by size. Large arrays of
            # the same container share one consolidated scratch store.
            self._store = _storage.ArrayStore.allocate(
                (coordinates.n_data,), dtype=dtype, fill_value=_np.nan,
                owner=coordinates)
        else:
            values = _np.array(values, ndmin=1, dtype=dtype)
            if len(values.shape) > 1:
                values = _np.squeeze(values)

            if len(values.shape) != 1:
                raise ValueError("Values must be 1-dimensional")

            if len(values) != coordinates.n_data:
                raise ValueError("Values and coordinates size mismatch")

            # by size, as above: an attribute built from values used to be
            # pinned in RAM while an empty one of the same length was not
            self._store = _storage.ArrayStore.from_values(
                values, owner=coordinates)

    @classmethod
    def encoded(cls, coordinates, values=None, labels=None):
        """An attribute holding text as integer codes plus a label list.

        Codes index `labels`; -1 is missing, and decodes to the empty string —
        which is how the categorical variables spell "not measured here".

        `labels` may be given when the categories are already known, as they
        are for a variable, which owns them. Values are then encoded against
        that list (anything not in it is missing), unless they are integers,
        in which case they already are the codes. Without values, the whole
        attribute is missing.
        """
        if labels is None:
            # pandas rather than `np.unique`: it drops the missing values
            # instead of making a category of them, codes them -1, and is how
            # the variables derive their own labels, so the two agree — which
            # they would not if the categories were stringified here and the
            # variable's were left as, say, the integers they came in as
            categorical = _pd.Categorical(_np.asarray(values))
            labels = list(categorical.categories)
            values = categorical.codes.astype(_code_dtype(len(labels)))
        elif values is None:
            values = _np.full(coordinates.n_data, -1,
                              dtype=_code_dtype(len(labels)))
        else:
            values = _np.asarray(values)
            if values.dtype.kind not in "iu":
                values = _encode(values, labels)

        attribute = cls(coordinates, values, dtype=values.dtype)
        attribute.labels = list(labels)
        return attribute

    @classmethod
    def from_store(cls, coordinates, store, labels=None):
        """Wrap an existing store (reopened from disk) without copying it."""
        attribute = cls.__new__(cls)
        attribute.coordinates = coordinates
        attribute._store = store
        if labels is not None:
            attribute.labels = list(labels)
        return attribute

    def to_numpy(self):
        """The values, with a coded attribute decoded back to its labels.

        The label list is extended with the empty string, so that -1 — the
        missing code — indexes it without a branch of its own.
        """
        values = self.values.to_numpy()
        if self.labels is None:
            return values
        return _np.asarray(self.labels + [""], dtype=object)[values]

    @property
    def values(self):
        return self._store

    @values.setter
    def values(self, new):
        # Region assignment (``attr.values[idx] = ...``) mutates the store
        # in place and never reaches this setter; this handles whole
        # replacement (``attr.values = array``).
        if isinstance(new, _storage.ArrayStore):
            self._store = new
        else:
            self._store = _storage.ArrayStore.from_numpy(_np.asarray(new))

    def __str__(self):
        return self.values.__array__().__str__()

    def __repr__(self):
        return self.values.__array__().__repr__()

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.values = _np.array(_np.asarray(self.values)[item], ndmin=1)
        return new_obj

    def as_series(self, sigma=None):
        """
        Reshapes the data in the form of a smoothed 1-dimensional series.

        Arguments
        ---------
        sigma : scalar or array
            Standard deviation of a Gaussian filter applied to the data, in grid steps.
            Default is no filter.

        Returns
        -------
        series : _np.array
            A 1-dimensional array.
        """
        if not isinstance(self.coordinates, Grid1D):
            raise ValueError("method only available for Grid1D data"
                             "objects")

        series = self.to_numpy()

        if sigma is not None:
            series = _filters.gaussian(series, sigma, preserve_range=True)
        return series

    def as_image(self, sigma=None):
        """
        Reshapes the data in the form of a matrix for plotting.

        The output can be used in plotting functions such as
        matplotlib's `imshow()`. If you use it, do not forget to set
        `origin="lower"`.

        Arguments
        ---------
        sigma : scalar or array
            Standard deviation of a Gaussian filter applied to the data, in grid steps.
            Default is no filter.

        Returns
        -------
        image : _np.array
            A 2-dimensional array.
        """
        if not isinstance(self.coordinates, Grid2D):
            raise ValueError("method only available for Grid2D data"
                             "objects")

        image = _np.reshape(self.to_numpy(),
                            self.coordinates.grid_size,
                            order="F")
        image = image.transpose()

        if sigma is not None:
            image = _filters.gaussian(image, sigma, preserve_range=True)
        return image

    def as_cube(self, sigma=None):
        """
        Returns a rank-3 array filled with the specified variable.

        Arguments
        ---------
        sigma : scalar or array
            Standard deviation of a Gaussian filter applied to the data, in grid steps.
            Default is no filter.

        Returns
        -------
        cube : _np.array
            A rank-3 array.
        """
        if not isinstance(self.coordinates, Grid3D):
            raise ValueError("method only available for Grid3D data"
                             "objects")

        cube = _np.reshape(self.to_numpy(),
                           self.coordinates.grid_size,
                           order="F")
        if sigma is not None:
            cube = _filters.gaussian(cube, sigma, preserve_range=True)
        return cube

    def smooth(self, sigma):
        if isinstance(self.coordinates, (Grid1D, Blocks1D)):
            series = self.as_series(sigma=sigma)
            self.values = series
        elif isinstance(self.coordinates, (Grid2D, Blocks2D)):
            image = self.as_image(sigma=sigma).T
            self.values = image.ravel()
        elif isinstance(self.coordinates, (Grid3D, Blocks3D, RotatedGrid3D)):
            cube = self.as_cube(sigma=sigma).transpose([2, 1, 0])
            self.values = cube.ravel()
        else:
            raise NotGriddedDataError("method only available for gridded data")

    def get_contour(self, value, sigma=None, close=False):
        """
        Isosurface extraction.

        This method calls `skimage.measure.marching_cubes()`.
        See the original documentation for details.

        Parameters
        ----------
        value : double
            The value on which to calculate the isosurface.
        sigma : scalar or array
            Standard deviation of a Gaussian filter applied to the data before contouring.
            Default is no filter.
        close : bool or str
            Whether to close the surface where it runs out of the grid, so
            that what comes back is a body rather than a sheet with a hole in
            the side. `"above"` (or `True`) keeps the region where the values
            exceed `value` — a grade shell — and `"below"` the region under
            it. False leaves the surface open at the boundary.

        Returns
        -------
        surf : Solid3D, Surface3D or Mesh3D
            Whichever the geometry calls for: a contour that closes inside
            the grid is a body and carries its volume, one the grid cuts off
            is a sheet, and `close` is what turns the second into the first.
        """
        cube = self.as_cube(sigma=sigma)

        if close:
            # A contour reaching the edge of the grid is left open there, the
            # cube ending before the surface does. One cell of "well outside"
            # all the way round gives it somewhere to close, and costs one
            # cell of offset, the padded cube starting a step earlier.
            kept = "above" if close is True else close
            if kept not in ("above", "below"):
                raise ValueError(
                    "close takes 'above', 'below' or True (meaning 'above'); "
                    "got %r" % close)
            beyond = (_np.nanmin(cube) - 1 if kept == "above"
                      else _np.nanmax(cube) + 1)
            cube = _np.pad(cube, 1, constant_values=beyond)

        verts, faces, normals, values = _measure.marching_cubes(
            cube, value, gradient_direction="ascent",
            allow_degenerate=False, spacing=self.coordinates.step_size)

        if close:
            verts = verts - _np.asarray(
                self.coordinates.step_size, dtype=float)[None, :]

        mat = self.coordinates.rotation_matrix()

        verts = _np.matmul(verts, mat) + self.coordinates.origin
        normals = _np.matmul(normals, mat)

        # return verts, faces, normals, values
        # a contour that closes inside the grid is a body; one the grid cuts
        # off is a sheet, and `mesh3d` is what tells them apart
        return mesh3d(verts, faces, normals)

    def export_contour(self, value, filename, triangles=True,
                       offset=None, sigma=None):
        surface = self.get_contour(value, sigma=sigma)
        verts = _np.asarray(surface.coordinates)
        faces = _np.asarray(surface.triangles)

        if offset is None:
            offset = _np.zeros([1, 3])
        else:
            offset = _np.array(_np.squeeze(offset))[None, :]
        verts = verts + offset
        
        with open(filename, 'w') as out_file:
            if triangles:
                out_file.write(
                    str(verts.shape[0]) + " " + str(faces.shape[0]) + "\n")
            for line in verts:
                out_file.write(" ".join(str(elem) for elem in line) + "\n")
            if triangles:
                for line in faces:
                    out_file.write(
                        " ".join(str(elem) for elem in line) + "\n")

    def _has_content(self):
        """Whether there is anything worth writing out.

        The test is vectorized: the built-in `all()` walked the array one
        element at a time, in Python, before a single value was written.
        """
        values = _np.asarray(self.values)
        if self.labels is not None:
            return not _np.all(values < 0)
        if values.dtype == object:
            return not _np.all(values == "")
        return not _np.all(_np.isnan(values))

    def fill_pyvista_cube(self, cube, label, sigma=None):
        if self._has_content():
            cube.point_data[label] = self.as_cube(sigma=sigma) \
                .transpose([2, 1, 0]).ravel()

    def fill_pyvista_points(self, points, label):
        if self._has_content():
            points.point_data[label] = self.to_numpy()

    def fill_pyvista_blocks(self, cube, label, sigma=None):
        if self._has_content():
            cube.cell_data[label] = self.as_cube(sigma=sigma) \
                .transpose([2, 1, 0]).ravel()

    def fill_pyvista_cells(self, mesh, label):
        """One value per cell, in the order the cells were built.

        The unstructured counterpart of `fill_pyvista_blocks`: where a regular
        grid has to fold its values back into a cube first, a mesh built one
        cell per location already has them in the right order.
        """
        if self._has_content():
            mesh.cell_data[label] = self.to_numpy()

    def draw_contour(self, value, sigma=None, **kwargs):
        """Creates plotly object with the contour at the specified value."""
        surf_obj = self.get_contour(value, sigma=sigma)
        return _py.isosurface(
            surf_obj.coordinates, surf_obj.triangles, **kwargs)

    def draw_numeric(self, **kwargs):
        if self.coordinates.n_dim != 3:
            raise NotImplemented("method currently available only for"
                                 "3D coordinates")

        if isinstance(self.coordinates, Section3D):
            values = _np.reshape(self.to_numpy(),
                                 self.coordinates.grid_shape,
                                 order="F")
            gridded_x = _np.reshape(self.coordinates.coordinates[:, 0],
                                    self.coordinates.grid_shape,
                                    order="F")
            gridded_y = _np.reshape(self.coordinates.coordinates[:, 1],
                                    self.coordinates.grid_shape,
                                    order="F")
            gridded_z = _np.reshape(self.coordinates.coordinates[:, 2],
                                    self.coordinates.grid_shape,
                                    order="F")

            return _py.numeric_section_3d(gridded_x, gridded_y, gridded_z,
                                          values, **kwargs)

        if isinstance(self.coordinates, Surface3D):
            return _py.isosurface(self.coordinates.coordinates,
                                  self.coordinates.triangles,
                                  values=self.to_numpy())

        return _py.numeric_points_3d(
            self.coordinates.coordinates,
            self.to_numpy(),
            **kwargs)

    def draw_categorical(self, colors, **kwargs):
        if self.coordinates.n_dim != 3:
            raise NotImplemented("method currently available only for"
                                 "3D coordinates")

        return _py.categorical_points_3d(
            self.coordinates.coordinates,
            self.values.to_numpy(),
            colors,
            **kwargs)


class _Variable(_TreeNode):
    """Representation of a dependent random variable."""

    # Subclasses that can be simulated replace this with an (n_data, n_sim)
    # store; declared here so the accessors below work on every variable.
    simulations = None

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self._length = 1
        self.metrics = None

    # What this variable's ``labels`` are called when it prints itself.
    _LABEL_KIND = "labels"

    def __repr__(self):
        s = "%s(%r, n_data=%s" % (
            self.__class__.__name__, self.name, self.coordinates.n_data)
        if self.n_sim > 0:
            s += ", n_sim=%d" % self.n_sim
        return s + ")"

    def __str__(self):
        """
        The variable, what it is made of, and what can be read off it.

        ``repr`` says which variable this is; this says what is on it, so that
        the columns do not have to be looked up in the source -- a variable
        carries different ones depending on what it models.

        Only the names are listed, not whether anything has been written into
        them: knowing that means reading each column, and a column on a large
        model lives on disk, which is too much to do for a print.
        """
        lines = [repr(self)]

        labels = getattr(self, "labels", None)
        if labels is not None:
            lines.append("    %s: %s" % (
                self._LABEL_KIND, ", ".join(str(label) for label in labels)))

        columns = [name for name, value in vars(self).items()
                   if isinstance(value, _Attribute)]
        if len(columns) > 0:
            lines.append("    columns: " + ", ".join(columns))

        for name in ("quantiles", "probabilities"):
            keys = getattr(self, name, None)
            if keys:
                lines.append("    %s: %s"
                             % (name, ", ".join(str(key) for key in keys)))

        if self.n_sim > 0:
            lines.append("    simulations: %d" % self.n_sim)

        return "\n".join(lines)

    @property
    def length(self):
        return self._length

    @classmethod
    def from_variable(cls, coordinates, variable):
        raise NotImplementedError

    def split_shares(self):
        """Which decisions this variable carries, and how often each cuts a
        block in two.

        `{name: (n_data,) array}`, the name saying which decision *within*
        this variable -- a cut-off for a grade, a boundary for a category --
        and empty where the variable declares none.

        Asked of the variable rather than worked out from outside, because
        what carries the answer differs by kind: a graded variable holds one
        column per cut-off in a dict, a category holds a single column of its
        own, and a variable made of components has whatever its components
        have. Reaching in for an attribute called `divided` and hoping finds
        an `_Attribute` on one and an `OrderedDict` on the other.
        """
        shares = {}
        for label, component in (
                getattr(self, "components", None) or {}).items():
            for key, values in component.split_shares().items():
                shares[("%s %s" % (label, key)).strip()] = values
        return shares

    def fill_pyvista_cells(self, mesh, prefix=None, simulations=False):
        """Every filled column of this variable, as cell data.

        One implementation for every variable type, rather than the per-type
        `fill_pyvista_*` above: a mesh of one cell per location wants the
        values in the order they are already in, so the roles the variable
        persists are exactly the ones worth exporting.
        """
        label = _export_label(prefix, self.name)

        for role in self._ZARR_ATTRS:
            attribute = getattr(self, role, None)
            if attribute is not None:
                attribute.fill_pyvista_cells(mesh, "%s - %s" % (label, role))

        for i, values in _selected_simulations(self.simulations, simulations):
            _Attribute(self.coordinates, values).fill_pyvista_cells(
                mesh, "%s - simulation %d" % (label, i))

        for component in (getattr(self, "components", None) or {}).values():
            component.fill_pyvista_cells(mesh, prefix=label,
                                         simulations=simulations)

    def carry_to(self, coordinates, keep, n_new):
        """This variable on a longer set of locations, keeping what still fits.

        The locations of `coordinates` are the `keep` ones of this variable, in
        their old order, followed by `n_new` that did not exist before. The
        first take their values across; the rest come back missing, which is
        what marks them as still to be predicted.

        Used when a block set is refined: a block that was not split is the
        same block on the same support, and its value is still the right
        answer for it, so re-predicting it would spend the work to arrive at
        the number already there.
        """
        new = self.__class__.from_variable(coordinates, self)
        self._carry_into(new, _np.asarray(keep, dtype=bool))
        return new

    def _carry_into(self, new, keep):
        """Fill an already-built variable with the `keep` rows of this one.

        Apart from `carry_to` so that a variable holding components fills the
        ones its own `from_variable` just created, rather than building a
        second set that would go nowhere.
        """
        n_kept = int(_np.count_nonzero(keep))

        for role in self._ZARR_ATTRS:
            old = getattr(self, role, None)
            fresh = getattr(new, role, None)
            if old is None or fresh is None or not old._has_content():
                continue
            fresh.labels = old.labels
            fresh.values[:n_kept] = _np.asarray(old.values)[keep]

        if self._ZARR_HAS_SIMS and self.simulations is not None:
            new.allocate_simulations(self.simulations.shape[1])
            _carry_rows(self.simulations, new.simulations, keep)

        if self._ZARR_HAS_QUANTILES:
            for source, target in ((self.quantiles, new.quantiles),
                                   (self.probabilities, new.probabilities),
                                   (self.proportions, new.proportions),
                                   (self.divided, new.divided)):
                for key, attr in source.items():
                    fresh = self._Attribute(new.coordinates)
                    fresh.values[:n_kept] = _np.asarray(attr.values)[keep]
                    target[key] = fresh

        for name, component in (getattr(self, "components", None) or {}).items():
            component._carry_into(new.components[name], keep)

    def __getitem__(self, item):
        raise NotImplementedError

    def as_data_frame(self, **kwargs):
        raise NotImplementedError

    def get_measurements(self):
        raise NotImplementedError

    def get_simulations(self):
        raise NotImplementedError

    @property
    def n_sim(self):
        """Number of simulations available, or 0 if none were drawn."""
        return 0 if self.simulations is None else self.simulations.shape[1]

    def simulation(self, index):
        """
        A single simulation, in the form of an `_Attribute`.

        Simulations are kept in one `(n_data, n_sim)` array, so a single one is
        not an attribute in its own right. This wraps the corresponding column,
        giving it the usual helpers (`as_image()`, `as_cube()`, `smooth()`,
        `get_contour()`, `draw_*()`, ...).

        Parameters
        ----------
        index : int
            Position of the simulation, from 0 to `n_sim - 1`.

        Returns
        -------
        attribute : _Attribute
            A copy of the simulation. Modifying it (with `smooth()`, for
            instance) does not affect the stored simulations; assign it back
            with `variable.simulations[:, index] = attribute.values` if that is
            the intention.
        """
        if self.simulations is None:
            raise NoDataError(
                f'No simulations available for variable {self.name}.')

        return self._Attribute(self.coordinates, self.simulations[:, index])

    def get_predictions(self):
        raise NotImplementedError

    def prediction_input(self):
        return {}

    def training_input(self, idx=None):
        return {}

    def copy_to(self, coordinates):
        coordinates.variables[self.name] = self.__class__.from_variable(
            coordinates, self
        )

    def update(self, idx, **kwargs):
        raise NotImplementedError

    def allocate_simulations(self, n_sim):
        raise NotImplementedError

    def set_coordinates(self, coordinates):
        raise NotImplementedError

    def fill_pyvista_cube(self, cube, prefix=None, simulations=False):
        raise NotImplementedError

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        raise NotImplementedError

    def compute_metrics(self, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Zarr persistence (see _SpatialData.to_zarr / _SpatialData.open)
    # ------------------------------------------------------------------ #
    # Scalar ``_Attribute`` roles to persist; overridden per subclass.
    _ZARR_ATTRS = ()
    _ZARR_HAS_SIMS = False        # has a (n_data, n_sim) simulations store
    _ZARR_HAS_QUANTILES = False   # has quantiles / probabilities dicts

    def _save_attr(self, group, prefix, role):
        """Write one ``_Attribute``'s store into ``group``; None-valued -> skip.

        String/object attributes are stored as fixed-length unicode; everything
        else is streamed as a numeric/bool Zarr array.
        """
        attr = getattr(self, role, None)
        if attr is None:
            return None
        key = prefix + "/" + role
        store = attr.values
        if _np.dtype(store.dtype) == object:
            unicode = _np.asarray(store).astype(str)
            if unicode.dtype.itemsize == 0:
                unicode = unicode.astype("<U1")
            target = group.create_array(
                name=key, shape=unicode.shape, chunks=unicode.shape,
                dtype=unicode.dtype)
            target[:] = unicode
            return {"key": key, "encoding": "str"}
        store.write_into(group, key)
        info = {"key": key, "encoding": "array"}
        if attr.labels is not None:
            # a coded attribute's categories are its own — a measurement
            # column's are not the variable's. Stringified, as the variable's
            # own labels already are: this goes into JSON, and categories can
            # come out of pandas as NumPy integers.
            info["labels"] = [str(label) for label in attr.labels]
        return info

    def _load_attr(self, group, info):
        z = group[info["key"]]
        if info["encoding"] == "str":
            return _storage.ArrayStore.from_numpy(
                _np.asarray(z[:]).astype(object))
        return _storage.ArrayStore.wrap_zarr(z)

    def _zarr_save(self, group, prefix):
        # str(name): a component is named after its category, which pandas can
        # hand over as a NumPy integer, and this is JSON
        meta = {"class": type(self).__name__, "name": str(self.name),
                "attrs": {}}
        if hasattr(self, "labels"):
            meta["labels"] = [str(x) for x in self.labels]
        for role in self._ZARR_ATTRS:
            info = self._save_attr(group, prefix, role)
            if info is not None:
                meta["attrs"][role] = info
        if self._ZARR_HAS_SIMS and self.simulations is not None:
            key = prefix + "/simulations"
            self.simulations.write_into(group, key)
            meta["simulations"] = key
        if self._ZARR_HAS_QUANTILES:
            meta["quantiles"] = []
            for p, attr in self.quantiles.items():
                key = prefix + "/quantile_" + str(p)
                attr.values.write_into(group, key)
                meta["quantiles"].append({"key": key, "p": float(p)})
            meta["probabilities"] = []
            for q, attr in self.probabilities.items():
                key = prefix + "/probability_" + str(q)
                attr.values.write_into(group, key)
                meta["probabilities"].append({"key": key, "q": float(q)})
            for role, source in (("proportions", "proportion"),
                                 ("divided", "divided")):
                meta[role] = []
                for c, attr in getattr(self, role, {}).items():
                    key = prefix + "/" + source + "_" + str(c)
                    attr.values.write_into(group, key)
                    meta[role].append({"key": key, "c": float(c)})
            if getattr(self, "cutoffs", None) is not None:
                meta["cutoffs"] = [float(c) for c in self.cutoffs]
        if getattr(self, "components", None):
            meta["components"] = {}
            for cname, comp in self.components.items():
                # str: this is a JSON key, and a category can be a NumPy
                # integer. The labels above are stringified for the same
                # reason, so the rebuilt variable's components match.
                meta["components"][str(cname)] = comp._zarr_save(
                    group, prefix + "/" + str(cname))
        return meta

    def _zarr_load(self, group, prefix, meta):
        for role, info in meta.get("attrs", {}).items():
            attribute = getattr(self, role)
            store = self._load_attr(group, info)
            if attribute.labels is not None and info["encoding"] == "str":
                # written before the categorical attributes held codes; a
                # value outside the variable's labels is lost here, there
                # being nothing else to read the categories from
                store = _storage.ArrayStore.from_numpy(
                    _encode(_np.asarray(store), attribute.labels))
            if "labels" in info:
                attribute.labels = list(info["labels"])
            attribute.values = store
        if meta.get("simulations") is not None:
            self.simulations = _storage.ArrayStore.wrap_zarr(
                group[meta["simulations"]])
        for info in meta.get("quantiles", []):
            attr = self._Attribute(self.coordinates)
            attr.values = _storage.ArrayStore.wrap_zarr(group[info["key"]])
            self.quantiles[info["p"]] = attr
        for info in meta.get("probabilities", []):
            attr = self._Attribute(self.coordinates)
            attr.values = _storage.ArrayStore.wrap_zarr(group[info["key"]])
            self.probabilities[info["q"]] = attr
        for role in ("proportions", "divided"):
            for info in meta.get(role, []):
                attr = self._Attribute(self.coordinates)
                attr.values = _storage.ArrayStore.wrap_zarr(group[info["key"]])
                getattr(self, role)[info["c"]] = attr
        if meta.get("cutoffs") is not None:
            self.cutoffs = [float(c) for c in meta["cutoffs"]]
        for cname, cmeta in meta.get("components", {}).items():
            self.components[cname]._zarr_load(
                group, prefix + "/" + str(cname), cmeta)


# `_Attribute` used to be defined inside `_Variable`; the alias keeps its
# `self._Attribute(...)` call sites working.
_Variable._Attribute = _Attribute


class ContinuousVariable(_Variable):
    """
    Representation of a continuous random variable.

    Attributes
    ----------
    measurements : _Attribute
        The raw measurements.
    latent_mean : _Attribute
        The mean of the latent Gaussian representation.
    latent_variance : _Attribute
        The variance of the latent Gaussian representation.
    dispersion : _Attribute
        How much the locations inside each block differ among themselves --
        the variance over a block's sub-blocks, averaged over the
        realizations, in the variable's own units rather than the latent
        ones. A different question from `latent_variance`, which is how sure
        the model is *of* the block: a well-known block can still be
        heterogeneous, and that is what decides whether cutting it finer
        would tell anyone anything. Filled only where the container
        discretizes; elsewhere a location has no interior and this stays
        missing rather than zero.
    noise_variance : _Attribute
        How far a fresh *measurement* here would fall from the value above --
        the likelihood noise carried into the variable's own units, averaged
        over the realizations. The third of three variances and the third
        question: `latent_variance` is how sure the model is of the value,
        `dispersion` is how much the ground varies inside a block, and this is
        how much a sample of it would scatter. A prediction reports the ground,
        with the noise integrated out, so this is what has to be added back to
        compare against an assay. Missing where the prediction was made with
        `include_noise=False`, there being no integration to read it from.
    simulations : ArrayStore
        Draws from the variable's posterior distribution, in a single
        `(n_data, n_sim)` array. Use `simulation()` to get one of them as an
        `_Attribute`.
    quantiles : dict
        The variables quantiles, indexed by the corresponding percentile.
    probabilities : dict
        Cumulative distribution probabilities, indexed by the corresponding
        quantile.
    """
    _ZARR_ATTRS = ("measurements", "latent_mean", "latent_variance",
                   "prediction", "dispersion", "noise_variance")
    _ZARR_HAS_SIMS = True
    _ZARR_HAS_QUANTILES = True
    _DICT_FAMILIES = ("quantiles", "probabilities", "proportions", "divided")
    _NODE_ATTRS = ("cutoffs",)

    def __init__(self, name, coordinates, measurements=None):
        super().__init__(name, coordinates)

        if measurements is None:
            self.measurements = self._Attribute(coordinates)
        else:
            self.measurements = self._Attribute(coordinates, measurements)

        self.latent_mean = self._Attribute(coordinates)
        self.latent_variance = self._Attribute(coordinates)
        self.prediction = self._Attribute(coordinates)

        # How much the locations *inside* each block differ among themselves,
        # which is a different question from how sure the model is of the block
        # (`latent_variance`). Filled only where the container discretizes;
        # everywhere else a location has no interior and this stays zero.
        self.dispersion = self._Attribute(coordinates)

        # What a sample taken here would read, as against what the ground
        # holds: the likelihood noise in this variable's units. A prediction
        # integrates that noise out, so this is the piece to add back before
        # comparing with an assay.
        self.noise_variance = self._Attribute(coordinates)

        # A single (n_data, n_sim) store (NumPy or Zarr by size); None until
        # ``allocate_simulations`` is called.
        self.simulations = None

        self.quantiles = _col.OrderedDict()
        self.probabilities = _col.OrderedDict()

        # The grades a decision turns on -- a mining cut-off, a contaminant
        # limit. Declared on the data and carried to whatever is predicted
        # from it; `None` means this variable takes no part in any decision,
        # which is the answer for the rest component of a composition.
        self.cutoffs = None
        # For each of them, how much of each block sits at or below it, over
        # the sub-blocks and the realizations both -- the recoverable share,
        # and what a partial-block report wants.
        self.proportions = _col.OrderedDict()
        # And for each, how often the cut-off passes *through* the block:
        # the share of realizations whose sub-blocks fall on both sides. A
        # different question, and the one that says whether cutting the block
        # finer would settle anything. See `likelihood._divided`.
        self.divided = _col.OrderedDict()

    def split_shares(self):
        return {"@ %g" % cutoff: attr.values.to_numpy()
                for cutoff, attr in self.divided.items()}

    def set_cutoffs(self, cutoffs):
        """The grades this variable is judged against.

        They travel with the variable, so a model trained on data that
        declares them hands them to every block model predicted from it, and
        `refine` knows what the blocks have to be resolved against without
        being told a second time.
        """
        self.cutoffs = None if cutoffs is None else \
            [float(c) for c in _np.atleast_1d(cutoffs)]
        return self

    def prediction_input(self):
        return {} if self.cutoffs is None else {"cutoffs": self.cutoffs}

    def get_measurements(self):
        values = self.measurements.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    def get_simulations(self):
        return _np.asarray(self.simulations)

    def get_predictions(self):
        return self.prediction.values.to_numpy()

    def reset_quantiles(self, probabilities=None):
        """
        Resets the variable's quantiles.

        Parameters
        ----------
        probabilities : array
            An array of probabilities, ordered values from 0 to 1 (exclusive),
            on which to compute the corresponding quantiles.
        """
        if self.simulations is None:
            raise NoDataError(f'No simulations available for variable {self.name}.')

        self.quantiles = _col.OrderedDict()
        if probabilities is not None:
            # All quantiles are computed lazily in a single chunk-by-chunk
            # pass over the simulations; the (n_data, n_sim) array is never
            # fully materialized.
            columns = self.simulations.row_quantiles(probabilities)
            targets = []
            for p in probabilities:
                attr = self._Attribute(self.coordinates)
                self.quantiles[p] = attr
                targets.append(attr.values)
            _storage.store_columns(columns, targets)

    def reset_probabilities(self, quantiles=None):
        """
        Resets the variable's probabilities.

        Parameters
        ----------
        quantiles : array
            An array of quantiles (cutoff values in the variable's units),
            ordered, on which to compute the corresponding cumulative
            probabilities in the (0, 1) interval.
        """
        if self.simulations is None:
            raise NoDataError(f'No simulations available for variable {self.name}.')

        self.probabilities = _col.OrderedDict()
        if quantiles is not None:
            # Empirical CDF, the inverse of reset_quantiles: for each cutoff,
            # the fraction of simulations at or below it, in (0, 1). Computed
            # lazily in a single chunk-by-chunk pass. (The previous
            # implementation misused np.percentile, treating the cutoff as a
            # percent rank.)
            columns = self.simulations.row_cdf(quantiles)
            targets = []
            for q in quantiles:
                attr = self._Attribute(self.coordinates)
                self.probabilities[q] = attr
                targets.append(attr.values)
            _storage.store_columns(columns, targets)

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates)
        # what the variable is judged against belongs to the variable, not to
        # the locations, so it follows it onto whatever it is predicted into
        new_var.cutoffs = variable.cutoffs
        return new_var

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        # the same list the Zarr round trip and `carry_to` walk, so a column
        # added in one place is subset in this one without being named twice
        for role in self._ZARR_ATTRS:
            column = getattr(self, role, None)
            if column is not None:
                setattr(new_obj, role, column[item])
        new_obj.proportions = _col.OrderedDict(
            (key, val[item]) for key, val in self.proportions.items())
        new_obj.divided = _col.OrderedDict(
            (key, val[item]) for key, val in self.divided.items())

        if self.simulations is not None:
            new_obj.simulations = _storage.ArrayStore.from_numpy(
                _np.asarray(self.simulations)[item])

        if len(self.quantiles) > 0:
            for key, val in self.quantiles.items():
                new_obj.quantiles[key] = val[item]

        if len(self.probabilities) > 0:
            for key, val in self.probabilities.items():
                new_obj.probabilities[key] = val[item]

        return new_obj

    def as_data_frame(self, measurements=True, predictions=True,
                      latent=True,
                      simulations=False, quantiles=True,
                      probability=True, **kwargs):
        """
        Converts the object to a DataFrame.

        Parameters
        ----------
        predictions : bool
            Whether to include the predictions.
        measurements : bool
            Whether to include the measurements.
        latent : bool
            Whether to include the latent Gaussian variable.
        simulations : bool
            Whether to include the simulations.
        quantiles : bool
            Whether to include the quantiles.
        probability : bool
            Whether to include the probabilities.
        kwargs : dict
            Ignored.

        Returns
        -------
        df : pd.DataFrame
            The converted object.
        """
        df = _pd.DataFrame({})

        if measurements:
            df[self.name] = self.measurements.values.to_numpy()

        if predictions:
            df[self.name + "_prediction"] = self.prediction.values.to_numpy()
            # dispersion only where a container discretizes, noise variance
            # only where the noise was integrated: elsewhere neither was ever
            # written and every point data set would carry an empty column
            if _filled(self.dispersion):
                df[self.name + "_dispersion"] = \
                    self.dispersion.values.to_numpy()
            if _filled(self.noise_variance):
                df[self.name + "_noise_variance"] = \
                    self.noise_variance.values.to_numpy()

        if latent:
            # a component has none of its own, and nothing is there before a
            # prediction has run
            if _filled(self.latent_mean):
                df[self.name + "_latent_mean"] = \
                    self.latent_mean.values.to_numpy()
            if _filled(self.latent_variance):
                df[self.name + "_latent_variance"] = \
                    self.latent_variance.values.to_numpy()

        for i, values in _selected_simulations(self.simulations,
                                              simulations):
            df[self.name + "_sim_" + str(i)] = values

        if quantiles:
            if len(self.quantiles) > 0:
                for key, val in self.quantiles.items():
                    df[self.name + "_q" + str(key)] = val.values.to_numpy()

        if probability:
            if len(self.probabilities) > 0:
                for key, val in self.probabilities.items():
                    df[self.name + "_p" + str(key)] = val.values.to_numpy()

        return df

    def update(self, idx, **kwargs):
        self.prediction.values[idx] = kwargs["average_sim"].numpy()

        if "mean" in kwargs.keys():
            self.latent_mean.values[idx] = kwargs["mean"].numpy()
            self.latent_variance.values[idx] = kwargs["variance"].numpy()

        if "dispersion" in kwargs.keys():
            self.dispersion.values[idx] = kwargs["dispersion"].numpy()

        if "noise_variance" in kwargs.keys():
            self.noise_variance.values[idx] = \
                kwargs["noise_variance"].numpy()

        for key, target in (("proportions", self.proportions),
                            ("divided", self.divided)):
            if key not in kwargs.keys():
                continue
            values = kwargs[key].numpy()
            cutoffs = self.cutoffs or []
            if values.shape[1] == 0:
                # a component of a vector variable that declared none, whose
                # columns its parent has already trimmed away
                continue
            # the model asked the *training* variable what the cut-offs were,
            # and the answer is being filed against this one; they match
            # unless somebody has moved one of them since
            if len(cutoffs) != values.shape[1]:
                raise ValueError(
                    "%r was predicted against %d cut-off(s) but declares %s; "
                    "set them on the data the model was trained from, and let "
                    "`copy_to` carry them"
                    % (self.name, values.shape[1], self.cutoffs))
            for i, cutoff in enumerate(cutoffs):
                if cutoff not in target:
                    target[cutoff] = self._Attribute(self.coordinates)
                target[cutoff].values[idx] = values[:, i]

        if "simulations" in kwargs.keys():
            # Whole (batch, n_sim) block written as one region into the store.
            self.simulations[idx, :] = kwargs["simulations"].numpy()

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.measurements.coordinates = coordinates
        self.latent_mean.coordinates = coordinates
        self.latent_variance.coordinates = coordinates
        self.prediction.coordinates = coordinates
        self.dispersion.coordinates = coordinates
        self.noise_variance.coordinates = coordinates

        for share in self.proportions.values():
            share.coordinates = coordinates
        for share in self.divided.values():
            share.coordinates = coordinates

        if len(self.quantiles) > 0:
            for p in self.quantiles.values():
                p.coordinates = coordinates

        if len(self.probabilities) > 0:
            for q in self.probabilities.values():
                q.coordinates = coordinates

    def fill_pyvista_cube(self, cube, prefix=None, sigma=None,
                          simulations=False):
        label = _export_label(prefix, self.name)
        self.measurements.fill_pyvista_cube(cube, label, sigma=sigma)

        if self.latent_mean:
            self.latent_mean.fill_pyvista_cube(
                cube, label + " - latent mean")
        if self.latent_variance:
            self.latent_variance.fill_pyvista_cube(
                cube, label + " - latent variance")
        self.prediction.fill_pyvista_cube(
            cube, label + " - prediction", sigma=sigma)
        self.dispersion.fill_pyvista_cube(
            cube, label + " - dispersion")
        self.noise_variance.fill_pyvista_cube(
            cube, label + " - noise variance")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_cube(cube, label + f" - simulation {i}")

        for p in self.quantiles.keys():
            self.quantiles[p].fill_pyvista_cube(
                cube, label + f" - quantile {p}", sigma=sigma
            )

        for q in self.probabilities.keys():
            self.probabilities[q].fill_pyvista_cube(
                cube, label + f" - probability {q}", sigma=sigma
            )

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        label = _export_label(prefix, self.name)
        self.measurements.fill_pyvista_points(points, label)

        if self.latent_mean:
            self.latent_mean.fill_pyvista_points(
                points, label + " - latent mean")
        if self.latent_variance:
            self.latent_variance.fill_pyvista_points(
                points, label + " - latent variance")
        self.dispersion.fill_pyvista_points(
            points, label + " - dispersion")
        self.noise_variance.fill_pyvista_points(
            points, label + " - noise variance")
        self.prediction.fill_pyvista_points(
            points, label + " - prediction")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_points(points, label + f" - simulation {i}")

        for p in self.quantiles.keys():
            self.quantiles[p].fill_pyvista_points(
                points, label + f" - quantile {p}")

        for q in self.probabilities.keys():
            self.probabilities[q].fill_pyvista_points(
                points, label + f" - probability {q}")

    def fill_pyvista_blocks(self, cube, prefix=None, sigma=None,
                            simulations=False):
        label = _export_label(prefix, self.name)
        self.measurements.fill_pyvista_blocks(cube, label, sigma=sigma)

        if self.latent_mean:
            self.latent_mean.fill_pyvista_blocks(
                cube, label + " - latent mean"
            )
        if self.latent_variance:
            self.latent_variance.fill_pyvista_blocks(
                cube, label + " - latent variance"
            )
        self.dispersion.fill_pyvista_blocks(
            cube, label + " - dispersion")
        self.noise_variance.fill_pyvista_blocks(
            cube, label + " - noise variance")
        self.prediction.fill_pyvista_blocks(
            cube, label + " - prediction", sigma=sigma
        )

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_blocks(cube, label + f" - simulation {i}")

        for p in self.quantiles.keys():
            self.quantiles[p].fill_pyvista_blocks(
                cube, label + f" - quantile {p}", sigma=sigma
            )

        for q in self.probabilities.keys():
            self.probabilities[q].fill_pyvista_blocks(
                cube, label + f" - probability {q}", sigma=sigma
            )

    def compute_metrics(self, alpha=0.05):
        y_true, has_value = self.get_measurements()
        if _np.sum(has_value) == 0:
            raise ValueError('No measurements available')

        y_pred = self.prediction.values.to_numpy()
        sims = self.get_simulations()

        has_value = has_value[:, 0]
        y_true = y_true[has_value == 1]
        y_pred = y_pred[has_value == 1]
        sims = sims[has_value == 1]

        metrics = {
            'Root Mean Square Error (prediction)': _skmetrics.root_mean_squared_error(y_true, y_pred),
            'Mean Absolute Error (prediction)': _skmetrics.mean_absolute_error(y_true, y_pred),
            'Median Absolute Error (prediction)': _skmetrics.median_absolute_error(y_true, y_pred),
            'Root Mean Square Error (simulations)': _skmetrics.root_mean_squared_error(
                _np.broadcast_to(y_true, sims.shape), sims),
            'Mean Absolute Error (simulations)': _skmetrics.mean_absolute_error(
                _np.broadcast_to(y_true, sims.shape), sims),
            'Median Absolute Error (simulations)': _skmetrics.median_absolute_error(
                _np.broadcast_to(y_true, sims.shape), sims),
        }

        bias_2, variance = _gmlmetrics.bias_variance_decomposition(y_true, sims)
        metrics['Bias squared (simulations)'] = bias_2
        metrics['Variance (simulations)'] = variance

        if not isinstance(alpha, (list, tuple)):
            alpha = [alpha]
        for a in alpha:
            metrics[f'Interval score ({a})'] = _gmlmetrics.interval_score(y_true, sims, a)

        self.metrics = _pd.Series(metrics, name=self.name)
        return self.metrics


class VectorVariable(_Variable):
    _ZARR_ATTRS = ("uncertainty",)
    _LABEL_KIND = "components"

    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates)

        if isinstance(measurements, _pd.DataFrame):
            measurements = measurements.values

        self.labels = labels
        self._length = len(labels)

        self.components = {}
        for i, label in enumerate(labels):
            self.components[label] = ContinuousVariable(
                label,
                coordinates,
                measurements[:, i] if measurements is not None else None,
            )

        self.uncertainty = self._Attribute(coordinates)

    def get_measurements(self):
        # not allowing partial missing data
        out = [self.components[label].measurements.values.to_numpy()
               for label in self.labels]
        out = _np.stack(out, axis=1)

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length])
        out = _np.where(has_value == 0.0, 1.0, out)
        return out, has_value

    def prediction_input(self):
        """The components' cut-offs, as one row each.

        They are declared per component -- two grades are judged against two
        different numbers -- but the model sees the variable whole, so they
        travel as a matrix with a row per component. A component declaring
        fewer than the widest is padded with infinity, which nothing is ever
        above, so its spare columns come back empty and `update` drops them.
        """
        declared = [self.components[label].cutoffs or [] for label in
                    self.labels]
        widest = max(len(row) for row in declared) if declared else 0
        if widest == 0:
            return {}
        return {"cutoffs": [row + [_np.inf] * (widest - len(row))
                            for row in declared]}

    def get_simulations(self):
        sims = _np.stack([self.components[v].get_simulations() for v in self.labels], axis=2)
        return sims

    def get_predictions(self):
        pred = _np.stack([self.components[v].get_predictions() for v in self.labels], axis=1)
        return pred

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        for comp in self.components.values():
            comp.set_coordinates(coordinates)
        self.uncertainty.coordinates = coordinates

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(
            variable.name,
            coordinates,
            variable.labels,
        )
        # the components are built fresh by `__init__`, so what belongs to
        # them rather than to the container has to be brought across by hand:
        # a cut-off is declared per component and a vector variable is how it
        # reaches the model
        for label, component in variable.components.items():
            new_var.components[label].cutoffs = component.cutoffs
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, columns=None,
                        *args, **kwargs):
        new_var = cls(
            name,
            coordinates,
            labels=columns,
            measurements=df.loc[:, columns].values,
        )
        return new_var

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)

        for name, comp in self.components.items():
            new_obj.components[name] = comp[item]

        new_obj.uncertainty = self.uncertainty[item]

        return new_obj

    def as_data_frame(self, measurements=True, predictions=True,
                      simulations=False, quantiles=True,
                      probability=True, **kwargs):
        all_dfs = []
        for key, val in self.components.items():
            cat_df = val.as_data_frame(
                measurements=measurements,
                predictions=predictions,
                simulations=simulations,
                quantiles=quantiles,
                probability=probability,
                latent=False
            )
            cat_df.columns = [self.name + "_" + col for col in cat_df.columns]
            all_dfs.append(cat_df)
        all_dfs = _pd.concat(all_dfs, axis=1)

        if predictions:
            all_dfs[self.name + "_uncertainty"] = \
                self.uncertainty.values.to_numpy()

        return all_dfs

    def update(self, idx, **kwargs):
        prediction = _tf.unstack(kwargs["average_sim"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)
        # each component is dispersed inside a block, and measured, on its own
        # account
        dispersion = _tf.unstack(kwargs["dispersion"], axis=1)
        noise = _tf.unstack(kwargs["noise_variance"], axis=1)
        blank = [None] * len(self.labels)
        shares = {
            key: (_tf.unstack(kwargs[key], axis=1)
                  if key in kwargs.keys() else blank)
            for key in ("proportions", "divided")}

        for i, (lb, p, s, d, nv) in enumerate(zip(self.labels, prediction,
                                                  simulations, dispersion,
                                                  noise)):
            values = {"average_sim": p, "simulations": s, "dispersion": d,
                      "noise_variance": nv}
            for key, unstacked in shares.items():
                if unstacked[i] is not None:
                    # the matrix was padded out to the widest component, so
                    # this one takes only the cut-offs it declared
                    declared = len(self.components[lb].cutoffs or [])
                    values[key] = unstacked[i][:, :declared]
            self.components[lb].update(idx, **values)

        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

    def allocate_simulations(self, n_sim):
        for comp in self.labels:
            self.components[comp].allocate_simulations(n_sim)

    def reset_quantiles(self, probabilities=None):
        for el in self.labels:
            self.components[el].reset_quantiles(probabilities)

    def reset_probabilities(self, quantiles=None):
        for el in self.labels:
            self.components[el].reset_probabilities(quantiles)

    def fill_pyvista_cube(self, cube, prefix=None, sigma=None,
                          simulations=False):
        for comp in self.labels:
            self.components[comp].fill_pyvista_cube(
                cube, self.name, sigma=sigma, simulations=simulations)
        self.uncertainty.fill_pyvista_cube(cube, self.name, sigma=sigma)

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        for comp in self.labels:
            self.components[comp].fill_pyvista_points(
                points, self.name, simulations=simulations)
        self.uncertainty.fill_pyvista_points(points, self.name)

    def fill_pyvista_blocks(self, cube, prefix=None, sigma=None,
                            simulations=False):
        for comp in self.labels:
            self.components[comp].fill_pyvista_blocks(
                cube, self.name, sigma=sigma, simulations=simulations)
        self.uncertainty.fill_pyvista_blocks(cube, self.name, sigma=sigma)

    def compute_metrics(self, alpha=0.05):
        metrics = [self.components[comp].compute_metrics(alpha) for comp in self.labels]
        metrics = _pd.concat(metrics, axis=1)
        metrics.columns = self.labels
        self.metrics = metrics
        return self.metrics


class _Component(ContinuousVariable):
    def __init__(self, name, coordinates, measurements=None):
        super().__init__(name, coordinates, measurements)
        self.latent_mean = None
        self.latent_variance = None

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        # driven by the list rather than written out: a column named there and
        # forgotten here survives the subset at its old length, and only
        # something that reads it afterwards finds out. A component has no
        # latent space of its own and says so with a None, as `_carry_into`
        # also has to allow for
        for role in self._ZARR_ATTRS:
            column = getattr(self, role, None)
            if column is not None:
                setattr(new_obj, role, column[item])

        if self.simulations is not None:
            new_obj.simulations = _storage.ArrayStore.from_numpy(
                _np.asarray(self.simulations)[item])

        if len(self.quantiles) > 0:
            for key, val in self.quantiles.items():
                new_obj.quantiles[key] = val[item]

        if len(self.probabilities) > 0:
            for key, val in self.probabilities.items():
                new_obj.probabilities[key] = val[item]

        return new_obj

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.prediction.coordinates = coordinates

        if len(self.quantiles) > 0:
            for p in self.quantiles.values():
                p.coordinates = coordinates

        if len(self.probabilities) > 0:
            for q in self.probabilities.values():
                q.coordinates = coordinates

    def as_data_frame(self, latent=False, **kwargs):
        # `ContinuousVariable`'s, with the latent space off: a component has
        # none of its own. It used to be a copy of that method minus the
        # latent columns, which is how it came to be missing `dispersion` and
        # `noise_variance` -- every column added upstream had to be added here
        # too, and the omission is silent
        return super().as_data_frame(latent=False, **kwargs)

    def update(self, idx, **kwargs):
        self.prediction.values[idx] = kwargs["prediction"].numpy()
        self.simulations[idx, :] = kwargs["simulations"].numpy()

        if "dispersion" in kwargs.keys():
            self.dispersion.values[idx] = kwargs["dispersion"].numpy()

        if "noise_variance" in kwargs.keys():
            self.noise_variance.values[idx] = \
                kwargs["noise_variance"].numpy()

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    def get_simulations(self):
        return _np.asarray(self.simulations)


class CompositionalVariable(VectorVariable):
    def __init__(self, name, coordinates, labels, measurements=None):
        super().__init__(name, coordinates, labels, measurements)
        for i, label in enumerate(labels):
            self.components[label] = _Component(
                label,
                coordinates,
                measurements[:, i] if measurements is not None else None)

    def get_measurements(self):
        # not allowing partial missing data
        out = [self.components[label].measurements.values.to_numpy()
               for label in self.labels]

        out = _np.stack(out, axis=1)
        total = _np.sum(out, axis=1, keepdims=True)
        total = _np.where(_np.abs(total - 1) < 1e-10, 1.0, _np.nan)
        out = out * total

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length]).astype(float)
        # out[_np.isnan(out)] = 1
        out = _np.where(has_value == 0.0, 1.0, out).astype(float)
        return out, has_value

        # allowing partial missing data
        # out = [self.components[label].measurements.values
        #        for label in self.labels]
        # out = _np.stack(out, axis=1)
        # has_value = ~ _np.isnan(out)
        # out[_np.isnan(out)] = 1
        # return out, has_value

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels)
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, columns=None,
                        *args, **kwargs):
        new_var = cls(
            name,
            coordinates,
            labels=columns,
            measurements=df.loc[:, columns].values)
        return new_var

    def as_data_frame(self, measurements=True, predictions=True,
                      simulations=False, quantiles=True, probability=True, **kwargs):
        all_dfs = []
        for key, val in self.components.items():
            cat_df = val.as_data_frame(
                measurements=measurements,
                predictions=predictions,
                simulations=simulations,
                quantiles=quantiles,
                probability=probability,
                **kwargs)
            cat_df.columns = [self.name + "_" + col for col in cat_df.columns]
            all_dfs.append(cat_df)
        all_dfs = _pd.concat(all_dfs, axis=1)

        if predictions:
            all_dfs[self.name + "_uncertainty"] = \
                self.uncertainty.values.to_numpy()

        return all_dfs

    def update(self, idx, **kwargs):
        prediction = _tf.unstack(kwargs["average_sim"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)
        # a part varies inside a block, and is assayed, on its own account
        dispersion = _tf.unstack(kwargs["dispersion"], axis=1)
        noise = _tf.unstack(kwargs["noise_variance"], axis=1)

        for lb, p, s, d, nv in zip(
                self.labels,
                prediction, simulations, dispersion, noise):
            self.components[lb].update(idx, **{
                "prediction": p,
                "simulations": s,
                "dispersion": d,
                "noise_variance": nv
            })

        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

    def compute_metrics(self, alpha=0.05):
        metrics = [self.components[comp].compute_metrics(alpha) for comp in self.labels]
        metrics = _pd.concat(metrics, axis=1)
        metrics.columns = self.labels

        comp_true, has_value = self.get_measurements()
        comp_pred = _np.stack([self.components[c].prediction.values.to_numpy()
                               for c in self.labels], axis=1)
        comp_true = comp_true[has_value[:, 0] == 1]
        comp_pred = comp_pred[has_value[:, 0] == 1]
        ad = _gmlmetrics.aitchison_distance(comp_true, comp_pred)
        metrics.loc['Aitchison distance', :] = ad

        self.metrics = metrics
        return self.metrics


class _Category(_Variable):
    _ZARR_ATTRS = ("probability", "indicator", "indicator_mean",
                   "indicator_variance", "indicator_predicted", "proportion",
                   "divided")
    _ZARR_HAS_SIMS = True

    def __init__(self, name, coordinates, indicator):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.indicator = self._Attribute(coordinates, indicator)
        self.indicator_mean = self._Attribute(coordinates)
        self.indicator_variance = self._Attribute(coordinates)
        self.indicator_predicted = self._Attribute(coordinates)
        # How much of each block this category holds, counted over the
        # sub-blocks: 1 where it takes the whole block, 0 where it takes none,
        # and in between where the block straddles a contact. A different
        # thing from `probability`, which is how sure the model is that the
        # block as a whole belongs here.
        self.proportion = self._Attribute(coordinates)
        # And whether the block is cut in two by this category's boundary,
        # which is the question splitting answers -- see `likelihood._divided`
        self.divided = self._Attribute(coordinates)
        self.simulations = None

    def split_shares(self):
        # one boundary, its own, and no cut-off to name it by: a category's
        # crossing is always zero
        if not self.divided._has_content():
            return {}
        return {"": self.divided.values.to_numpy()}

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.probability = self.probability[item]
        new_obj.indicator = self.indicator[item]
        new_obj.indicator_mean = self.indicator_mean[item]
        new_obj.indicator_variance = self.indicator_variance[item]
        new_obj.indicator_predicted = self.indicator_predicted[item]
        new_obj.proportion = self.proportion[item]
        new_obj.divided = self.divided[item]

        if self.simulations is not None:
            new_obj.simulations = _storage.ArrayStore.from_numpy(
                _np.asarray(self.simulations)[item])

        return new_obj

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.probability.coordinates = coordinates
        self.indicator_mean.coordinates = coordinates
        self.indicator_variance.coordinates = coordinates
        self.indicator_predicted.coordinates = coordinates
        self.proportion.coordinates = coordinates
        self.divided.coordinates = coordinates

    def as_data_frame(self, probability=True, predictions=True,
                      latent=True, simulations=False):
        df = _pd.DataFrame({})

        if probability:
            df[self.name + "_probability"] = self.probability.values.to_numpy()
            df[self.name + "_indicator"] = self.indicator.values.to_numpy()

        if latent:
            df[self.name + "_indicator_mean"] = \
                self.indicator_mean.values.to_numpy()
            df[self.name + "_indicator_variance"] = \
                self.indicator_variance.values.to_numpy()

        if predictions:
            df[self.name + "_indicator_predicted"] = \
                self.indicator_predicted.values.to_numpy()
            if self.proportion._has_content():
                df[self.name + "_proportion"] = \
                    self.proportion.values.to_numpy()

        for i, values in _selected_simulations(self.simulations,
                                              simulations):
            df[self.name + "_sim_" + str(i)] = values

        return df

    def update(self, idx, **kwargs):
        self.indicator_predicted.values[idx] = kwargs["indicator"].numpy()
        self.indicator_mean.values[idx] = kwargs["mean"].numpy()
        self.indicator_variance.values[idx] = kwargs["variance"].numpy()
        self.probability.values[idx] = kwargs["probability"].numpy()

        if "proportion" in kwargs.keys():
            self.proportion.values[idx] = kwargs["proportion"].numpy()
        if "divided" in kwargs.keys():
            self.divided.values[idx] = kwargs["divided"].numpy()

        self.simulations[idx, :] = kwargs["simulations"].numpy()

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    def fill_pyvista_cube(self, cube, prefix=None, simulations=False):
        label = prefix + " - " + self.name

        self.indicator.fill_pyvista_cube(
            cube, label + " - indicator")
        self.indicator_mean.fill_pyvista_cube(
            cube, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_cube(
            cube, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_cube(
            cube, label + " - indicator predicted")
        self.probability.fill_pyvista_cube(
            cube, label + " - probability")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_cube(cube, label + " - simulation %d" % i)

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        label = prefix + " - " + self.name

        self.indicator.fill_pyvista_points(
            points, label + " - indicator")
        self.indicator_mean.fill_pyvista_points(
            points, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_points(
            points, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_points(
            points, label + " - indicator predicted")
        self.probability.fill_pyvista_points(
            points, label + " - probability")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_points(points, label + " - simulation %d" % i)

    def fill_pyvista_blocks(self, cube, prefix=None, simulations=False):
        label = prefix + " - " + self.name

        self.indicator.fill_pyvista_blocks(
            cube, label + " - indicator")
        self.indicator_mean.fill_pyvista_blocks(
            cube, label + " - indicator mean")
        self.indicator_variance.fill_pyvista_blocks(
            cube, label + " - indicator variance")
        self.indicator_predicted.fill_pyvista_blocks(
            cube, label + " - indicator predicted")
        self.probability.fill_pyvista_blocks(
            cube, label + " - probability")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_blocks(cube, label + " - simulation %d" % i)


class RockTypeVariable(_Variable):
    _ZARR_ATTRS = ("predicted", "entropy", "uncertainty",
                   "measurements_a", "measurements_b", "boundary")
    _LABEL_KIND = "categories"

    def __init__(self, name, coordinates, labels=None, measurements_a=None,
                 measurements_b=None):
        # Coerce to numpy arrays: with plain Python lists the element-wise
        # comparisons below (indicator building, boundary detection) would
        # silently collapse to scalars.
        if measurements_a is not None:
            measurements_a = _np.asarray(measurements_a)
        if measurements_b is not None:
            measurements_b = _np.asarray(measurements_b)

        if measurements_b is None:
            measurements_b = measurements_a

        if labels is None:
            if measurements_a is None:
                raise Exception("either the labels or measurements"
                                "must be provided")
            cat_a = _pd.Categorical(measurements_a)
            cat_b = _pd.Categorical(measurements_b)
            labels = _pd.api.types.union_categoricals([cat_a, cat_b])
            labels = labels.categories.values

        n_cat = len(labels)
        n_data = coordinates.n_data

        avg_vals = None
        if measurements_a is not None:
            vals_a = _np.zeros([n_data, n_cat])
            vals_b = vals_a.copy()
            for i, label in enumerate(labels):
                vals_a[measurements_a == label, i] = 1
                vals_b[measurements_b == label, i] = 1
            avg_vals = 0.5 * (vals_a + vals_b)

        super().__init__(name, coordinates)
        self.labels = labels
        self._length = len(labels)

        self.components = {}
        for i, label in enumerate(labels):
            self.components[label] = _Category(
                label,
                coordinates,
                avg_vals[:, i] if avg_vals is not None else None,
            )

        # the categories are known here, so these three hold codes into
        # `labels` rather than one string object per data location
        self.predicted = self._Attribute.encoded(coordinates, labels=labels)
        self.entropy = self._Attribute(coordinates)
        self.uncertainty = self._Attribute(coordinates)

        if measurements_a is None:
            self.measurements_a = self._Attribute.encoded(
                coordinates, labels=labels)
            self.measurements_b = self._Attribute.encoded(
                coordinates, labels=labels)
            self.boundary = self._Attribute(
                coordinates, [False]*n_data, dtype=bool)
        else:
            # their own categories, not the variable's: a measurement outside
            # `labels` is still worth keeping as what it says
            self.measurements_a = self._Attribute.encoded(
                coordinates, measurements_a)
            self.measurements_b = self._Attribute.encoded(
                coordinates, measurements_b)
            self.boundary = self._Attribute(
                coordinates, measurements_a != measurements_b, dtype=bool)

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

        for comp in self.components.values():
            comp.set_coordinates(coordinates)

        self.predicted.coordinates = coordinates
        self.entropy.coordinates = coordinates
        self.uncertainty.coordinates = coordinates
        self.boundary.coordinates = coordinates

    def get_measurements(self):
        # not allowing partial missing data
        out = [self.components[label].indicator.values.to_numpy()
               for label in self.labels]

        out = _np.stack(out, axis=1)
        total = _np.sum(out, axis=1, keepdims=True)
        total = _np.where(_np.abs(total - 1) < 1e-10, 1.0, _np.nan)
        out = out * total

        has_value = _np.all(~ _np.isnan(out), axis=1, keepdims=True) * 1.0
        has_value = _np.tile(has_value, [1, self.length]).astype(float)
        out = _np.where(has_value == 0.0, 1.0, out).astype(float)
        return out, has_value

    def allocate_simulations(self, n_sim):
        for comp in self.labels:
            self.components[comp].allocate_simulations(n_sim)

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels)
        return new_var

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col_a=None, col_b=None,
                        *args, **kwargs):
        labels = _pd.api.types.union_categoricals(
            [_pd.Categorical(df[col_a]),
             _pd.Categorical(df[col_b])])
        labels = labels.categories.values

        new_var = cls(name, coordinates, labels,
                      measurements_a=df[col_a].values,
                      measurements_b=df[col_b].values)
        return new_var

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)

        new_obj.entropy = self.entropy[item]
        new_obj.uncertainty = self.uncertainty[item]
        new_obj.boundary = self.boundary[item]
        new_obj.measurements_a = self.measurements_a[item]
        new_obj.measurements_b = self.measurements_b[item]

        # Only the categories still present: slicing is a pre-processing step
        # before training, and a category with no data left in it has nothing
        # to teach the model. The parent's order is kept, since it is the order
        # of the components and of the codes below.
        present = set(new_obj.measurements_a.to_numpy()) \
            | set(new_obj.measurements_b.to_numpy())
        labels = [label for label in self.labels if label in present]
        if len(labels) == 0:
            # nothing is measured here at all — a prediction target, say, whose
            # categories come from the model rather than from the data
            labels = list(self.labels)
        new_obj.labels = labels
        new_obj._length = len(labels)

        # the components and the prediction follow the labels: `update` writes
        # the winning label's position, which the dropped ones would shift
        new_obj.components = {label: self.components[label][item]
                              for label in labels}
        predicted = self.predicted[item]
        predicted.values = _encode(predicted.to_numpy(), labels)
        predicted.labels = list(labels)
        new_obj.predicted = predicted

        return new_obj

    def as_data_frame(self, measurements=True, probability=True, predictions=True,
                      latent=True, simulations=False, **kwargs):
        all_dfs = []
        for key, val in self.components.items():
            cat_df = val.as_data_frame(
                predictions=predictions,
                simulations=simulations,
                latent=latent,
                probability=probability,
                **kwargs)
            cat_df.columns = [self.name + "_" + col for col in cat_df.columns]
            all_dfs.append(cat_df)
        df = _pd.concat(all_dfs, axis=1)

        if measurements:
            df[self.name + "_a"] = self.measurements_a.to_numpy()
            df[self.name + "_b"] = self.measurements_b.to_numpy()

        if predictions:
            df[self.name + "_predicted"] = self.predicted.to_numpy()
            df[self.name + "_entropy"] = self.entropy.values.to_numpy()
            df[self.name + "_uncertainty"] = self.uncertainty.values.to_numpy()

        return df

    def update(self, idx, **kwargs):
        self.entropy.values[idx] = kwargs["entropy"].numpy()
        self.uncertainty.values[idx] = kwargs["uncertainty"].numpy()

        # the winning category's position is the code
        self.predicted.values[idx] = _np.argmax(
            kwargs["probability"].numpy(), axis=1)

        mean = _tf.unstack(kwargs["mean"], axis=1)
        variance = _tf.unstack(kwargs["variance"], axis=1)
        indicators = _tf.unstack(kwargs["indicators"], axis=1)
        probability = _tf.unstack(kwargs["probability"], axis=1)
        simulations = _tf.unstack(kwargs["simulations"], axis=1)
        # the share of each block this category holds, one column per label
        blank = [None] * len(self.labels)
        proportions = _tf.unstack(kwargs["proportions"], axis=1) \
            if "proportions" in kwargs.keys() else blank
        divided = _tf.unstack(kwargs["divided"], axis=1) \
            if "divided" in kwargs.keys() else blank

        for lb, m, v, i, p, s, share, cut in zip(
                self.labels, mean, variance, indicators,
                probability, simulations, proportions, divided):
            values = {"mean": m, "variance": v, "indicator": i,
                      "probability": p, "simulations": s}
            if share is not None:
                values["proportion"] = share
                values["divided"] = cut
            self.components[lb].update(idx, **values)

    def training_input(self, idx=None):
        if idx is None:
            idx = _np.arange(self.coordinates.n_data)
        return {"is_boundary": _tf.constant(
            self.boundary.values.to_numpy()[idx, None], _tf.bool)}

    def fill_pyvista_cube(self, cube, prefix=None, simulations=False):
        self.measurements_a.fill_pyvista_cube(
            cube, self.name + " - measurements_a")
        self.measurements_b.fill_pyvista_cube(
            cube, self.name + " - measurements_b")
        self.predicted.fill_pyvista_cube(
            cube, self.name + " - predicted")
        self.entropy.fill_pyvista_cube(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_cube(
            cube, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_cube(
                cube, self.name, simulations=simulations)

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        self.measurements_a.fill_pyvista_points(
            points, self.name + " - measurements_a")
        self.measurements_b.fill_pyvista_points(
            points, self.name + " - measurements_b")
        self.predicted.fill_pyvista_points(
            points, self.name + " - predicted")
        self.entropy.fill_pyvista_points(
            points, self.name + " - entropy")
        self.uncertainty.fill_pyvista_points(
            points, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_points(
                points, self.name, simulations=simulations)

    def fill_pyvista_blocks(self, cube, prefix=None, simulations=False):
        self.measurements_a.fill_pyvista_blocks(
            cube, self.name + " - measurements_a")
        self.measurements_b.fill_pyvista_blocks(
            cube, self.name + " - measurements_b")
        self.predicted.fill_pyvista_blocks(
            cube, self.name + " - predicted")
        self.entropy.fill_pyvista_blocks(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_blocks(
            cube, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_blocks(
                cube, self.name, simulations=simulations)

    def compute_metrics(self, **kwargs):
        y_pred = self.predicted.to_numpy()
        y_true_a = self.measurements_a.to_numpy()
        y_true_b = self.measurements_b.to_numpy()

        valid = y_true_a == y_true_b

        y_pred = y_pred[valid]
        y_true = y_true_a[valid]

        series = []
        for lab in self.labels:
            yp = _np.where(y_pred == lab, 1, 0)
            yt = _np.where(y_true == lab, 1, 0)
            d = {
                'Balanced accuracy': _skmetrics.balanced_accuracy_score(yt, yp),
                'Jaccard': _skmetrics.jaccard_score(yt, yp),
                'Matthews': _skmetrics.matthews_corrcoef(yt, yp),
            }
            series.append(_pd.Series(d, name=lab))

        self.metrics = _pd.concat(series, axis=1)
        return self.metrics


class CategoricalVariable(RockTypeVariable):
    def __init__(self, name, coordinates, labels=None, measurements=None):
        super().__init__(name, coordinates, labels=labels, measurements_a=measurements)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, measurements_col=None,
                        *args, **kwargs):
        labels = _pd.Categorical(df[measurements_col])
        labels = labels.categories.values

        new_var = cls(name, coordinates, labels,
                      measurements=df[measurements_col].values)
        return new_var

    def fill_pyvista_cube(self, cube, prefix=None, simulations=False):
        self.measurements_a.fill_pyvista_cube(
            cube, self.name + " - measurements")
        self.predicted.fill_pyvista_cube(
            cube, self.name + " - predicted")
        self.entropy.fill_pyvista_cube(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_cube(
            cube, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_cube(
                cube, self.name, simulations=simulations)

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        self.measurements_a.fill_pyvista_points(
            points, self.name + " - measurements")
        self.predicted.fill_pyvista_points(
            points, self.name + " - predicted")
        self.entropy.fill_pyvista_points(
            points, self.name + " - entropy")
        self.uncertainty.fill_pyvista_points(
            points, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_points(
                points, self.name, simulations=simulations)

    def fill_pyvista_blocks(self, cube, prefix=None, simulations=False):
        self.measurements_a.fill_pyvista_blocks(
            cube, self.name + " - measurements")
        self.predicted.fill_pyvista_blocks(
            cube, self.name + " - predicted")
        self.entropy.fill_pyvista_blocks(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_blocks(
            cube, self.name + " - uncertainty")

        for comp in self.labels:
            self.components[comp].fill_pyvista_blocks(
                cube, self.name, simulations=simulations)


class OrderedRockType(RockTypeVariable):
    _ZARR_ATTRS = RockTypeVariable._ZARR_ATTRS + ("implicit_values",)

    def __init__(self, name, coordinates, labels=None, measurements_a=None,
                 measurements_b=None):
        super().__init__(name, coordinates, labels, measurements_a,
                         measurements_b)
        self._length = 1

        if measurements_a is None:
            # Labels-only construction (prediction targets, reloading from
            # disk): no measured contacts, all implicit values missing.
            self.implicit_values = self._Attribute(coordinates)
            return

        measurements_a = _np.asarray(measurements_a)
        if measurements_b is None:
            measurements_b = measurements_a
        else:
            measurements_b = _np.asarray(measurements_b)

        # Labels may have been derived from the measurements by the parent.
        self.implicit_values = self._Attribute(
            coordinates,
            self._implicit_values(measurements_a, measurements_b, self.labels))

    @staticmethod
    def _implicit_values(measurements_a, measurements_b, labels):
        """Where each pair of measurements sits in the label sequence.

        A function of the labels, so it has to be recomputed whenever they
        change — slicing away a category shifts every position after it.
        """
        # Built as a float array from the start; the previous
        # ``-0.5 * _np.ones_like(measurements_a)`` crashed on string inputs.
        implicit_values = _np.full(len(measurements_a), -0.5)
        for i in range(len(labels[:-1])):
            implicit_values = _np.where(
                (measurements_a == labels[i])
                & (measurements_b == labels[i + 1]),
                i,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i + 1])
                & (measurements_b == labels[i]),
                i,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i])
                & (measurements_b == labels[i]),
                i - 0.5,
                implicit_values
            )
            implicit_values = _np.where(
                (measurements_a == labels[i + 1])
                & (measurements_b == labels[i + 1]),
                i + 0.5,
                implicit_values
            )
            # implicit_values[(measurements_a == labels[i])
            #                 & (measurements_b == labels[i + 1])] = i
            # implicit_values[(measurements_a == labels[i + 1])
            #                 & (measurements_b == labels[i])] = i
            # implicit_values[(measurements_a == labels[i])
            #                 & (measurements_b == labels[i])] = i - 0.5
            # implicit_values[(measurements_a == labels[i + 1])
            #                 & (measurements_b == labels[i + 1])] = i + 0.5

        return implicit_values

    def get_measurements(self):
        values = self.implicit_values.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    def __getitem__(self, item):
        new_obj = super().__getitem__(item)
        implicit = self.implicit_values[item]
        if new_obj.measurements_a._has_content():
            # positions in the label sequence, and the slice may have dropped
            # labels, so these are recomputed rather than carried over
            implicit.values = self._implicit_values(
                new_obj.measurements_a.to_numpy(),
                new_obj.measurements_b.to_numpy(),
                new_obj.labels)
        new_obj.implicit_values = implicit
        return new_obj


class BinaryVariable(_Variable):
    _ZARR_ATTRS = ("indicator", "measurements", "weights", "predicted",
                   "probability", "entropy", "uncertainty",
                   "latent_mean", "latent_variance")
    _LABEL_KIND = "categories"
    _ZARR_HAS_SIMS = True

    def __init__(self, name, coordinates, labels=None, measurements=None):
        super().__init__(name, coordinates)
        n_data = coordinates.n_data

        # Coerce to a numpy array: with a plain Python list the element-wise
        # comparisons below (indicator and weights) silently collapse to
        # scalars and the indicators stay NaN.
        if measurements is not None:
            measurements = _np.asarray(measurements)

        if labels is None:
            if measurements is None:
                raise Exception("either the labels or measurements"
                                "must be provided")
            cat = _pd.Categorical(measurements)
            labels = cat.categories.values

        self.labels = labels
        self._length = 1
        if len(labels) != 2:
            raise ValueError(f"There must be exactly 2 labels - found {len(labels)}.")

        self.indicator = self._Attribute(
            coordinates, _np.array([_np.nan]*n_data))
        if measurements is None:
            self.measurements = self._Attribute.encoded(
                coordinates, labels=labels)
            self.weights = self._Attribute(coordinates)
        else:
            # their own categories: an `AnomalyVariable` labels everything that
            # is not the anomaly `_dummy`, but the measurements say what it was
            self.measurements = self._Attribute.encoded(
                coordinates, measurements)
            self.weights = self._Attribute(coordinates, _np.ones(n_data))
            self.indicator.values[measurements == labels[0]] = 1
            self.indicator.values[measurements == labels[1]] = 0

        self.predicted = self._Attribute.encoded(coordinates, labels=labels)
        self.probability = self._Attribute(coordinates, _np.zeros(n_data))
        self.entropy = self._Attribute(coordinates)
        self.uncertainty = self._Attribute(coordinates)

        self.latent_mean = self._Attribute(coordinates)
        self.latent_variance = self._Attribute(coordinates)

        self.simulations = None

        if measurements is not None:
            for label in labels:
                idx = measurements == label
                n_in_label = _np.sum(idx)
                if n_in_label > 0:
                    self.weights.values[idx] = \
                        n_data / (n_in_label * len(labels))

    def get_measurements(self):
        values = self.indicator.values.copy()[:, None]
        has_value = (~ _np.isnan(values)) * 1.0
        values[_np.isnan(values)] = 0
        return values, has_value

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels)
        return new_var

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        new_obj.average = self.probability[item]
        new_obj.indicator = self.indicator[item]
        new_obj.latent_mean = self.latent_mean[item]
        new_obj.latent_variance = self.latent_variance[item]
        new_obj.predicted = self.predicted[item]
        new_obj.entropy = self.entropy[item]
        new_obj.uncertainty = self.uncertainty[item]
        new_obj.measurements = self.measurements[item]
        new_obj.weights = self.weights[item]

        if self.simulations is not None:
            new_obj.simulations = _storage.ArrayStore.from_numpy(
                _np.asarray(self.simulations)[item])

        return new_obj

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.probability.coordinates = coordinates
        self.indicator.coordinates = coordinates
        self.latent_mean.coordinates = coordinates
        self.latent_variance.coordinates = coordinates
        self.predicted.coordinates = coordinates
        self.entropy.coordinates = coordinates
        self.uncertainty.coordinates = coordinates
        self.measurements.coordinates = coordinates
        self.weights.coordinates = coordinates

    def as_data_frame(self, measurements=True, latent=True,
                      predictions=True, simulations=True):
        df = _pd.DataFrame({})

        if measurements:
            df[self.name + "_measurements"] = self.measurements.to_numpy()
            df[self.name + "_weights"] = self.weights.values.to_numpy()

        if predictions:
            df[self.name + "_predicted"] = self.predicted.to_numpy()
            df[self.name + "_probability"] = self.probability.values.to_numpy()
            df[self.name + "_entropy"] = self.entropy.values.to_numpy()
            df[self.name + "_uncertainty"] = self.uncertainty.values.to_numpy()

        if latent:
            df[self.name + "_latent_mean"] = self.latent_mean.values.to_numpy()
            df[self.name + "_latent_variance"] = \
                self.latent_variance.values.to_numpy()

        for i, values in _selected_simulations(self.simulations,
                                              simulations):
            df[self.name + "_sim_" + str(i)] = values

        return df

    def update(self, idx, **kwargs):
        prob = kwargs["probability"].numpy()
        mean = kwargs["mean"].numpy()
        var = kwargs["variance"].numpy()
        entropy = kwargs["entropy"].numpy()
        uncertainty = kwargs["uncertainty"].numpy()
        sims = kwargs["simulations"].numpy()

        if len(prob.shape) > 1:
            prob = prob[:, 0]
            mean = mean[:, 0]
            var = var[:, 0]
            # entropy = entropy[:, 0]
            # uncertainty = uncertainty[:, 0]
            sims = sims[:, 0, :]

        label_idx = _np.zeros(prob.shape, dtype=int)  # positive class
        label_idx[prob < 0.5] = 1  # negative class

        # the label's position is the code
        self.predicted.values[idx] = label_idx
        self.latent_mean.values[idx] = mean
        self.latent_variance.values[idx] = var
        self.entropy.values[idx] = entropy
        self.uncertainty.values[idx] = uncertainty
        self.probability.values[idx] = prob

        self.simulations[idx, :] = sims

    def allocate_simulations(self, n_sim):
        self.simulations = _storage.ArrayStore.allocate(
            (self.coordinates.n_data, n_sim), dtype=float, fill_value=_np.nan,
            owner=self.coordinates)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col, positive_class):
        labels = _pd.Categorical(df[col])
        labels = labels.categories.values.tolist()

        pos = None
        for i, label in enumerate(labels):
            if label == positive_class:
                pos = i
        labels.pop(pos)
        labels.append(positive_class)
        labels = labels[::-1]

        new_var = cls(name, coordinates, labels,
                      measurements=df[col].values)
        return new_var

    def fill_pyvista_cube(self, cube, prefix=None, simulations=False):
        self.indicator.fill_pyvista_cube(
            cube, self.name + " - indicator")
        self.latent_mean.fill_pyvista_cube(
            cube, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_cube(
            cube, self.name + " - latent variance")
        self.predicted.fill_pyvista_cube(
            cube, self.name + " - predicted")
        self.probability.fill_pyvista_cube(
            cube, self.name + " - probability")
        self.entropy.fill_pyvista_cube(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_cube(
            cube, self.name + " - uncertainty")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_cube(cube, self.name + " - simulation %d" % i)

    def fill_pyvista_points(self, points, prefix=None, simulations=False):
        self.indicator.fill_pyvista_points(
            points, self.name + " - indicator")
        self.latent_mean.fill_pyvista_points(
            points, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_points(
            points, self.name + " - latent variance")
        self.predicted.fill_pyvista_points(
            points, self.name + " - predicted")
        self.probability.fill_pyvista_points(
            points, self.name + " - probability")
        self.entropy.fill_pyvista_points(
            points, self.name + " - entropy")
        self.uncertainty.fill_pyvista_points(
            points, self.name + " - uncertainty")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_points(points, self.name + " - simulation %d" % i)

    def fill_pyvista_blocks(self, cube, prefix=None, simulations=False):
        self.indicator.fill_pyvista_blocks(
            cube, self.name + " - indicator")
        self.latent_mean.fill_pyvista_blocks(
            cube, self.name + " - latent mean")
        self.latent_variance.fill_pyvista_blocks(
            cube, self.name + " - latent variance")
        self.predicted.fill_pyvista_blocks(
            cube, self.name + " - predicted")
        self.probability.fill_pyvista_blocks(
            cube, self.name + " - probability")
        self.entropy.fill_pyvista_blocks(
            cube, self.name + " - entropy")
        self.uncertainty.fill_pyvista_blocks(
            cube, self.name + " - uncertainty")

        for i, values in _selected_simulations(self.simulations, simulations):
            col = self._Attribute(self.coordinates, values)
            col.fill_pyvista_blocks(cube, self.name + " - simulation %d" % i)


class AnomalyVariable(BinaryVariable):
    def __init__(self, name, coordinates, label, measurements=None):
        labels = [label, "_dummy"]
        super().__init__(name, coordinates, labels, measurements)

    @classmethod
    def from_data_frame(cls, name, coordinates, df, col, positive_class):
        new_var = cls(name, coordinates, positive_class,
                      measurements=df[col].values)
        return new_var

    @classmethod
    def from_variable(cls, coordinates, variable):
        new_var = cls(variable.name, coordinates, variable.labels[0])
        return new_var


class _SpatialData(_TreeNode):
    """Abstract class for spatial data in general"""

    def __init__(self):
        self._n_dim = None
        self._bounding_box = None
        self._n_data = None
        self._diagonal = None
        self.variables = {}
        self.metadata = {}

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------ #
    # the container as a node (see `_TreeNode`)
    # ------------------------------------------------------------------ #
    @property
    def _node_name(self):
        # the container is the root, whatever else it may call itself
        return ""

    def child_nodes(self):
        return dict(self.variables)

    def own_leaves(self):
        """The metadata columns, under a root of their own.

        `_metadata` is reserved rather than mixed in with the variables: a
        fold that means to write every modelled column must be able to leave
        the air/rock code and the cross-validation fold out of it, and a
        metadata column may legitimately share a name with a variable.
        """
        for name, column in self.metadata.items():
            yield (METADATA_ROOT, name), column

    def _resolve(self, path):
        if len(path) == 2 and path[0] == METADATA_ROOT:
            return self.metadata.get(path[1])
        return super()._resolve(path)

    @property
    def rows_per_location(self):
        """Rows the model evaluates for each location of this object.

        One, except where a location fans out into several — a block with
        discretization. Prediction divides the batch size by this, so that
        `prediction_batch_size` counts the rows actually handed to the model
        rather than meaning something different for every container.
        """
        return 1

    def add_metadata(self, name, values, labels=None):
        """
        Attaches point-wise information to this object's locations.

        Metadata is anything known per location that the models do not use —
        an air/solid code, a cross-validation fold, a sample weight. It follows
        the object through subsetting, `as_data_frame()` and `to_zarr()`, but
        is never modeled: use a variable for that.

        Parameters
        ----------
        name : str
            Column name. An existing column with this name is replaced.
        values : array-like
            One value per data location. Text is stored as integer codes.
        labels : list, optional
            The categories `values` indexes into, when the codes are given
            directly rather than the text they stand for.
        """
        values = _np.asarray(values)
        if labels is None and values.dtype.kind not in "OUS":
            self.metadata[name] = _Attribute(self, values, dtype=values.dtype)
        else:
            self.metadata[name] = _Attribute.encoded(self, values, labels)

    def get_metadata(self, name):
        """
        The values of a metadata column, as an array.

        A text column comes back as its labels, not as the codes it is stored
        as. Use `obj.metadata[name]` for the column itself, which carries the
        gridding and plotting helpers.
        """
        if name not in self.metadata:
            raise ValueError(
                f"there is no metadata column named {name}; "
                f"found {list(self.metadata.keys())}")
        return self.metadata[name].to_numpy()

    def _check_three_dimensional(self):
        """A surface can only be assigned to locations that have a height."""
        if self.n_dim != 3:
            raise DimensionMismatchError(
                f"a surface can only be assigned to three-dimensional "
                f"locations; this {type(self).__name__} has {self.n_dim}")

    def assign_from_surface(self, surface, name, labels=("above", "below"),
                            uncovered=_np.nan):
        """
        Records which side of a surface each location lies on.

        The surface must be a sheet — open, and single valued, so that "above"
        and "below" mean something: a topography, a seam roof, a weathering
        front. Each location is compared with the sheet's elevation directly
        over it, interpolated across the triangle its (x, y) falls in.
        Locations beyond the sheet's edge are left empty, the surface having
        nothing to say about them.

        The answer is a metadata column — point-wise, and never seen by the
        models, which is what a domain code should be. It follows the
        container through `as_data_frame()` and `to_zarr()`. A point set can
        be cut down by it directly, as
        `data[data.get_metadata("ground") == "below"]`; a grid keeps its shape
        and carries the flag as a column, a grid being a grid.

        Parameters
        ----------
        surface : Surface3D
            The sheet to compare against. A closed body is refused; use
            `assign_from_solid` for one of those.
        name : str
            Name of the metadata column to write. An existing column with
            this name is replaced.
        labels : tuple
            What to call the two sides, in the order (above, below).
        uncovered : float or "raise"
            What to make of a location the sheet does not reach. Its flag is
            left empty either way; the value given here is what a block
            model's `fraction` column records for it — `numpy.nan` by
            default, so an uncovered block cannot pass for an empty one, and
            `0.0` to count it as nothing instead. Pass `"raise"` to refuse a
            surface that does not cover every location, for the cases where
            it is required to.
        """
        self._check_three_dimensional()
        coordinates = _np.asarray(self.coordinates, dtype=float)
        elevation = _gmt.sheet_elevation(_sheet_interpolator(surface), coordinates)
        _check_covered(elevation, uncovered)
        self.add_metadata(name, _side_codes(coordinates[:, 2], elevation),
                          labels=list(labels))

    def assign_from_solid(self, solid, name, labels=("outside", "inside")):
        """
        Records whether each location falls inside a closed body.

        The body must be watertight — an ore envelope, a stope, a dyke — and
        is tested by VTK, through pyvista. Two surfaces are refused, both
        because "inside" is undefined for them: one that is not closed (a
        sheet, or a body with a face missing — use `assign_from_surface` for
        the first), and one whose triangles disagree about which way is out.
        A body that is merely wound inwards, all of it, is unambiguous and is
        turned round rather than refused.

        Vertices are welded before any of that is judged, so a body that is
        closed in space counts as closed however its corners are indexed —
        which many meshes, `pyvista.Cylinder` among them, would otherwise
        fail on.

        The answer is a metadata column, as in `assign_from_surface`.

        Parameters
        ----------
        solid : Surface3D
            The closed body to test against.
        name : str
            Name of the metadata column to write. An existing column with
            this name is replaced.
        labels : tuple
            What to call the two sides, in the order (outside, inside).
        """
        self._check_three_dimensional()
        coordinates = _np.asarray(self.coordinates, dtype=float)
        inside = _gmt.inside_solid(_closed_body(solid), coordinates)
        self.add_metadata(name, inside.astype(_np.int8), labels=list(labels))

    def _metadata_frame(self):
        """The metadata columns as a data frame, empty if there are none."""
        return _pd.DataFrame(
            {name: column.to_numpy()
             for name, column in self.metadata.items()},
            index=_np.arange(self.n_data))

    def _subset_metadata(self, target, item):
        """Copies this object's metadata columns onto a subset of it."""
        for name, column in self.metadata.items():
            target.add_metadata(name, column.values[item], column.labels)

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def n_data(self):
        return self._n_data

    @property
    def diagonal(self):
        return self._bounding_box.diagonal

    def aspect_ratio(self, vertical_exaggeration=1):
        """
        Returns a list with plotly layout data.
        """
        if self._n_dim == 2:
            return _py.aspect_ratio_2d(vertical_exaggeration)
        elif self._n_dim == 3:
            return _py.aspect_ratio_3d(self.bounding_box, vertical_exaggeration)
        else:
            raise ValueError("aspect ratio only available for 2- and "
                             "3-dimensional data objects")

    def draw_bounding_box(self, **kwargs):
        if self._n_dim == 2:
            raise NotImplementedError
        elif self._n_dim == 3:
            return _py.bounding_box_3d(self.bounding_box, **kwargs)
        else:
            raise ValueError("bounding_box only available for 2- and "
                             "3-dimensional data objects")

    def to_zarr(self, path):
        """Persist this container to a single on-disk Zarr store.

        Coordinates and every variable's arrays are written into one Zarr group
        at ``path``, streamed chunk-by-chunk (never fully materialized), along
        with the metadata needed to rebuild the container with :meth:`open`.
        All point-based containers and variable types are supported;
        ``DrillholeData`` is not (it is raw input data, not a prediction
        target).
        """
        group = _zarr.open_group(path, mode="w")
        meta = {"geoml_format": _GEOML_ZARR_FORMAT,
                "container": _write_container(group, self),
                "metadata": _write_metadata(group, self),
                "variables": {}}
        for name, var in self.variables.items():
            meta["variables"][name] = _write_variable(group, var)
        group.attrs["geoml"] = meta
        return path

    @classmethod
    def open(cls, path):
        """Rebuild a container previously saved with :meth:`to_zarr`.

        The stored container type is honoured regardless of which class ``open``
        is called on. Variable arrays are reopened on disk (read/write), so the
        result can be inspected or predicted into again without recomputing.
        """
        group = _zarr.open_group(path, mode="r+")
        meta = dict(group.attrs["geoml"])
        container = _rebuild_container(meta["container"], group)
        _rebuild_metadata(container, group, meta.get("metadata", {}))
        for vmeta in meta["variables"].values():
            _rebuild_variable(container, group, vmeta)
        return container


class _PointBased(_SpatialData):
    """Abstract class for data objects based on points"""
    def __init__(self):
        super().__init__()
        self.coordinates = None
        self.coordinate_labels = None

    def __str__(self):
        s = "Object of class %s with %s data locations\n\n" \
            % (self.__class__.__name__, str(self.n_data))

        if len(self.variables) > 0:
            s += "Variables:\n"
            for name, var in self.variables.items():
                s += "    %s: %s\n" % (name, var.__class__.__name__)
        return s

    def add_continuous_variable(self, name, measurements=None):
        """
        Adds a continuous variable to this point set.

        Parameters
        ----------
        name : str
            Variable name.
        measurements : array-like
            The variable values. Its length must correspond to the number of data locations.
        """
        # self.variables[name] = ContinuousVariable(
        #     name, self, measurements, quantiles=quantiles,
        #     probabilities=probabilities)
        self.variables[name] = ContinuousVariable(name, self, measurements)

    def add_vector_variable(self, name, labels=None, measurements=None):
        self.variables[name] = VectorVariable(name, self, labels, measurements)

    def add_categorical_variable(self, name, labels=None, measurements=None):
        """
        Adds a categorical variable to this point set.

        Parameters
        ----------
        name : str
            Variable name.
        labels : tuple
            The category labels. If None, they are determined from the measurements.
        measurements : array-like
            The variable values. Its length must correspond to the number of data locations.
        """
        self.variables[name] = CategoricalVariable(
            name, self, labels, measurements)

    def add_rock_type_variable(self, name, labels=None, measurements_a=None,
                               measurements_b=None, ordered=False):
        """
        Adds a rock type variable to this point set.

        This type of variable uses two labels for each data point. If `measurements_a` is different from
        `measurements_b`, the point is considered to lie in the contact between rock types. If they are equal, the
        point is considered inside a rock type.

        Parameters
        ----------
        name : str
            Variable name.
        labels : tuple
            The category labels. If `None`, they are determined from the measurements.
        measurements_a : array-like
            The variable values. Its length must correspond to the number of data locations.
        measurements_b : array-like
            The variable values. Its length must correspond to the number of data locations. If `None`, its default
            value is the same as `measurements_a`.
        ordered : bool
            If `True`, the labels are considered to indicate sequential, conformable rock layers. Otherwise they
            are considered independent.
        """
        if ordered:
            self.variables[name] = OrderedRockType(
                name, self, labels, measurements_a, measurements_b)
        else:
            self.variables[name] = RockTypeVariable(
                name, self, labels, measurements_a, measurements_b)

    def add_binary_variable(self, name, labels=None, measurements=None):
        """
        Adds a binary variable to this point set.

        Parameters
        ----------
        name : str
            Variable name.
        labels : tuple
            The category labels. If `None`, they are determined from the measurements.
        measurements : array-like
            The variable values. Its length must correspond to the number of data locations.
        """
        self.variables[name] = BinaryVariable(name, self, labels, measurements)

    def add_anomaly_variable(self, name, label, measurements=None):
        """
        Adds an anomaly variable to this point set.

        An anomaly is a type of binary variable that only has the positive label.

        Parameters
        ----------
        name : str
            Variable name.
        label : str
            The positive class label.
        measurements : array-like
            The variable values. Its length must correspond to the number of data locations.
        """
        self.variables[name] = AnomalyVariable(name, self, label, measurements)

    def add_compositional_variable(self, name, labels, measurements=None):
        """
        Adds a compositional variable to this data set.

        Parameters
        ----------
        name : str
            Variable name.
        labels : tuple
            The labels for each part in the composition.
        measurements : array-like
            A rank 2 matrix containing the compositions. It is assumed to be strictly positive and its rows must
            add to 1.
        """
        self.variables[name] = CompositionalVariable(
            name, self, labels, measurements)

    def get_batched_coordinates(self, index=None):
        if index is None:
            index = _np.arange(self._n_data)

        return self.coordinates[index], None

    def _batch_rows(self, index=None):
        """Rows and aggregation that ``get_batched_coordinates`` yields here.

        Kept apart from the coordinates themselves so that
        ``get_batched_variance`` can match their shape without generating them.
        """
        if index is None:
            return self._n_data, None
        index = _np.asarray(index)
        n = int(_np.count_nonzero(index)) if index.dtype == bool else index.size
        return n, None

    def get_batched_variance(self, index=None):
        """Variance of the input locations, mirroring get_batched_coordinates.

        Zero unless the object was built with an explicit variance — see
        `GaussianData`. Only the requested batch is built: deriving the zeros
        from the coordinates would cost O(n_data) on every batch, which
        dominates prediction on large objects.
        """
        rows, aggregation = self._batch_rows(index)
        return _np.zeros([rows, self._n_dim]), aggregation


class PointData(_PointBased):
    """
        Data represented as points in arbitrary locations.
    """

    def __init__(self, data, coordinates):
        """

        Parameters
        ----------
        data : _pd.DataFrame
        coordinates : str or list
        """
        super().__init__()

        if isinstance(coordinates, str):
            coordinates = [coordinates]

        self._init_coordinates(
            _np.array(data.loc[:, coordinates], ndmin=2, dtype=float),
            coordinates)

    def _init_coordinates(self, coordinates, labels):
        """Sets up the coordinate store and everything derived from it.

        `coordinates` may be a plain array or an existing `ArrayStore`. Arrays
        larger than the storage threshold are spilled to Zarr, so a large point
        cloud need not stay in memory once this object owns it.
        """
        self.coordinate_labels = list(labels)
        if isinstance(coordinates, _storage.ArrayStore):
            self.coordinates = coordinates
        else:
            self.coordinates = _storage.ArrayStore.from_values(
                _np.array(coordinates, ndmin=2, dtype=float), owner=self)

        self._n_data, self._n_dim = self.coordinates.shape
        if self._n_data > 0:
            # chunk-by-chunk, so an on-disk store is never fully materialized
            darr = self.coordinates.as_dask()
            lo, hi = _da.compute(darr.min(axis=0), darr.max(axis=0))
            self._bounding_box = BoundingBox(lo, hi)
        else:
            self._bounding_box = BoundingBox.from_array(
                _np.zeros([2, self.n_dim]))

        self.metadata = {}

    def as_data_frame(self, metadata=True, **kwargs):
        """
        Conversion of a spatial object to a data frame.

        The following kwargs can be used to control the kind of information to include in the DataFrame. They all
        default to `True`.
        - `metadata`: miscellaneous information about each data point.
        - `measurements`: the raw measurements used to create each variable.
        - `latent`: predicted latent variables mean and variance.
        - `predictions`: predicted labels and support information like entropy and uncertainty.
        - `quantiles`: values corresponding to probability thresholds, when applicable.
        - `probability`: predicted probabilities.
        - `simulations`: samples from the predictive distribution.
        """
        df = [_pd.DataFrame(_np.asarray(self.coordinates),
                            columns=self.coordinate_labels)]
        for variable in self.variables.values():
            df.append(variable.as_data_frame(**kwargs))
        df = _pd.concat(df, axis=1)
        if metadata:
            df = _pd.concat([self._metadata_frame(), df], axis=1)
        return df

    @staticmethod
    def default_coordinate_labels(n_dim):
        if n_dim <= 3:
            return ["X", "Y", "Z"][0:n_dim]
        return ["V" + str(i) for i in range(n_dim)]

    @classmethod
    def from_array(cls, coordinates, coordinate_labels=None):
        if coordinate_labels is None:
            coordinate_labels = PointData.default_coordinate_labels(
                coordinates.shape[1])
        # The coordinates go straight into the store: routing them through a
        # DataFrame would materialize a full copy of a large point cloud.
        # Not `cls`: the subclasses that inherit this need more than
        # coordinates (directions, a grid shape), so they would come out
        # half-built. `GaussianData` has its own version.
        new_obj = PointData.__new__(PointData)
        _PointBased.__init__(new_obj)
        new_obj._init_coordinates(coordinates, coordinate_labels)
        return new_obj

    def _subset_coordinates(self, item):
        """A container of this class holding only the rows in `item`."""
        return PointData.from_array(self.coordinates[item],
                                    self.coordinate_labels)

    def __getitem__(self, item):
        self_copy = _copy.deepcopy(self)
        new_obj = self_copy._subset_coordinates(item)
        self_copy._subset_metadata(new_obj, item)
        for name, var in self_copy.variables.items():
            new_obj.variables[name] = var[item]
            new_obj.variables[name].set_coordinates(new_obj)
        return new_obj

    def subset_region(self, min_val, max_val,
                      include_min=None, include_max=None):
        if not (isinstance(min_val, list)
                or isinstance(min_val, tuple)
                or isinstance(min_val, _np.ndarray)):
            min_val = [min_val]
        if not (isinstance(max_val, list)
                or isinstance(max_val, tuple)
                or isinstance(max_val, _np.ndarray)):
            max_val = [max_val]
        if include_min is None:
            include_min = [True] * self.n_dim
        if include_max is None:
            include_max = [False] * self.n_dim
        if not (isinstance(include_min, list)
                or isinstance(include_min, tuple)):
            include_min = [include_min]
        if not (isinstance(include_max, list)
                or isinstance(include_max, tuple)):
            include_max = [include_max]

        checks = (len(min_val) == self.n_dim,
                  len(max_val) == self.n_dim,
                  len(include_min) == self.n_dim,
                  len(include_max) == self.n_dim)
        if not all(checks):
            raise ValueError("all arguments must match the data dimension")

        keep = _np.array([True] * self.n_data)
        for i in range(self.n_dim):
            keep = keep & (self.coordinates[:, i] >= min_val[i])
            if not include_min[i]:
                keep = keep & (self.coordinates[:, i] > min_val[i])
            keep = keep & (self.coordinates[:, i] <= max_val[i])
            if not include_max[i]:
                keep = keep & (self.coordinates[:, i] < max_val[i])

        return self[keep] if sum(keep) > 0 else None

    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        if not self.n_dim == 3:
            raise ValueError("as_pyvista method is only supported "
                             "for 3-dimensional data")

        pv_points = _pv.PolyData(_np.asarray(self.coordinates))

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_points(
                pv_points, simulations=simulations)

        return pv_points

    def spatial_k_fold(self, test_data, k=5, bins=50):
        raise NotImplementedError("under development")
        # setup
        test_dist = _tftools.pairwise_dist(self.coordinates, test_data.coordinates).numpy()
        data_dist = _tftools.pairwise_dist(self.coordinates, self.coordinates)

        dist_bins = _np.linspace(0, _np.max(test_dist), bins)

        # test_dist = _np.ravel(test_dist)
        test_hist = _np.histogram(_np.ravel(test_dist), dist_bins)[0]
        test_ecdf = _np.cumsum(test_hist) / _np.sum(test_hist)

        data_hist = _np.histogram(_np.ravel(data_dist), dist_bins)[0]
        data_ecdf = _np.cumsum(data_hist) / _np.sum(data_hist)

        point_hist = _np.stack(
            [_np.histogram(line, dist_bins)[0] for line in data_dist]
        )
        point_ecdf = _np.cumsum(point_hist, axis=1) / _np.sum(point_hist, axis=1, keepdims=True)

        # optimization
        weights = _tf.Variable(_np.random.normal(scale=0.0001, size=[self.n_data, k]))
        mask = _tf.Variable(_np.random.normal(loc=-3, scale=0.0001, size=[self.n_data, 1]))

        def get_fold_ecdf():
            w = _tf.nn.softmax(weights, axis=1)
            m = _tf.nn.sigmoid(mask)

            total_weight = _tf.reduce_sum(w * (1 - m), axis=0)
            total_m = _tf.reduce_sum(1 - m)

            # total_weight = _tf.reduce_sum(w, axis=0)

            fold_ecdf = _tf.stack(
                [_tf.reduce_sum((1 - w[:, i, None]) * point_ecdf * (1 - m), axis=0) / (total_m - total_weight[i]) for i
                 in range(k)],
                axis=1
            )
            # fold_ecdf = _tf.stack(
            #     [_tf.reduce_sum((1 - w[:, i, None]) * point_ecdf, axis=0) / (self.n_data - total_weight[i])
            #      for i in range(k)],
            #     axis=1
            # )

            return fold_ecdf

        def loss():
            fold_ecdf = get_fold_ecdf()

            w_dist = _tf.reduce_sum(_tf.math.abs(fold_ecdf - test_ecdf[:, None])) / k

            w = _tf.nn.softmax(weights, axis=1)
            entropy = - _tf.reduce_mean(_tf.reduce_sum(w * _tf.math.log(w + 1e-6), axis=1)) * 0.1

            m = _tf.nn.sigmoid(mask)

            avg_weight = _tf.reduce_sum(w * (1 - m), axis=0) / _tf.reduce_sum(1 - m)
            # avg_weight = _tf.reduce_sum(w, axis=0)
            penalty = _tf.reduce_sum(avg_weight ** 2) + _tf.reduce_mean(m ** 2)

            # entropy_2 = - _tf.reduce_sum(avg_weight * _tf.math.log(avg_weight + 1e-6))
            # penalty = - entropy_2 + _tf.reduce_mean(m ** 2)

            # silhouette
            avg_dist_in_cluster = _tf.matmul(data_dist**2, weights) / _tf.reduce_sum(weights, axis=0, keepdims=True)
            avg_dist_out_cluster = _tf.matmul(data_dist**2, 1 - weights) / _tf.reduce_sum(1 - weights, axis=0, keepdims=True)
            silhouette = (avg_dist_out_cluster - avg_dist_in_cluster) / _tf.maximum(avg_dist_in_cluster, avg_dist_out_cluster)
            avg_s = _tf.reduce_sum(_tf.reduce_sum(silhouette * weights, axis=1, keepdims=True) * (1 - m)) / _tf.reduce_sum(1 - m)

            return w_dist + entropy + penalty - avg_s * 0.1

        optimizer = _tf.keras.optimizers.Adam(1e-3)

        n_iter = 1000
        history = []
        for _ in range(n_iter):
            _tftools.training_step(optimizer, loss, [weights, mask])
            history.append(loss().numpy())

        # output
        final_w = _tf.nn.softmax(weights, axis=1).numpy()
        final_mask = _tf.nn.sigmoid(mask).numpy()
        total_weight = _np.sum(final_w * (1 - final_mask), axis=0)
        # total_weight = _np.sum(final_w, axis=0)
        final_folds = _np.argmax(final_w, axis=1)

        final_ecdf = get_fold_ecdf().numpy()

        self.add_metadata('spatial_fold', final_folds)
        self.add_metadata('sample_weight', 1 - final_mask)
        n_points = _np.array([_np.sum(self.get_metadata('spatial_fold') == i) for i in range(k)])

        return history, n_points, dist_bins[:-1], test_ecdf, final_ecdf


class GaussianData(PointData):
    """Points whose locations are uncertain, with a variance per coordinate."""

    def __init__(self, data, coordinates_mean, coordinates_variance):
        super().__init__(data, coordinates_mean)

        if isinstance(coordinates_variance, str):
            coordinates_variance = [coordinates_variance]
        self._init_variance(
            _np.array(data.loc[:, coordinates_variance], ndmin=2, dtype=float),
            coordinates_variance)

    def _init_variance(self, variance, labels):
        """Sets up the variance store, kept alongside the coordinate one."""
        self.variance_labels = list(labels)
        if isinstance(variance, _storage.ArrayStore):
            self.variance = variance
        else:
            self.variance = _storage.ArrayStore.from_values(
                _np.array(variance, ndmin=2, dtype=float), owner=self)

        if self.variance.shape != (self._n_data, self._n_dim):
            raise ValueError(
                "variance must have the same shape as the coordinates - "
                f"expected {(self._n_data, self._n_dim)}, "
                f"got {self.variance.shape}")

    @classmethod
    def from_array(cls, coordinates, coordinates_variance,
                   coordinate_labels=None):
        if coordinate_labels is None:
            coordinate_labels = PointData.default_coordinate_labels(
                coordinates.shape[1])
        var_labels = [s + "_var" for s in coordinate_labels]

        new_obj = cls.__new__(cls)
        _PointBased.__init__(new_obj)
        new_obj._init_coordinates(coordinates, coordinate_labels)
        new_obj._init_variance(coordinates_variance, var_labels)
        return new_obj

    def _subset_coordinates(self, item):
        return GaussianData.from_array(self.coordinates[item],
                                       self.variance[item],
                                       self.coordinate_labels)

    def as_data_frame(self, metadata=True, **kwargs):
        df = super().as_data_frame(metadata=metadata, **kwargs)
        variance = _pd.DataFrame(_np.asarray(self.variance),
                                 columns=self.variance_labels)
        return _pd.concat([df, variance], axis=1)

    def get_batched_variance(self, index=None):
        if index is None:
            index = _np.arange(self._n_data)

        return self.variance[index], None


class _GriddedData(_PointBased):
    """Base class for regular grids; also its own lazy coordinate provider.

    A regular grid's coordinates are the Cartesian product of its per-axis node
    vectors, laid out in a fixed order (axis 0 varying fastest, matching
    ``itertools.product(*axes[::-1])[:, ::-1]``). Materializing the full
    ``(n_data, n_dim)`` array would cost several gigabytes for a large 3D grid,
    so a grid instead regenerates only the requested rows on demand and serves
    as its own ``coordinates`` (``grid.coordinates is grid``). It duck-types a
    2-D ``float64`` ndarray for the operations the rest of the package performs
    on ``coordinates``: ``shape``, row selection (``coords[batch]`` with an
    int/bool array or slice — the prediction hot path), column access
    (``coords[:, i]``), and full materialization via ``__array__``.

    An optional affine ``transform`` ``(origin, matrix)`` is applied to every
    generated row as ``(row - origin) @ matrix + origin``; ``RotatedGrid3D``
    uses it to stay lazy while still following a predictable order.

    Concrete subclasses build the per-axis node vectors and pass them to
    ``__init__``, then fill in ``grid``/``grid_size``/``step_size``/``origin``.
    """

    def __init__(self, axes, labels, transform=None):
        super().__init__()
        self._axes = [_np.asarray(a, dtype=float) for a in axes]
        self._sizes = [len(a) for a in self._axes]
        self._transform = transform

        if isinstance(labels, str):
            labels = [labels]
        self.coordinate_labels = list(labels)
        self.coordinates = self
        self._n_dim = len(self._axes)
        self._n_data = int(_np.prod(self._sizes)) if self._axes else 0
        lo = _np.array([axis.min() for axis in self._axes])
        hi = _np.array([axis.max() for axis in self._axes])
        self._bounding_box = BoundingBox.from_array(_np.stack([lo, hi], axis=0))
        self.metadata = {}

        # grid geometry, filled in by the concrete subclasses
        self.grid = None
        self.grid_size = None
        self.step_size = None
        self.origin = None

    # -- lazy coordinate surface ------------------------------------------- #
    @property
    def shape(self):
        return self._n_data, self._n_dim

    @property
    def dtype(self):
        return _np.dtype(float)

    def __len__(self):
        return self._n_data

    def _generate(self, flat):
        """Materialize the rows at the given flat indices as a 2-D array."""
        remaining = _np.atleast_1d(_np.asarray(flat)).astype(_np.int64,
                                                             copy=False)
        cols = []
        for size, axis in zip(self._sizes, self._axes):
            cols.append(axis[remaining % size])
            remaining = remaining // size
        rows = _np.stack(cols, axis=-1)
        if self._transform is not None:
            origin, matrix = self._transform
            rows = _np.matmul(rows - origin, matrix) + origin
        return rows

    def _resolve_rows(self, key):
        if isinstance(key, slice):
            return _np.arange(self._n_data)[key]
        arr = _np.asarray(key)
        if arr.dtype == bool:
            return _np.flatnonzero(arr)
        return arr

    def __array__(self, dtype=None, copy=None):
        rows = self._generate(_np.arange(self._n_data))
        if dtype is not None and rows.dtype != _np.dtype(dtype):
            rows = rows.astype(dtype)
        return rows

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row_key, col_key = item
            return self._generate(self._resolve_rows(row_key))[:, col_key]
        rows = self._generate(self._resolve_rows(item))
        # match ndarray behaviour: a scalar row index drops the first axis
        if _np.isscalar(item) or (isinstance(item, _np.ndarray)
                                  and item.ndim == 0):
            return rows[0]
        return rows

    # -- container behaviour ----------------------------------------------- #
    def index_data(self, data):
        if data.n_dim != self.n_dim:
            raise DimensionMismatchError(
                f"Data dimension mismatch. Expected dimension {self.n_dim}"
                " and found {data.n_dim}."
            )

        cell_id = [
            _np.ceil((data.coordinates[:, i] - self.grid[i][0]
                       - self.step_size[i]/2) / self.step_size[i]).astype(int)
            for i in range(self.n_dim)
        ]

        return _np.stack(cell_id, axis=1)

    def as_data_frame(self, metadata=True, **kwargs):
        df = [_pd.DataFrame(_np.asarray(self.coordinates),
                            columns=self.coordinate_labels)]
        for variable in self.variables.values():
            df.append(variable.as_data_frame(**kwargs))
        df = _pd.concat(df, axis=1)
        if metadata:
            df = _pd.concat([self._metadata_frame(), df], axis=1)
        for i, s in enumerate(self.coordinate_labels):
            df[f'_{s}'] = self.step_size[i]
        return df

    def subset_region(self, min_val, max_val,
                      include_min=None, include_max=None):
        # A subset of a regular grid is irregular, so it materializes into a
        # PointData (carrying any predicted variables), mirroring
        # PointData.subset_region.
        coords = _np.asarray(self.coordinates)
        if not isinstance(min_val, (list, tuple, _np.ndarray)):
            min_val = [min_val]
        if not isinstance(max_val, (list, tuple, _np.ndarray)):
            max_val = [max_val]
        if include_min is None:
            include_min = [True] * self.n_dim
        if include_max is None:
            include_max = [False] * self.n_dim
        if not isinstance(include_min, (list, tuple)):
            include_min = [include_min]
        if not isinstance(include_max, (list, tuple)):
            include_max = [include_max]

        checks = (len(min_val) == self.n_dim,
                  len(max_val) == self.n_dim,
                  len(include_min) == self.n_dim,
                  len(include_max) == self.n_dim)
        if not all(checks):
            raise ValueError("all arguments must match the data dimension")

        keep = _np.ones(self.n_data, dtype=bool)
        for i in range(self.n_dim):
            keep = keep & (coords[:, i] >= min_val[i])
            if not include_min[i]:
                keep = keep & (coords[:, i] > min_val[i])
            keep = keep & (coords[:, i] <= max_val[i])
            if not include_max[i]:
                keep = keep & (coords[:, i] < max_val[i])

        if not keep.any():
            return None

        self_copy = _copy.deepcopy(self)
        new_obj = PointData.from_array(coords[keep], self.coordinate_labels)
        self._subset_metadata(new_obj, keep)
        for name, var in self_copy.variables.items():
            new_obj.variables[name] = var[keep]
            new_obj.variables[name].set_coordinates(new_obj)
        return new_obj


class Grid1D(_GriddedData):
    """
    Equally spaced points in 1D.

    Attributes
    ----------
    step_size : list
        Distance between grid nodes.
    grid : list
        The grid coordinates.
    grid_size : list
        The number of points in grid.
    """

    def __init__(self, start, n, step=None, end=None, labels=None):
        """
        Initializer for Grid1D.

        Parameters
        ----------
        start :
            Starting point for grid.
        n : int
            Number of grid nodes.
        step :
            Spacing between grid nodes.
        end :
            Last grid point.
        labels : str
            The label for the coordinate.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        if step is not None:
            end = start + (n - 1) * step
        else:
            step = (end - start) / (n - 1)
        grid = _np.linspace(start, end, n, dtype=float)

        if labels is None:
            labels = "X"

        super().__init__([grid], labels)
        self.step_size = [step]
        self.grid = [grid]
        self.grid_size = [int(n)]
        self.origin = start

    def aggregate_categorical(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([x for x in range(int(self.grid_size[0]))])
        cols = ["xid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements_a.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data_2 = raw_data.copy()
        raw_data_2["value"] = \
            data.variables[variable].measurements_b.to_numpy()
        raw_data = _pd.concat([raw_data, raw_data_2],
                              axis=0).reset_index(drop=True)

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = CategoricalVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_binary(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([x for x in range(int(self.grid_size[0]))])
        cols = ["xid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = BinaryVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_numeric(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.arange(int(self.grid_size[0]))[:, None]
        cols = ["xid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])

        # aggregating
        data_2 = raw_data.groupby(cols).mean()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = grid_full.join(data_2.set_index(cols), on=cols) \
            .reset_index(drop=False)
        self.variables[variable] = ContinuousVariable(
            variable, self, data_3["value"].values
        )

    @classmethod
    def from_bounding_box(cls, box, step, margin=0.1, rounding_decimals=0):
        if not isinstance(margin, (list, tuple)):
            margin = (margin, margin)

        dif = box.max[0, 0] - box.min[0, 0]
        new_min = _np.round(box.min[0, 0] - dif * margin[0], rounding_decimals)
        new_max = _np.round(box.max[0, 0] + dif * margin[1], rounding_decimals)
        n = int(_np.ceil((new_max - new_min)/step)) + 1

        label = box.labels[0] if box.labels else None
        return cls(start=new_min, n=n, step=step, labels=label)


class Grid2D(_GriddedData):
    """
    Equally spaced points in 2D.

    Attributes
    ----------
    step_size : list
        Distance between grid nodes.
    grid : list
        The grid coordinates.
    grid_size : list
        The number of points in grid.
    """

    def __init__(self, start, n, step=None, end=None, labels=None):
        """
        Initializer for Grid2D.

        Parameters
        ----------
        start : length 2 array, list, or tuple
            Starting point for grid.
        n : length 2 array, list, or tuple of ints
            Number of grid nodes.
        step : length 2 array, list, or tuple
            Spacing between grid nodes.
        end : length 2 array, list, or tuple
            Last grid point.
        labels : list
            The labels for the coordinates.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        start = _np.array(start)
        n = _np.array(n)
        if step is not None:
            step = _np.array(step)
            end = start + (n - 1) * step
        else:
            end = _np.array(end)
            step = _np.array([(end[0] - start[0]) / (n[0] - 1),
                              (end[1] - start[1]) / (n[1] - 1)])
        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])

        if labels is None:
            labels = ["X", "Y"]

        super().__init__([grid_x, grid_y], labels)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y]
        self.grid_size = [int(num) for num in n]
        self.origin = start

    def aggregate_categorical(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y)
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements_a.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data_2 = raw_data.copy()
        raw_data_2["value"] = \
            data.variables[variable].measurements_b.to_numpy()
        raw_data = _pd.concat([raw_data, raw_data_2],
                              axis=0).reset_index(drop=True)

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = CategoricalVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_binary(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y)
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = BinaryVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_numeric(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y)
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])

        # aggregating
        data_2 = raw_data.groupby(cols).mean()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = grid_full.join(data_2.set_index(cols), on=cols)\
            .reset_index(drop=False)
        self.variables[variable] = ContinuousVariable(
            variable, self, data_3["value"].values
        )

    @classmethod
    def from_bounding_box(cls, box, step, margin=0.1, rounding_decimals=0):
        margin = _np.array(margin)
        if margin.shape == ():
            margin = _np.full([2, 2], margin)
        elif margin.shape == (2,):
            margin = _np.stack([margin]*2, axis=1)

        dif = box.max - box.min
        new_min = _np.round(box.min - dif * margin[0], rounding_decimals)
        new_max = _np.round(box.max + dif * margin[1], rounding_decimals)
        n = _np.ceil((new_max - new_min) / step).astype(int) + 1

        return cls(start=new_min[0], n=n[0], step=step, labels=box.labels)


class Grid3D(_GriddedData):
    """
    Equally spaced points in 3D.

    Attributes
    ----------
    step_size : list
        Distance between grid nodes.
    grid : list
        The grid coordinates.
    grid_size : list
        The number of points in grid.
    """

    def __init__(self, start, n, step=None, end=None, labels=None):
        """
        Initializer for Grid3D.

        Parameters
        ----------
        start : length 2 array, list, or tuple
            Starting point for grid.
        n : length 2 array, list, or tuple of ints
            Number of grid nodes.
        step : length 2 array, list, or tuple
            Spacing between grid nodes.
        end : length 2 array, list, or tuple
            Last grid point.
        labels : list
            The labels for the coordinates.


        Either step or end must be given. If both are given, end is ignored.
        """
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        start = _np.array(start)
        n = _np.array(n)
        if step is not None:
            step = _np.array(step)
            end = start + (n - 1) * step
        else:
            end = _np.array(end)
            step = _np.array([(end[0] - start[0]) / (n[0] - 1),
                              (end[1] - start[1]) / (n[1] - 1),
                              (end[2] - start[2]) / (n[2] - 1)])

        grid_x = _np.linspace(start[0], end[0], n[0])
        grid_y = _np.linspace(start[1], end[1], n[1])
        grid_z = _np.linspace(start[2], end[2], n[2])

        if labels is None:
            labels = ["X", "Y", "Z"]

        super().__init__([grid_x, grid_y, grid_z], labels)
        self.step_size = step.tolist()
        self.grid = [grid_x, grid_y, grid_z]
        self.grid_size = [int(num) for num in n]
        self.origin = start

    def aggregate_categorical(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y, z)
                             for z in range(int(self.grid_size[2]))
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid", "zid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements_a.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data["zid"] = _np.round(
            (data.coordinates[:, 2] - self.grid[2][0]
             - self.step_size[2] / 2) / self.step_size[2])
        raw_data_2 = raw_data.copy()
        raw_data_2["value"] = \
            data.variables[variable].measurements_b.to_numpy()
        raw_data = _pd.concat([raw_data, raw_data_2],
                              axis=0).reset_index(drop=True)

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = CategoricalVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_binary(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y, z)
                             for z in range(int(self.grid_size[2]))
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid", "zid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data["zid"] = _np.round(
            (data.coordinates[:, 2] - self.grid[2][0]
             - self.step_size[2] / 2) / self.step_size[2])

        # counting values inside cells
        raw_data["dummy"] = 0
        data_2 = raw_data.groupby(cols + ["value"]).count()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # determining dominant label
        data_3 = data_2.groupby(cols).idxmax()
        data_3 = data_2.loc[data_3.iloc[:, 0], :]

        # output
        data_4 = grid_full.set_index(cols) \
            .join(data_3.set_index(cols)) \
            .reset_index(drop=True)
        self.variables[variable] = BinaryVariable(
            variable, self, data.variables[variable].labels,
            data_4["value"].values
        )

    def aggregate_numeric(self, data, variable):
        data = data.subset_region(self.bounding_box.min[0],
                                  self.bounding_box.max[0])

        grid_id = _np.array([(x, y, z)
                             for z in range(int(self.grid_size[2]))
                             for y in range(int(self.grid_size[1]))
                             for x in range(int(self.grid_size[0]))])
        cols = ["xid", "yid", "zid"]

        grid_full = _pd.DataFrame(
            _np.concatenate([self.coordinates, grid_id], axis=1),
            columns=self.coordinate_labels + cols)

        # identifying cell id
        raw_data = _pd.DataFrame({
            "value": data.variables[variable].measurements.values.to_numpy(),
        })
        raw_data["xid"] = _np.round(
            (data.coordinates[:, 0] - self.grid[0][0]
             - self.step_size[0] / 2) / self.step_size[0])
        raw_data["yid"] = _np.round(
            (data.coordinates[:, 1] - self.grid[1][0]
             - self.step_size[1] / 2) / self.step_size[1])
        raw_data["zid"] = _np.round(
            (data.coordinates[:, 2] - self.grid[2][0]
             - self.step_size[2] / 2) / self.step_size[2])

        # aggregating
        data_2 = raw_data.groupby(cols).mean()
        data_2.reset_index(level=data_2.index.names, inplace=True)

        # output
        data_3 = grid_full.join(data_2.set_index(cols), on=cols) \
            .reset_index(drop=False)
        self.variables[variable] = ContinuousVariable(
            variable, self, data_3["value"].values
        )

    def make_interpolator(self, coordinates):
        return _gint.cubic_conv_3d(coordinates,
                                   self.grid[0], self.grid[1], self.grid[2])

    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        pv_grid = _pv.ImageData(
            dimensions=self.grid_size,
            spacing=self.step_size,
            origin=self.origin
        )

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_cube(
                pv_grid, simulations=simulations)

        return pv_grid

    def rotation_matrix(self):
        return _np.eye(3)

    def assign_from_surface(self, surface, name, labels=("above", "below"),
                            uncovered=_np.nan):
        """
        As `_SpatialData.assign_from_surface`, reading the sheet once for
        each column of cells rather than once for each cell.

        A grid repeats the same (x, y) at every level — `_generate` varies the
        first axis fastest, so the pair cycles with period n_x * n_y — and a
        sheet depends on nothing else, which makes interpolating it n_z times
        over the same arithmetic n_z times. The columns are generated by the
        same method that generates the rows, so the two cannot fall out of
        step. A `RotatedGrid3D` does not lie on an axis-aligned lattice and
        takes the general path.
        """
        self._check_three_dimensional()
        if self._transform is not None:
            return super().assign_from_surface(surface, name, labels,
                                               uncovered=uncovered)

        n_x, n_y, n_z = self._sizes
        columns = self._generate(_np.arange(n_x * n_y))[:, :2]
        elevation = _np.tile(
            _gmt.sheet_elevation(_sheet_interpolator(surface), columns), n_z)
        height = _np.repeat(self._axes[2], n_x * n_y)
        _check_covered(elevation, uncovered)
        self.add_metadata(name, _side_codes(height, elevation),
                          labels=list(labels))

    @classmethod
    def from_bounding_box(cls, box, step, margin=0.1, rounding_decimals=0):
        margin = _np.array(margin)
        if margin.shape == ():
            margin = _np.full([2, 3], margin)
        elif margin.shape == (2,):
            margin = _np.stack([margin] * 3, axis=1)

        dif = box.max - box.min
        new_min = _np.round(box.min - dif * margin[0], rounding_decimals)
        new_max = _np.round(box.max + dif * margin[1], rounding_decimals)
        n = _np.ceil((new_max - new_min) / step).astype(int) + 1

        return cls(start=new_min[0], n=n[0], step=step, labels=box.labels)


class GridND(_GriddedData):
    """
    Implicit grid in N dimensions.
    """
    def __init__(self, start, n, step=None, end=None, labels=None):
        if (step is None) & (end is None):
            raise ValueError("one of step or end must be given")
        start = _np.array(start)
        n = _np.array(n)
        if step is not None:
            step = _np.array(step)
            end = start + (n - 1) * step
        else:
            end = _np.array(end)
            step = _np.array([(e - st) / (n_ - 1)
                              for st, e, n_ in zip(start, end, n)])

        grids = []
        for st, e, n_ in zip(start, end, n):
            grids.append(_np.linspace(st, e, n_))

        if labels is None:
            labels = [f"X_{i}" for i in range(len(n))]

        super().__init__(grids, labels)

        self.step_size = step.tolist()
        self.grid = grids
        self.grid_size = [int(num) for num in n]
        self.labels = labels

    def __str__(self):
        s = "Object of class %s with %s data locations\n" \
            % (self.__class__.__name__, str(self.n_data))

        return s


class DirectionalData(PointData):
    def __init__(self, data, coordinates, directions):
        """

        Parameters
        ----------
        data : _pd.DataFrame
        coordinates : str or list
        """
        super().__init__(data, coordinates)

        if isinstance(directions, str):
            directions = [directions]
        if len(directions) != self.n_dim:
            raise ValueError("arguments coordinates and directions must"
                             "have the same length")

        self.direction_labels = directions
        self.directions = _np.array(data.loc[:, directions], ndmin=2)

        all_coords = _np.concatenate([self._bounding_box.as_array(),
                                      self.directions],
                                     axis=0)
        self._bounding_box = BoundingBox.from_array(all_coords)

    def as_data_frame(self, full=False):
        """
        Conversion of a spatial object to a data frame.
        """
        df = [_pd.DataFrame(_np.asarray(self.coordinates),
                            columns=self.coordinate_labels),
              _pd.DataFrame(self.directions, columns=self.direction_labels)]
        for variable in self.variables.values():
            df.append(variable.as_data_frame(full))
        df = _pd.concat(df, axis=1)
        return df

    def __getitem__(self, item):
        new_obj = _copy.deepcopy(self)
        coords = _np.array(self.coordinates[item], ndmin=2)
        new_obj.coordinates = _storage.ArrayStore.from_values(
            coords, owner=new_obj)
        new_obj.directions = _np.array(self.directions[item], ndmin=2)
        new_obj._n_data = new_obj.coordinates.shape[0]

        all_coords = _np.concatenate([coords, new_obj.directions], axis=0)
        new_obj._bounding_box, new_obj._diagonal = bounding_box(all_coords)

        for name, var in new_obj.variables.items():
            new_obj.variables[name] = var[item]
            new_obj.variables[name].set_coordinates(new_obj)
        return new_obj

    @classmethod
    def from_azimuth(cls, data, coordinates, azimuth):
        azimuth = data[azimuth]
        data = data.copy()
        data["dX"] = _np.sin(azimuth / 180 * _np.pi)
        data["dY"] = _np.cos(azimuth / 180 * _np.pi)
        return cls(data, coordinates, ["dX", "dY"])

    @classmethod
    def from_planes(cls, data, coordinates, azimuth, dip):
        # conversions
        dip = -data[dip].values * _np.pi / 180
        azimuth = (90 - data[azimuth].values) * _np.pi / 180
        strike = azimuth - _np.pi / 2

        # dip and strike vectors
        dipvec = _np.concatenate([
            _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
            _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
            _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)
        strvec = _np.concatenate([
            _np.array(_np.cos(strike), ndmin=2).transpose(),
            _np.array(_np.sin(strike), ndmin=2).transpose(),
            _np.zeros([dipvec.shape[0], 1])], axis=1)

        # result
        vecs = _np.concatenate([dipvec, strvec], axis=0)
        vecs = _pd.DataFrame(vecs, columns=["dX", "dY", "dZ"])
        data = _pd.concat([data, data], axis=0).reset_index(drop=True)
        data = _pd.concat([data, vecs], axis=1)
        return cls(data, coordinates, ["dX", "dY", "dZ"])

    @classmethod
    def from_azimuth_and_dip(cls, data, coordinates, azimuth, dip):
        dip = -data[dip] * _np.pi / 180
        azimuth = (90 - data[azimuth]) * _np.pi / 180

        dipvec = _np.concatenate([
            _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
            _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
            _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1)

        # result
        vecs = _pd.DataFrame(dipvec, columns=["dX", "dY", "dZ"])
        data = _pd.concat([data, vecs], axis=1)
        return cls(data, coordinates, ["dX", "dY", "dZ"])

    @classmethod
    def from_normals(cls, data, coordinates, azimuth, dip):
        n_data = data.shape[0]
        plane_dirs = cls.from_planes(data, coordinates, azimuth, dip)
        vec1 = plane_dirs.directions[0:n_data, :]
        vec2 = plane_dirs.directions[n_data:(2 * n_data), :]

        normalvec = vec1[:, [1, 2, 0]] * vec2[:, [2, 0, 1]] \
                    - vec1[:, [2, 0, 1]] * vec2[:, [1, 2, 0]]
        normalvec = _np.apply_along_axis(
            lambda x: x / _np.sqrt(_np.sum(x ** 2)),
            axis=1,
            arr=normalvec)

        # result
        vecs = _pd.DataFrame(- normalvec, columns=["dX", "dY", "dZ"])
        data = _pd.concat([data, vecs], axis=1)
        return cls(data, coordinates, ["dX", "dY", "dZ"])


def batch_index(n_data, batch_size):
    n_batches = int(_np.ceil(n_data / batch_size))
    idx = [_np.arange(i * batch_size,
                      _np.minimum((i + 1) * batch_size,
                                  n_data))
           for i in range(n_batches)]
    return idx


def export_planes(coordinates, dip, azimuth, filename, size=1):
    # conversions
    dip = -dip * _np.pi / 180
    azimuth = (90 - azimuth) * _np.pi / 180
    strike = azimuth - _np.pi / 2

    # dip and strike vectors
    dipvec = _np.concatenate([
        _np.array(_np.cos(dip) * _np.cos(azimuth), ndmin=2).transpose(),
        _np.array(_np.cos(dip) * _np.sin(azimuth), ndmin=2).transpose(),
        _np.array(_np.sin(dip), ndmin=2).transpose()], axis=1) * size
    strvec = _np.concatenate([
        _np.array(_np.cos(strike), ndmin=2).transpose(),
        _np.array(_np.sin(strike), ndmin=2).transpose(),
        _np.zeros([dipvec.shape[0], 1])], axis=1) * size

    points = _np.stack([
        coordinates + dipvec,
        coordinates - 0.5 * strvec - 0.5 * dipvec,
        coordinates + 0.5 * strvec - 0.5 * dipvec
    ], axis=0)
    points = _np.reshape(points, [3 * points.shape[1], points.shape[2]],
                         order="F")
    idx = _np.reshape(_np.arange(points.shape[0]), coordinates.shape)

    # export
    with open(filename, 'w') as out_file:
        out_file.write(
            str(points.shape[0]) + " " + str(idx.shape[0]) + "\n")
        for line in points:
            out_file.write(" ".join(str(elem) for elem in line) + "\n")
        for line in idx:
            out_file.write(" ".join(str(elem) for elem in line) + "\n")


class Section3D(PointData):
    def __init__(self, center, azimuth, dip, width, height, n_x, n_y,
                 coordinate_labels=("X", "Y", "Z")):
        grid = Grid2D(start=[- width/2, - height/2],
                      end=[width/2, height/2],
                      n=[n_x, n_y])
        base_coords = _np.concatenate(
            [grid.coordinates, _np.zeros([grid.n_data, 1])], axis=1)

        # azimuth = azimuth * _np.pi / 180
        # dip = - dip * _np.pi / 180
        # ry = _np.reshape(_np.array(
        #     [1, 0, 0,
        #      0, _np.cos(dip), -_np.sin(dip),
        #      0, _np.sin(dip), _np.cos(dip)]
        # ), [3, 3])
        # rz = _np.reshape(_np.array(
        #     [_np.cos(azimuth), _np.sin(azimuth), 0,
        #      -_np.sin(azimuth), _np.cos(azimuth), 0,
        #      0, 0, 1]
        # ), [3, 3])
        # rotation_matrix = _np.matmul(rz, ry).T
        rotation_matrix = _gmt.rotation_matrix(azimuth, dip)

        center = _np.array(center, ndmin=2)
        rotated_coords = _np.matmul(base_coords, rotation_matrix) + center

        df = _pd.DataFrame(rotated_coords, columns=coordinate_labels)
        super().__init__(df, coordinate_labels)
        self.grid_shape = [n_x, n_y]

    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        n_x, n_y = self.grid_shape
        faces = []
        for i in range(n_x - 1):
            for j in range(n_y - 1):
                p_1 = j * n_y + i
                p_2 = p_1 + 1
                p_3 = (j + 1) * n_y + i + 1
                p_4 = p_3 - 1
                faces.append([4, p_1, p_2, p_3, p_4])
        faces = _np.stack(faces)

        pv_surf = _pv.PolyData(_np.asarray(self.coordinates), faces)

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_points(
                pv_surf, simulations=simulations)

        return pv_surf


def _sheet_interpolator(surface):
    """Prepares a surface to be asked its elevation, refusing a closed body.

    A closed body stands at two heights over most of its footprint, so it has
    no elevation in the sense meant here -- and `matplotlib` takes such a
    triangulation without complaint, answering with whichever of the two
    sheets it happens to find, which is why this is checked before it is
    handed over.
    """
    if surface.closed:
        raise ValueError(
            "this surface is closed -- every edge belongs to two triangles -- "
            "so there is no single elevation above a location, and 'above' "
            "and 'below' do not describe it; use assign_from_solid for a body")

    return _gmt.sheet_interpolator(
        _np.asarray(surface.coordinates, dtype=float), surface.triangles)


def _uncovered_rule(uncovered):
    """Splits the `uncovered` argument into refuse-or-not and a fill value."""
    if isinstance(uncovered, str):
        if uncovered != "raise":
            raise ValueError(
                "uncovered takes 'raise', or the value to record where the "
                "sheet does not reach; got %r" % uncovered)
        return True, _np.nan
    return False, float(uncovered)


def _check_covered(elevation, uncovered):
    """Refuses a sheet that leaves locations out, when asked to.

    Returns the value to record for those locations, which is of no use where
    there is nothing numeric to record it in -- only a block model's fraction
    column has room for it, the flag being empty either way.
    """
    refuse, fill = _uncovered_rule(uncovered)
    if refuse:
        missing = int(_np.sum(_np.isnan(elevation)))
        if missing > 0:
            raise ValueError(
                "the surface does not reach %d of the %d locations; pass "
                "uncovered=0.0, or numpy.nan, to record those as unknown "
                "rather than refuse them" % (missing, elevation.shape[0]))
    return fill


def _side_codes(height, elevation):
    """0 above the sheet, 1 below it, -1 where the sheet does not reach.

    The comparison is false wherever the elevation is NaN, so those start out
    as "above" and are corrected; -1 is what the metadata layer reads as
    missing, and decodes to the empty string.
    """
    codes = _np.where(height < elevation, 1, 0).astype(_np.int8)
    codes[_np.isnan(elevation)] = -1
    return codes


def _closed_body(solid):
    """A body as a pyvista mesh, refusing one that is not watertight.

    Checked here rather than left to `select_interior_points`, which would
    repeat the check for every chunk of a block model's sub-blocks.
    """
    if not solid.closed:
        open_edges = _gmt.open_edges(solid.coordinates, solid.triangles)
        raise ValueError(
            "this surface is not closed -- %d of its edges belong to a single "
            "triangle, so it is a sheet, or a body with a face missing -- and "
            "it has no inside to test for; use assign_from_surface for a "
            "sheet" % open_edges)

    # A closed body still says nothing about which side is out unless its
    # triangles agree, and the test reads a disagreement as a hole rather
    # than as an error, so it is caught here instead.
    if not solid.consistent:
        reversed_edges = _gmt.reversed_edges(solid.coordinates,
                                             solid.triangles)
        raise ValueError(
            "this surface is closed, but its triangles disagree about which "
            "way is out -- %d of its edges are walked the same way round by "
            "both triangles sharing them -- so part of it would be read as a "
            "hole; reverse the offending triangles, or rebuild the surface "
            "from as_pyvista().compute_normals(consistent_normals=True, "
            "auto_orient_normals=True)" % reversed_edges)

    # A body can be closed and consistent and still be wound inwards, which
    # the test answers the exact complement of. Nothing is ambiguous about
    # one of those -- only its idea of "out" is reversed -- so it is turned
    # round rather than refused. The variables are left behind with it: the
    # test wants the shape and nothing else.
    triangles = _np.asarray(solid.triangles)
    if solid._signed_volume < 0:
        triangles = triangles[:, ::-1]

    faces = _np.concatenate(
        [_np.full([triangles.shape[0], 1], 3, int), triangles], axis=1)
    return _pv.PolyData(_np.asarray(solid.coordinates, dtype=float),
                        faces.ravel())


class Mesh3D(_PointBased):
    """
    A triangulated surface: vertices, the triangles indexing them, normals.

    The primitive `Surface3D` and `Solid3D` are built on, and the only one of
    the three that promises nothing about its shape — which is what a mesh
    must be allowed to be while it is still being repaired. What it does do
    is measure itself as it is built, so that everything downstream can ask
    rather than work it out again: `area`, and whether it is `closed` and
    `consistent`. Those cost a few milliseconds on a mesh of tens of
    thousands of triangles.

    `mesh3d(points, triangles, normals)` builds whichever of the three the
    geometry calls for, and is what the readers use.

    Attributes
    ----------
    area : float
        The surface area, whether or not the mesh closes.
    closed : bool
        Whether every edge is shared by two triangles, so that the mesh
        bounds a volume. Vacuously true of an empty mesh.
    consistent : bool
        Whether the triangles agree about which way is out. A closed mesh
        that is not consistent bounds nothing that can be tested.
    """

    def __init__(self, points, triangles, normals):
        super().__init__()

        if points.shape[1] != 3:
            raise ValueError("points must be an array with 3 columns")
        if triangles.shape[1] != 3:
            raise ValueError("triangles must be an array with 3 columns")
        if normals.shape[1] != 3:
            raise ValueError("normals must be an array with 3 columns")

        self.coordinates = points
        self.triangles = triangles
        self.normals = normals

        self._n_dim = 3
        self._n_data = self.coordinates.shape[0]
        if self._n_data > 0:
            self._bounding_box = BoundingBox.from_array(self.coordinates)
        else:
            self._bounding_box = BoundingBox.from_array(
                _np.zeros([2, self.n_dim]))

        self.area = _gmt.area(points, triangles)
        self.closed = _gmt.open_edges(points, triangles) == 0
        self.consistent = _gmt.reversed_edges(points, triangles) == 0
        self._signed_volume = _gmt.signed_volume(points, triangles)

    def _polydata(self):
        """The bare geometry as a pyvista mesh, carrying no variables."""
        triangles = _np.asarray(self.triangles)
        faces = _np.concatenate(
            [_np.full([triangles.shape[0], 1], 3, int), triangles], axis=1)
        return _pv.PolyData(_np.asarray(self.coordinates, dtype=float),
                            faces.ravel())

    def split(self):
        """
        The mesh's connected pieces, each as an object of its own.

        A boolean operation readily answers with a body in several pieces —
        an ore shell cut in two by a fault — and each piece is a body in its
        own right, while together they are still one legitimate mesh. This is
        how to take them apart; each piece comes back as whichever class its
        own geometry calls for.

        Returns
        -------
        pieces : list
            One mesh per connected piece, longest-standing order. A mesh
            already in one piece returns `[self]`.
        """
        count, labels = _gmt.components(self.coordinates, self.triangles)
        if count <= 1:
            return [self]

        points = _np.asarray(self.coordinates, dtype=float)
        triangles = _np.asarray(self.triangles)
        normals = _np.asarray(self.normals)

        pieces = []
        for piece in range(count):
            keep = labels == piece
            if not keep.any():
                continue
            used = _np.unique(triangles[keep])
            index = _np.zeros(points.shape[0], dtype=int)
            index[used] = _np.arange(used.size)
            pieces.append(mesh3d(points[used], index[triangles[keep]],
                                 normals[used]))
        return pieces

    def heal(self, hole_size=None):
        """
        A repaired copy of this mesh.

        Three things are put right, in the order that works: coincident
        vertices are welded, so that seams stop reading as boundaries; holes
        smaller than `hole_size` are covered over; and the triangles are made
        to agree about which way is out, then turned to face outward. That
        last step is not optional — filling a hole leaves the new triangles
        wound however they came, which would leave the mesh closed and still
        untestable.

        What comes back is whichever class the repaired geometry calls for,
        which may be the same one, and may be an empty `Mesh3D` if nothing
        survived. Healing is not guaranteed: a mesh with a hole larger than
        `hole_size`, or one self-intersecting, can come back no better.

        Parameters
        ----------
        hole_size : float, optional
            The largest hole to cover, in the mesh's own units. None to weld
            and reorient only, leaving every boundary where it is.

        Returns
        -------
        mesh : Mesh3D, Surface3D or Solid3D
        """
        mesh = self._polydata().clean()
        if hole_size is not None:
            mesh = mesh.fill_holes(float(hole_size))
        mesh = mesh.compute_normals(consistent_normals=True,
                                    auto_orient_normals=True).triangulate()

        if mesh.n_points == 0 or mesh.n_cells == 0:
            return Mesh3D(_np.zeros([0, 3]), _np.zeros([0, 3], dtype=int),
                          _np.zeros([0, 3]))

        points = _np.asarray(mesh.points, dtype=float)
        triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        return mesh3d(points, triangles,
                      _gmt.vertex_normals(points, triangles))

    @classmethod
    def from_dxf(cls, filename):
        """
        Reads a triangulated surface from a DXF file.

        Three ways of writing a triangulation are understood. The `MESH`
        entity that `export_dxf` writes already holds a vertex list and the
        faces that index into it, and is taken as it stands. `POLYFACE`
        meshes and loose `3DFACE` entities instead repeat the coordinates of
        every corner they share, and are welded back into shared vertices,
        matched to six decimal places. Faces with more than three corners are
        split into a fan of triangles.

        Every mesh in the file is read and the results are concatenated, so a
        file holding several bodies comes back as one surface in several
        disconnected pieces. Each `MESH` entity keeps its own vertices, while
        the welded entities share one vertex list, so pieces that meet there
        are joined. Entities nested inside blocks are not searched.

        Only the geometry is read: see `export_dxf` on what a DXF file has no
        room for.

        Parameters
        ----------
        filename : str
            Path of the file to read.

        Returns
        -------
        mesh : Surface3D, Solid3D or Mesh3D
            Whichever the geometry read calls for, with normals computed from
            the triangles (see `geometry.vertex_normals`), since a DXF file
            carries none.
        """
        model = _ezdxf.readfile(filename).modelspace()

        blocks = [(_np.asarray(mesh.vertices, dtype=float),
                   [list(face) for face in mesh.faces])
                  for mesh in model.query("MESH")]

        # A 3DFACE carries no vertex list at all -- each one spells out the
        # coordinates of its corners, so a vertex shared by six triangles
        # arrives six times -- and a POLYFACE is read one face at a time.
        # The merger is what turns those back into shared vertices.
        merger = _MeshVertexMerger()
        for polyline in model.query("POLYLINE"):
            if not polyline.is_poly_face_mesh:
                continue
            body = _MeshVertexMerger.from_polyface(polyline)
            vertices = _np.asarray(body.vertices, dtype=float)
            for face in body.faces:
                merger.add_face(vertices[list(face)].tolist())
        for face in model.query("3DFACE"):
            merger.add_face(face.wcs_vertices())
        if len(merger.faces) > 0:
            blocks.append((_np.asarray(merger.vertices, dtype=float),
                           [list(face) for face in merger.faces]))

        if len(blocks) == 0:
            raise ValueError(
                "no triangulated surface found in %s: the file holds none of "
                "the MESH, POLYFACE or 3DFACE entities a surface is written "
                "as" % filename)

        points, triangles, start = [], [], 0
        for block_points, block_faces in blocks:
            points.append(block_points)
            triangles.append(_gmt.fan_triangulation(block_faces) + start)
            start += block_points.shape[0]
        points = _np.concatenate(points, axis=0)
        triangles = _np.concatenate(triangles, axis=0)

        return mesh3d(points, triangles,
                      _gmt.vertex_normals(points, triangles))

    def export_dxf(self, filename, offset=None):
        """
        Writes this surface to a DXF file, as a single MESH entity.

        A `MESH` holds the vertex list and the triangles that index into it,
        so the surface comes back from `from_dxf` exactly as it went out --
        nothing is welded and there is no ceiling on the number of vertices,
        unlike the `POLYFACE` mesh DXF is more often written as.

        Only the geometry travels. A DXF file has nowhere to put the
        variables and metadata a surface carries: `to_zarr` keeps a container
        whole, and `as_pyvista` carries the values onto a mesh object.

        Parameters
        ----------
        filename : str
            Path of the file to write.
        offset : array-like
            Added to the coordinates on the way out, as in
            `export_micromine`, for writing into a local grid. It is not
            recorded in the file, so reading it back gives the shifted
            coordinates.
        """
        points = _np.asarray(self.coordinates, dtype=float)
        if offset is not None:
            points = points + _np.asarray(offset, dtype=float).reshape([1, 3])

        document = _ezdxf.new()
        mesh = document.modelspace().add_mesh()
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = points.tolist()
            mesh_data.faces = _np.asarray(self.triangles, dtype=int).tolist()
        document.saveas(filename)

    def export_micromine(self, points_filename="points",
                         triangles_filename="triangles",
                         offset=[0, 0, 0], **kwargs):
        points_df = [
            _pd.DataFrame({"id": _np.arange(self.n_data)}),
            _pd.DataFrame(self.coordinates, columns=["EAST", "NORTH", "RL"])
        ]
        for variable in self.variables.values():
            points_df.append(variable.as_data_frame(**kwargs))
        points_df = _pd.concat(points_df, axis=1)
        points_df["EAST"] += offset[0]
        points_df["NORTH"] += offset[1]
        points_df["RL"] += offset[2]
        points_df.to_csv(points_filename + ".csv", index=False)

        triangles_df = _pd.DataFrame(
            self.triangles, columns=["PointId1", "PointId2", "PointId3"])
        triangles_df.to_csv(triangles_filename + ".csv", index=False)

    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        faces = _np.concatenate(
            [_np.full([self.triangles.shape[0], 1], 3, int), self.triangles],
            axis=1
        )

        pv_surf = _pv.PolyData(self.coordinates, faces.ravel())

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_points(
                pv_surf, simulations=simulations)

        return pv_surf


class Surface3D(Mesh3D):
    """
    A mesh that does not close: a sheet, with an edge to it.

    A topography, a seam roof, a weathering front, a fault plane — anything
    that has two sides rather than an inside. The promise is checked where it
    is made, so `assign_from_surface` need only be given one of these.
    """

    def __init__(self, points, triangles, normals):
        super().__init__(points, triangles, normals)

        # an empty mesh keeps every promise, having nothing to break them
        # with, and is what an operation that comes to nothing returns
        if self.n_data > 0 and self.closed:
            raise MeshTypeError(
                "this mesh closes -- every edge belongs to two triangles -- "
                "so it bounds a volume rather than being a sheet, and has no "
                "single elevation above a location; build a Solid3D, or "
                "mesh3d(...) for whichever the geometry calls for")

    def intersection(self, solid):
        """
        The part of this sheet lying inside a body.

        The sheet is cut where it crosses the body's surface, so what comes
        back follows the body's shape rather than the triangles' — the piece
        of a fault plane inside an ore envelope, say. A sheet lying wholly
        outside comes back empty.

        Parameters
        ----------
        solid : Solid3D
            The body to cut against.

        Returns
        -------
        surface : Surface3D
        """
        return self._clipped(solid, inside=True)

    def difference(self, solid):
        """
        The part of this sheet lying outside a body.

        The complement of `intersection`: together the two hold the whole
        sheet. A sheet lying wholly inside comes back empty.

        Parameters
        ----------
        solid : Solid3D
            The body to cut away.

        Returns
        -------
        surface : Surface3D
        """
        return self._clipped(solid, inside=False)

    def _clipped(self, solid, inside):
        """The sheet cut by a body, keeping one side of it."""
        if not isinstance(solid, Solid3D):
            raise MeshTypeError(
                "a sheet can only be cut by a Solid3D, a body being what has "
                "an inside to cut against; got %s" % type(solid).__name__)

        if self.n_data == 0 or solid.n_data == 0:
            return self if inside is False else _empty_surface()

        clipped = self._polydata().clip_surface(solid._polydata(),
                                                invert=inside)
        if clipped.n_points == 0 or clipped.n_cells == 0:
            return _empty_surface()

        clipped = clipped.triangulate()
        points = _np.asarray(clipped.points, dtype=float)
        triangles = clipped.faces.reshape(-1, 4)[:, 1:]
        return Surface3D(points, triangles,
                         _gmt.vertex_normals(points, triangles))


class Solid3D(Mesh3D):
    """
    A mesh that closes: a body, with an inside.

    An ore envelope, a stope, a dyke, a contoured shell. Both promises are
    checked where they are made — the mesh must close, and its triangles must
    agree which way is out — so `assign_from_solid` need only be given one of
    these, and `volume` always means something.

    A body wound inwards is turned round on the way in rather than refused:
    nothing about it is ambiguous, only reversed. The triangles are what get
    reversed; the normals are left as they were given.

    Attributes
    ----------
    volume : float
        The volume enclosed, always positive. Zero for an empty body, which
        is what an intersection of two bodies that do not meet comes to.
    """

    def __init__(self, points, triangles, normals):
        super().__init__(points, triangles, normals)

        if self.n_data > 0 and not self.closed:
            raise NotClosedError(
                "this mesh does not close -- %d of its edges belong to a "
                "single triangle -- so it has no inside; build a Surface3D "
                "for a sheet, or heal() it if a body is what was meant"
                % _gmt.open_edges(points, triangles))
        if not self.consistent:
            raise InconsistentMeshError(
                "this mesh closes, but its triangles disagree about which "
                "way is out -- %d of its edges are walked the same way round "
                "by both triangles sharing them -- so what it bounds is not "
                "defined; heal() puts this right"
                % _gmt.reversed_edges(points, triangles))

        if self._signed_volume < 0:
            self.triangles = _np.ascontiguousarray(
                _np.asarray(self.triangles)[:, ::-1])
            self._signed_volume = -self._signed_volume
        self.volume = self._signed_volume

    def union(self, other):
        """
        A body covering everything either of these two covers.

        Two bodies that do not meet make a union in two pieces, which is one
        legitimate body; `split()` takes it apart.
        """
        return self._combine(other, "union")

    def intersection(self, other):
        """
        A body covering what both of these two cover, empty where they do not
        meet at all.
        """
        return self._combine(other, "intersection")

    def difference(self, other):
        """
        A body covering what this one covers and the other does not.

        Where `other` lies wholly inside this one the answer is this body
        with a cavity in it, which is written as both surfaces, the inner one
        turned inwards — so `volume` comes to the difference of the two, and
        a location in the cavity tests as outside.
        """
        return self._combine(other, "difference")

    def _combine(self, other, operation):
        """Works the boolean out, VTK being unable to when they do not cross.

        VTK answers with nothing at all whenever the two surfaces have no
        face crossing another -- whether they stand apart or one contains the
        other, and with no error either way -- so an empty answer is not
        taken at face value but worked out from which body contains which.
        """
        if isinstance(other, Surface3D):
            return self._cut_by_sheet(other, operation)
        if not isinstance(other, Solid3D):
            raise MeshTypeError(
                "a body can only be combined with another Solid3D, or cut by "
                "a Surface3D; got %s" % type(other).__name__)

        if self.n_data > 0 and other.n_data > 0:
            combined = getattr(self._polydata(),
                               "boolean_" + operation)(other._polydata())
            if combined.n_points > 0:
                points = _np.asarray(combined.points, dtype=float)
                triangles = combined.triangulate().faces.reshape(-1, 4)[:, 1:]
                return Solid3D(points, triangles,
                               _gmt.vertex_normals(points, triangles))

        return self._without_crossing(other, operation)

    def _without_crossing(self, other, operation):
        """The answer where neither surface cuts the other."""
        here, there = _empty_solid(), _empty_solid()
        if self.n_data > 0:
            here = self
        if other.n_data > 0:
            there = other

        mine_inside = theirs_inside = False
        if here.n_data > 0 and there.n_data > 0:
            mine_inside = bool(_gmt.inside_solid(
                there._polydata(), _np.asarray(here.coordinates)[:1])[0])
            theirs_inside = bool(_gmt.inside_solid(
                here._polydata(), _np.asarray(there.coordinates)[:1])[0])

        if operation == "union":
            if mine_inside:
                return there
            if theirs_inside or there.n_data == 0:
                return here
            if here.n_data == 0:
                return there
            return _joined([here, there])

        if operation == "intersection":
            if mine_inside:
                return here
            if theirs_inside:
                return there
            return _empty_solid()

        if mine_inside:
            return _empty_solid()
        if theirs_inside:
            # a body with a cavity: the inner surface turned inwards, so the
            # volumes subtract and a location in the hollow reads as outside
            return _joined([here, there], reverse=[False, True])
        return here


    def _cut_by_sheet(self, sheet, operation):
        """This body divided by a sheet, keeping what lies under or over it.

        A sheet has no volume of its own, so there is nothing to add to or
        subtract from directly. What it does have, being single valued, is an
        underneath: extruded downwards past everything here, it becomes the
        ground beneath itself, and the ordinary body-to-body operations do
        the rest. `intersection` therefore keeps what lies below the sheet
        and `difference` what lies above it.
        """
        if operation == "union":
            raise MeshTypeError(
                "a sheet encloses no volume, so there is nothing in it to "
                "add to a body; use intersection to keep what lies below it, "
                "or difference to keep what lies above")

        if sheet.n_data == 0:
            return _empty_solid() if operation == "intersection" else self
        if self.n_data == 0:
            return _empty_solid()

        if not _gmt.single_valued(sheet.coordinates, sheet.triangles):
            raise NotSingleValuedError(
                "this sheet folds over, so 'below' and 'above' it are not "
                "one region each and a body cannot be divided by it; a DTM3D "
                "is the kind of surface this works with")

        here = self.bounding_box
        there = sheet.bounding_box
        low, high = _np.ravel(here.min), _np.ravel(here.max)
        sheet_low, sheet_high = _np.ravel(there.min), _np.ravel(there.max)
        if _np.any(sheet_low[:2] > low[:2]) \
                or _np.any(sheet_high[:2] < high[:2]):
            raise MeshTypeError(
                "the sheet does not reach across the whole body -- it spans "
                "x %g to %g and y %g to %g, against the body's x %g to %g "
                "and y %g to %g -- so it would cut at its own edge and leave "
                "a face that means nothing; extend it, or trim the body first"
                % (sheet_low[0], sheet_high[0], sheet_low[1], sheet_high[1],
                   low[0], high[0], low[1], high[1]))

        return self._combine(_ground_below(sheet, here), operation)


def _ground_below(sheet, box):
    """The body under a sheet, reaching below everything in `box`.

    The extrusion arrives with its walls and its floor wound against its lid,
    so it is reoriented before it can be a body at all.
    """
    top = float(_np.ravel(sheet.bounding_box.max)[2])
    floor = float(min(_np.ravel(box.min)[2],
                      _np.ravel(sheet.bounding_box.min)[2]))
    drop = (top - floor) + max(abs(top - floor), 1.0)

    body = sheet._polydata().extrude((0, 0, -drop), capping=True)
    body = body.clean().compute_normals(
        consistent_normals=True, auto_orient_normals=True).triangulate()

    points = _np.asarray(body.points, dtype=float)
    triangles = body.faces.reshape(-1, 4)[:, 1:]
    return Solid3D(points, triangles, _gmt.vertex_normals(points, triangles))


def _empty_solid():
    """A body enclosing nothing, which is what an empty answer looks like."""
    return Solid3D(_np.zeros([0, 3]), _np.zeros([0, 3], dtype=int),
                   _np.zeros([0, 3]))


def _empty_surface():
    """A sheet covering nothing, for an operation that clips everything away."""
    return Surface3D(_np.zeros([0, 3]), _np.zeros([0, 3], dtype=int),
                     _np.zeros([0, 3]))


def _joined(meshes, reverse=None):
    """One mesh holding several, each keeping its own vertices."""
    if reverse is None:
        reverse = [False] * len(meshes)

    points, triangles, normals, start = [], [], [], 0
    for mesh, turn in zip(meshes, reverse):
        block = _np.asarray(mesh.triangles) + start
        points.append(_np.asarray(mesh.coordinates, dtype=float))
        triangles.append(block[:, ::-1] if turn else block)
        normals.append(_np.asarray(mesh.normals))
        start += mesh.n_data

    return mesh3d(_np.concatenate(points, axis=0),
                  _np.ascontiguousarray(_np.concatenate(triangles, axis=0)),
                  _np.concatenate(normals, axis=0))


class DTM3D(Surface3D):
    """
    A terrain: a sheet standing at one height over each (x, y).

    A digital terrain model, and the shape most of the surfaces in a project
    have — a topography, a seam roof, a weathering front. The promise is that
    it never folds back over itself, checked where the object is made, which
    is what lets a body be divided into what lies under it and what lies over
    it, and what makes "the elevation here" a question with one answer.

    Not what `mesh3d` returns: an ordinary sheet is a `Surface3D` unless a
    terrain is asked for, this being a promise to make rather than a fact to
    detect. Triangles standing exactly vertical are allowed, a cliff being
    single valued everywhere but along the line of its face.
    """

    def __init__(self, points, triangles, normals):
        super().__init__(points, triangles, normals)

        if self.n_data > 0 and not _gmt.single_valued(points, triangles):
            raise NotSingleValuedError(
                "this sheet folds over: some of its triangles face the "
                "ground and some face away from it, so it stands at more "
                "than one height over some of its footprint and is not a "
                "terrain. A Surface3D holds it without that promise")


def mesh3d(points, triangles, normals):
    """
    A mesh of whichever class its geometry calls for.

    A `Solid3D` where the triangles close and agree which way is out, a
    `Surface3D` where they do not close, and a plain `Mesh3D` where they
    close but disagree — the one case that is neither a sheet nor a body, and
    what `Mesh3D.heal` exists for.

    Parameters
    ----------
    points : array
        An (n, 3) array of vertex coordinates.
    triangles : array
        An (m, 3) array of vertex indices.
    normals : array
        An (n, 3) array of vertex normals.

    Returns
    -------
    mesh : Surface3D, Solid3D or Mesh3D
    """
    if _gmt.open_edges(points, triangles) > 0:
        return Surface3D(points, triangles, normals)
    if _gmt.reversed_edges(points, triangles) > 0:
        return Mesh3D(points, triangles, normals)
    return Solid3D(points, triangles, normals)


def _below_sheet(surface):
    """The test a sheet poses of a location: is it under the surface?"""
    interpolator = _sheet_interpolator(surface)

    def below(coordinates):
        return coordinates[:, 2] < _gmt.sheet_elevation(interpolator,
                                                        coordinates)
    return below


def _within_body(solid):
    """The test a closed body poses of a location: is it inside?"""
    mesh = _closed_body(solid)
    return lambda coordinates: _gmt.inside_solid(mesh, coordinates)


def _mesh_test(mesh):
    """Which side of `mesh` a location falls on, whichever kind it is."""
    if isinstance(mesh, Solid3D):
        return _within_body(mesh)
    if isinstance(mesh, Surface3D):
        return _below_sheet(mesh)
    raise MeshTypeError(
        "a %s says nothing about which side a block is on: a sheet has an "
        "above and a below, a body an inside and an outside, and a mesh that "
        "is neither has no sides to speak of. `heal()` is what turns one into "
        "a body" % type(mesh).__name__)


def _sub_block_shares(blocks, test, rows=None):
    """The share of each block's sub-blocks that `test` accepts.

    Walked in chunks: a 5 x 5 x 5 discretization is 125 sub-blocks for
    every location, and materializing all of them at once is what the
    batching in `get_batched_coordinates` exists to avoid. `rows` narrows it
    to the blocks worth asking about; the rest come back `nan`.
    """
    per_block = blocks.rows_per_location
    chunk = max(1, 1000000 // per_block)
    if rows is None:
        rows = _np.arange(blocks._n_data)

    shares = _np.full(blocks._n_data, _np.nan)
    for start in range(0, len(rows), chunk):
        index = rows[start:start + chunk]
        coordinates, _ = blocks.get_batched_coordinates(index)
        # block-major, so every block's sub-blocks are one row
        accepted = test(coordinates).reshape([len(index), per_block])
        shares[index] = accepted.mean(axis=1)
    return shares


def _blocks_from_surface(blocks, base, surface, name, labels, fraction,
                         uncovered):
    """`assign_from_surface` for anything made of blocks."""
    base(blocks, surface, name, labels, uncovered=uncovered)

    if fraction is not None:
        shares = _sub_block_shares(blocks, _below_sheet(surface))
        # a sheet that misses the centre describes the block no better
        # than it describes a location, and left the flag empty for the
        # same reason -- so the fraction says so rather than reading 0
        missed = _np.asarray(blocks.metadata[name].values).ravel() < 0
        shares[missed] = _uncovered_rule(uncovered)[1]
        blocks.add_metadata(fraction, shares)


def _blocks_from_solid(blocks, base, solid, name, labels, fraction):
    """`assign_from_solid` for anything made of blocks."""
    base(blocks, solid, name, labels)

    if fraction is not None:
        blocks.add_metadata(
            fraction, _sub_block_shares(blocks, _within_body(solid)))


def _blockdata(cls):
    # Decorator to extend functionality of some classes
    # Used to avoid multiple inheritance

    old_init = cls.__init__

    def new_init(self, start, n, step=None, end=None, labels=None, discretization=None):
        old_init(self, start=start, n=n, step=step, end=end, labels=labels)
        if discretization is None:
            discretization = [1] * self.n_dim
        self.discretization = discretization

        lo = _np.array([axis.min() for axis in self.grid])
        hi = _np.array([axis.max() for axis in self.grid])
        self._bounding_box = BoundingBox(
            lo - _np.array(self.step_size) / 2,
            hi + _np.array(self.step_size) / 2,
        )

        sub_grid = _np.array(
            list(_iter.product(
                *[_np.arange(d) for d in self.discretization[::-1]]
            )),
            dtype=float)[:, ::-1]
        sub_grid -= (_np.array(self.discretization)[None, :] - 1) / 2
        sub_grid *= _np.array(self.step_size)[None, :]
        sub_grid /= _np.array(self.discretization)[None, :]
        self.sub_grid = sub_grid

    def discretized_coordinates(self, index):
        center = _np.array([g[i] for g, i in zip(self.grid, index)])[None, :]
        return self.sub_grid + center

    def inducing_grid(self, index):
        center = _np.array([g[i] for g, i in zip(self.grid, index)])[None, :]
        grid = self.sub_grid
        discr = _np.array(self.discretization)[None, :]
        # grid = grid * (discr + 1) / (discr - 1) + center
        grid = grid * (discr + 2) / (discr + 1) + center
        return PointData.from_array(grid)

    def get_batched_coordinates(self, index):
        if index is None:
            index = _np.arange(self._n_data)

        centers = _np.asarray(self.coordinates[index])
        # block-major, one temporary instead of one per block: block b owns
        # rows b*k to (b+1)*k, which is what `_aggregate` averages back
        coords = (centers[:, None, :] + self.sub_grid[None, :, :])
        coords = coords.reshape(-1, centers.shape[1])

        splits = None if _np.prod(self.discretization) == 1 else len(index)
        return coords, splits

    @property
    def rows_per_location(self):
        return int(_np.prod(self.discretization))

    def _batch_rows(self, index=None):
        # one row per discretization point of every block in the batch
        n, _ = _PointBased._batch_rows(self, index)
        n_sub = int(_np.prod(self.discretization))
        return n * n_sub, (None if n_sub == 1 else n)

    # captured before being replaced, as `old_init` is: the block versions
    # only add the partial blocks on top of what the grid already answers
    base_assign_from_surface = cls.assign_from_surface
    base_assign_from_solid = cls.assign_from_solid

    def _sub_block_fraction(self, test):
        """The share of each block's sub-blocks that `test` accepts."""
        return _sub_block_shares(self, test)

    def assign_from_surface(self, surface, name, labels=("above", "below"),
                            fraction=None, uncovered=_np.nan):
        """
        As `Grid3D.assign_from_surface`, measuring the partial blocks on
        request.

        The flag in `name` follows the block centre, as a whole-block code
        does everywhere else. Name a `fraction` column as well and the share
        of each block lying below the sheet is measured over the sub-blocks
        `discretization` already defines — what a tonnage near surface needs,
        where counting a half-buried block whole is the error.

        Where the sheet reaches part of a block but not all of it, the
        sub-blocks past its edge count as not below. Where it does not reach
        the block at all — the centre included, which is what leaves the flag
        empty — the fraction is `uncovered` instead of a measurement.

        Parameters
        ----------
        surface : Surface3D
            The sheet to compare against.
        name : str
            Name of the metadata column holding the whole-block flag.
        labels : tuple
            What to call the two sides, in the order (above, below).
        fraction : str, optional
            Name of a second metadata column, to hold the share of each block
            below the sheet. Costs `prod(discretization)` queries per block.
        uncovered : float or "raise"
            What the `fraction` column records for a block the sheet does not
            reach: `numpy.nan` by default, so it cannot pass for a block
            genuinely above ground, or `0.0` to count it as nothing. Pass
            `"raise"` to refuse a surface that does not cover every block.
        """
        _blocks_from_surface(self, base_assign_from_surface, surface, name,
                             labels, fraction, uncovered)

    def assign_from_solid(self, solid, name, labels=("outside", "inside"),
                          fraction=None):
        """
        As `_SpatialData.assign_from_solid`, measuring the partial blocks on
        request.

        `fraction` behaves as it does in `assign_from_surface`: the flag in
        `name` follows the block centre, while the column named here holds the
        share of each block's sub-blocks falling inside the body — the share
        of its volume, for the regular sub-blocks a discretization defines.

        Parameters
        ----------
        solid : Surface3D
            The closed body to test against.
        name : str
            Name of the metadata column holding the whole-block flag.
        labels : tuple
            What to call the two sides, in the order (outside, inside).
        fraction : str, optional
            Name of a second metadata column, to hold the share of each block
            inside the body. Costs `prod(discretization)` queries per block.
        """
        _blocks_from_solid(self, base_assign_from_solid, solid, name, labels,
                           fraction)

    cls.__init__ = new_init
    cls.discretized_coordinates = discretized_coordinates
    cls.inducing_grid = inducing_grid
    cls.get_batched_coordinates = get_batched_coordinates
    cls._batch_rows = _batch_rows
    cls.rows_per_location = rows_per_location
    cls._sub_block_fraction = _sub_block_fraction
    cls.assign_from_surface = assign_from_surface
    cls.assign_from_solid = assign_from_solid

    return cls


@_blockdata
class Blocks1D(Grid1D):
    pass


@_blockdata
class Blocks2D(Grid2D):
    pass


@_blockdata
class Blocks3D(Grid3D):
    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        pv_blocks = _pv.ImageData(
            dimensions=_np.array(self.grid_size) + 1,
            spacing=self.step_size,
            origin=_np.array([ax[0] for ax in self.grid]) - _np.array(self.step_size) / 2
        )

        for var in self.variables.keys():
            self.variables[var].fill_pyvista_blocks(
                pv_blocks, simulations=simulations)

        return pv_blocks


def _sub_block_index(discretization):
    """Which sub-block sits where, as integer counts along each axis.

    Axis 0 varies fastest, the order `_blockdata` has always used and the one
    the likelihood's noise is indexed by. Both the sub-block offsets and, in a
    `BlockSet3D`, the children of a split are built from this, so sub-block
    `j` of a block and child `j` of that same block are the same corner of it.
    """
    return _np.array(
        list(_iter.product(*[_np.arange(d) for d in discretization[::-1]])),
        dtype=_np.int64)[:, ::-1]


def _unit_sub_grid(discretization):
    """Sub-block offsets from a block's centre, as fractions of its size.

    The same layout `_blockdata` builds, but divided through by the block so
    that one array serves every size. Scaling it per block is the whole of
    what a variable-size block model has to do differently when it fans out.
    """
    counts = _np.array(discretization)[None, :]
    return (_sub_block_index(discretization) - (counts - 1) / 2) / counts


# a hexahedron's eight corners, in the order VTK reads them
_HEX_CORNERS = _np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                          [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                         dtype=_np.int64)


def _trilinear_weights(discretization):
    """What each of a block's eight corners is worth at each sub-block centre.

    A corner carries what the blocks meeting there say, so reading the corners
    at the sub-blocks is how a child learns the shape running across its
    parent. The layout is symmetric about the centre, so the weights average
    to an eighth apiece and a correction built from them cancels over the
    children -- which is what keeps a block's own estimate the mean of the
    children standing in for it.
    """
    t = _unit_sub_grid(discretization) + 0.5
    return _np.prod(_np.where(_HEX_CORNERS[None, :, :] == 1,
                              t[:, None, :], 1.0 - t[:, None, :]), axis=2)


def _grow(corners, marked, rings):
    """Add `rings` of neighbouring blocks, through the corners blocks share.

    Sparse on purpose: dilating a mask over the base lattice would cost a cell
    for every one the model exists to avoid carrying.
    """
    for _ in range(rings):
        touched = _np.zeros(int(corners.max()) + 1, dtype=bool)
        touched[corners[marked].ravel()] = True
        marked = touched[corners].any(axis=1)
    return marked


class BlockSet3D(PointData):
    """
    Blocks of several sizes, on one integer lattice.

    A block model where the interesting ground can be carried finely and the
    rest coarsely. On a real deposit the ground worth resolving is a small
    part of the volume, and a uniform model at the resolution that part needs
    spends almost all of its cells saying nothing: refining 5 m only where it
    is wanted takes a 29-million-cell model to under 700 000.

    Every block's position and size are whole numbers of a **base cell**, the
    finest the model may go, which is `step / discretization ** max_levels`.
    Working in those integers rather than in metres is what makes it exact:
    blocks meet without a tolerance, a block is a whole number of its own
    children, and regrouping conserves mass to the last digit. It also means
    the model can say which of its answers rest on coarse blocks, which is the
    one thing a mixed-support model has to be able to prove -- see
    `docs/variable-block-models.md`.

    It is built **full**: the blocks tile their box exactly, and every
    operation keeps them that way. Ground to leave out is filtered, not
    removed, so that grouping is always safe -- a half-populated group would
    average over blocks that are not there and quietly weigh the answer wrong.

    `discretization` does two jobs, and they are the same job. It is how
    finely a block is sampled to average it, and it is how a block splits:
    each sub-block becomes a child. So the refinement ratio is the
    discretization, per axis and not necessarily two -- `[2, 2, 1]` refines in
    plan and leaves the bench height alone. Being the same at every level is
    what lets a block of any size fan out into the same number of rows, so the
    model sees one shape whatever it is looking at and nothing downstream has
    to know that levels exist. It costs a coarse block some accuracy in its
    own average, always by overstating how variable it is, which errs towards
    splitting it -- and splitting is what removes the error.

    Parameters
    ----------
    start : array-like
        Centre of the first (coarsest) block.
    n : array-like
        Number of blocks along each axis, at the coarsest level.
    step : array-like
        Size of a coarsest-level block.
    discretization : array-like
        Sub-blocks per axis, used at every level, and so also the ratio a
        block splits by. An axis given 1 is never refined.
    max_levels : int
        How many times a block may be split. Fixes the base cell, and so the
        lattice everything else is counted in.
    labels : list
        Coordinate names.

    Attributes
    ----------
    block_size : array
        `(n_data, 3)`, each block's size in the coordinates' own units. Not
        called `step_size`: that name means one size for the whole object, and
        anything reading it would take the product of this array for a volume.
    block_volume : array
        `(n_data,)`, what each block is worth in a tonnage.
    level : array
        `(n_data,)`, 0 for a coarsest block up to `max_levels` for a base one.
    """

    def __init__(self, start, n, step, discretization=(2, 2, 2), max_levels=3,
                 labels=("X", "Y", "Z")):
        if int(max_levels) < 0:
            raise ValueError("max_levels cannot be negative; got %r"
                             % max_levels)

        self.max_levels = int(max_levels)
        self.discretization = [int(d) for d in discretization]
        if len(self.discretization) != 3 or any(
                d < 1 for d in self.discretization):
            raise ValueError(
                "discretization needs three positive numbers; got %r"
                % (discretization,))
        if self.max_levels > 0 and _np.prod(self.discretization) == 1:
            raise ValueError(
                "a discretization of one sub-block cannot refine anything, so "
                "max_levels=%d would have nothing to do; give a coarser "
                "discretization or max_levels=0" % self.max_levels)

        n = _np.asarray(n, dtype=_np.int64)
        step = _np.asarray(step, dtype=float)
        start = _np.asarray(start, dtype=float)
        if not (n.shape == step.shape == start.shape == (3,)):
            raise ValueError(
                "start, n and step are three numbers each; got %s, %s and %s"
                % (start.shape, n.shape, step.shape))

        # Splitting a block hands each of its sub-blocks a block of its own, so
        # the discretization is also the refinement ratio -- per axis, and not
        # necessarily two. That is what keeps sub-block `j` and child `j` the
        # same corner of the parent, which is what lets the criterion read off
        # a coarse prediction mean anything about the children it would make.
        # It also lets an axis be left alone: [2, 2, 1] refines in plan and
        # keeps the bench height.
        self.base_step = step / self._coarse_size
        # the box corner, not the first centre: a block's origin is its lower
        # corner, which is what makes the lattice arithmetic come out integer
        self.box_corner = start - step / 2
        self.lattice_shape = n * self._coarse_size

        cells = _np.stack(_np.meshgrid(
            *[_np.arange(k) for k in n], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self._origin = cells * self._coarse_size
        self._level = _np.zeros(len(cells), dtype=_np.int64)
        # nothing has been predicted yet, so every block is new
        self._fresh = _np.ones(len(cells), dtype=bool)

        self._sub_grid = _unit_sub_grid(self.discretization)

        super().__init__(
            _pd.DataFrame(self._centres(), columns=list(labels)), list(labels))
        self._bounding_box = BoundingBox(
            self.box_corner,
            self.box_corner + self.lattice_shape * self.base_step)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def _centres(self):
        return self.box_corner + (self._origin + self._size / 2) \
            * self.base_step

    @property
    def _coarse_size(self):
        """A coarsest block's extent, in base cells, along each axis.

        Not the number of sub-blocks, which is `prod(discretization)` and is
        the same at every level -- this is how far a level-0 block reaches
        across the lattice, and so how many base cells it would come to if it
        were split all the way down.
        """
        return _np.array(self.discretization,
                         dtype=_np.int64) ** self.max_levels

    @property
    def _size(self):
        """Each block's extent in base cells, from its level.

        Held as the level rather than the size: a level is one small number
        against three, and the two cannot then disagree.
        """
        ratio = _np.array(self.discretization, dtype=_np.int64)
        return self._coarse_size[None, :] \
            // ratio[None, :] ** self._level[:, None]

    @property
    def block_size(self):
        return self._size * self.base_step

    @property
    def block_volume(self):
        return _np.prod(self.block_size, axis=1)

    @property
    def level(self):
        return self._level

    @property
    def rows_per_location(self):
        return int(_np.prod(self.discretization))

    def is_full(self):
        """Whether the blocks tile their box exactly -- no gap, no overlap.

        Volume alone cannot answer: a gap and an overlap of the same size
        cancel. So the base cells are counted, in an array the shape of the
        lattice, which costs a byte per base cell (29 MB for a 30-million-cell
        model). Construction is full and both `split` and `group` preserve it,
        so this is for checking something that came from elsewhere rather than
        for routine use.
        """
        covered = int(_np.prod(self._size, axis=1).sum())
        if covered != int(_np.prod(self.lattice_shape)):
            return False

        seen = _np.zeros(tuple(self.lattice_shape), dtype=_np.uint8)
        for origin, size in zip(self._origin, self._size):
            seen[origin[0]:origin[0] + size[0],
                 origin[1]:origin[1] + size[1],
                 origin[2]:origin[2] + size[2]] += 1
        return bool(seen.min() == 1 and seen.max() == 1)

    _NO_SUBSET = (
        "a %s cannot be subsetted: it is structurally complete by design, and "
        "removing blocks would leave a group averaging over children that are "
        "no longer there -- the mass-conservation error the lattice exists to "
        "prevent. Exclude ground by value instead, with a metadata column "
        "(`add_metadata`); `predict(..., where=...)` visits part of a model "
        "without making a smaller one. To hand the blocks to something else, "
        "`as_data_frame` carries a size per row, `as_pyvista` writes them as "
        "hexahedra, and `to_zarr` round-trips the lattice whole.")

    def __getitem__(self, item):
        # PointData's would hand back a plain PointData, silently: no origin,
        # no level, no size, so the size columns vanish, `as_pyvista` draws
        # points rather than blocks and `grade_tonnage` refuses. Better to say
        # so here than to return something that looks usable.
        raise TypeError(self._NO_SUBSET % type(self).__name__)

    def subset_region(self, min_val, max_val,
                      include_min=None, include_max=None):
        raise TypeError(self._NO_SUBSET % type(self).__name__)

    # ------------------------------------------------------------------ #
    # refinement
    # ------------------------------------------------------------------ #
    def split(self, mask, carry=True, labels=("X", "Y", "Z")):
        """A new block set with each marked block cut into its own sub-blocks.

        A block becomes `prod(discretization)` children, one per sub-block and
        in the same order, so the values a coarse prediction already holds for
        those sub-blocks describe the blocks this makes.

        A block that was **not** split keeps what was predicted for it: it is
        the same block on the same support, so its value is still the right
        answer and arriving at it again would be work for nothing. The
        children start missing, and `unpredicted()` says which they are, so
        `predict(..., where=...)` visits only them. A parent's value is never
        handed down -- that would manufacture children agreeing exactly, which
        is the one thing refining is meant to find out rather than assume.

        Parameters
        ----------
        mask : array-like
            One boolean per block, or the indices of the blocks to split.
        carry : bool
            Whether to bring the variables and metadata across. `False` gives
            bare geometry, for building a mesh to predict onto from scratch.
        """
        mask = _np.asarray(mask)
        if mask.dtype != bool:
            index = _np.zeros(self.n_data, dtype=bool)
            index[mask] = True
            mask = index
        if mask.shape != (self.n_data,):
            raise ValueError(
                "the mask needs one value per block: got %d for %d"
                % (mask.size, self.n_data))

        finest = self._level >= self.max_levels
        if _np.any(mask & finest):
            raise ValueError(
                "%d block(s) are already at the finest level the lattice "
                "allows (max_levels=%d); build the set with more levels to "
                "go further"
                % (int(_np.count_nonzero(mask & finest)), self.max_levels))

        # one child per sub-block, in the sub-blocks' own order
        corner = _sub_block_index(self.discretization)
        child_size = self._size[mask] // _np.array(self.discretization)
        children = (self._origin[mask][:, None, :]
                    + corner[None, :, :] * child_size[:, None, :]
                    ).reshape(-1, 3)
        child_level = _np.repeat(self._level[mask] + 1, len(corner))

        new = _copy.copy(self)
        new._origin = _np.concatenate([self._origin[~mask], children])
        new._level = _np.concatenate([self._level[~mask], child_level])
        # the blocks this made, so that predicting can be told to visit only
        # them without first being told what was predicted before
        new._fresh = _np.concatenate([
            _np.zeros(int(_np.count_nonzero(~mask)), dtype=bool),
            _np.ones(len(children), dtype=bool)])
        new.variables = {}
        new.metadata = {}
        new._init_coordinates(new._centres(), list(labels))
        # `_init_coordinates` has just measured the box off the coordinates,
        # which are block *centres*: that box is half a block short on every
        # side, and short by a different amount after every split, since
        # refining an edge block moves its centre nearer the boundary. Two
        # sets over the same ground would then disagree about their extent.
        # The real box is the lattice, which no refinement changes.
        new._bounding_box = BoundingBox(
            self.box_corner,
            self.box_corner + self.lattice_shape * self.base_step)

        if carry:
            # A block that was not split is the same block on the same
            # support, so what was predicted for it still stands; only the new
            # children have nothing yet. Metadata goes to the children as
            # well, and by inheritance rather than as missing: it describes the
            # ground, and a child occupies its parent's ground.
            parent = _np.concatenate(
                [_np.flatnonzero(~mask),
                 _np.repeat(_np.flatnonzero(mask), len(corner))])
            for name, variable in self.variables.items():
                new.variables[name] = variable.carry_to(
                    new, ~mask, len(children))
            for name, column in self.metadata.items():
                values = _np.asarray(column.values)[parent]
                fresh = _Attribute(new, values, dtype=values.dtype)
                fresh.labels = column.labels
                new.metadata[name] = fresh
        return new

    def group(self, mask, carry=True, labels=("X", "Y", "Z")):
        """The inverse of `split`: whole families of children, back into the
        parent they came from.

        A block is grouped with its siblings, so the mask must name **every**
        child of a parent or none of them. A partial family would average over
        children that are not there and mis-weight the parent -- the
        mass-conservation error the lattice exists to prevent -- so it is
        refused rather than approximated. That check is what makes conversion
        between supports two-directional: `group` undoes `split` exactly, and
        the blocks tile their box afterwards as they did before.

        A block that was **not** grouped keeps what it holds, exactly as in
        `split` and for the same reason: it is the same block on the same
        support, so its value is still the right answer for it. The parents
        are the ones on new ground, and they come back missing --
        `unpredicted()` names them and `predict(..., where=...)` fills them.

        A parent's value is never averaged from its children. Coarsening is a
        change of support and almost nothing survives it: a parent's spread is
        not its children's, its within-block dispersion is larger by exactly
        what the grouping absorbed, and a category has no mean. The one thing
        that would come across exactly is a realization, and a variable is
        more than its realizations. Metadata does come across, from the first
        child -- it describes the ground rather than the model, and where the
        children disagree about it there is no right answer to be had.

        Parameters
        ----------
        mask : array-like
            One boolean per block, or the indices of the blocks to group.
        carry : bool
            Whether to bring across what the blocks that were not grouped
            hold. `False` gives bare geometry.
        labels : list
            Coordinate names.
        """
        mask = _np.asarray(mask)
        if mask.dtype != bool:
            index = _np.zeros(self.n_data, dtype=bool)
            index[mask] = True
            mask = index
        if mask.shape != (self.n_data,):
            raise ValueError(
                "the mask needs one value per block: got %d for %d"
                % (mask.size, self.n_data))

        coarsest = self._level < 1
        if _np.any(mask & coarsest):
            raise ValueError(
                "%d block(s) are already at the coarsest level the lattice "
                "has and belong to no parent"
                % int(_np.count_nonzero(mask & coarsest)))

        ratio = _np.array(self.discretization, dtype=_np.int64)
        family = self._size[mask] * ratio[None, :]
        parent = (self._origin[mask] // family) * family
        level = self._level[mask] - 1

        # one key a parent, level included: two families of different sizes can
        # share a lower corner, and they are not the same parent
        shape = self.lattice_shape
        key = (((parent[:, 0] * shape[1] + parent[:, 1]) * shape[2]
                + parent[:, 2]) * (self.max_levels + 1) + level)
        _, first, counts = _np.unique(key, return_index=True,
                                      return_counts=True)

        whole = int(_np.prod(self.discretization))
        if _np.any(counts != whole):
            short = _np.flatnonzero(counts != whole)
            raise ValueError(
                "grouping takes whole families: %d parent(s) were named by "
                "only some of their %d children, the first by %d. A partial "
                "family averages over blocks that are not there, which is the "
                "one thing the lattice exists to prevent"
                % (len(short), whole, int(counts[short[0]])))

        new = _copy.copy(self)
        new._origin = _np.concatenate([self._origin[~mask], parent[first]])
        new._level = _np.concatenate([self._level[~mask], level[first]])
        # the blocks this made, on a support nothing has been predicted on
        new._fresh = _np.concatenate([
            _np.zeros(int(_np.count_nonzero(~mask)), dtype=bool),
            _np.ones(len(first), dtype=bool)])
        new.variables = {}
        new.metadata = {}
        new._init_coordinates(new._centres(), list(labels))
        # the lattice is the box, not the spread of the centres -- see `split`
        new._bounding_box = BoundingBox(
            self.box_corner,
            self.box_corner + self.lattice_shape * self.base_step)

        if carry:
            source = _np.concatenate(
                [_np.flatnonzero(~mask), _np.flatnonzero(mask)[first]])
            for name, variable in self.variables.items():
                new.variables[name] = variable.carry_to(
                    new, ~mask, len(first))
            for name, column in self.metadata.items():
                values = _np.asarray(column.values)[source]
                fresh = _Attribute(new, values, dtype=values.dtype)
                fresh.labels = column.labels
                new.metadata[name] = fresh
        return new

    def block_shares(self, split_on=None):
        """How often each decision cuts a block in two.

        A continuous variable contributes one column per cut-off it declares,
        a categorical one per category, and both mean the same thing: the
        share of realizations in which this block's sub-blocks fall on both
        sides of something that matters. A grade is judged against the grades
        someone declared; an indicator against zero, that being where one
        category stops winning and its rival starts.

        Returns a dict of name -> `(n_data,)` array, empty where nothing
        declared a decision to make.
        """
        chosen = list(self.variables) if split_on is None else \
            [str(name) for name in _np.atleast_1d(split_on)]
        shares = {}
        for name in chosen:
            if name not in self.variables:
                raise ValueError(
                    "no variable named %r to split on; found %s"
                    % (name, ", ".join(sorted(self.variables)) or "none"))
            # each variable reports its own -- see `_Variable.split_shares`
            for key, values in self.variables[name].split_shares().items():
                shares[("%s %s" % (name, key)).strip()] = values
        return shares

    def needs_splitting(self, split_on=None, tolerance=0.05):
        """Which blocks hold more than one answer, and so are worth cutting.

        A block whose sub-blocks agree, realization by realization, holds one
        answer however finely it is cut. One whose sub-blocks disagree holds
        two, and cutting is what separates them.

        Note what this does *not* mark: a block the model is merely unsure
        about. Realizations either side of a cut-off are the model not
        knowing, and no amount of cutting will settle that -- the answer to it
        is another drillhole. Only disagreement *within* a realization counts.

        The test is over every decision at once and any one is enough, which
        is the cautious way round on purpose: a block worth splitting for one
        variable is worth splitting whatever the others say. Name `split_on`
        to narrow it -- letting every element of a polymetallic deposit vote
        marks most of the model and gives back the saving.

        Blocks already at the finest level are never marked, there being
        nothing to cut them into.

        Parameters
        ----------
        split_on : str or list, optional
            Which variables get a say. All of them by default.
        tolerance : float
            The share of realizations that must find the block divided. Small
            but not zero, so that one realization in twenty does not carry it.
        """
        shares = self.block_shares(split_on)
        mask = _np.zeros(self.n_data, dtype=bool)
        for values in shares.values():
            mask |= _np.asarray(values, dtype=float) > tolerance
        return mask & (self._level < self.max_levels)

    def _by_level(self):
        """Where the blocks of each level sit, as a sorted table of origins.

        Sorted rather than rasterized on purpose: naming the block covering a
        point is then a `searchsorted`, where painting the base lattice would
        cost a cell for every one the model exists to avoid carrying.
        """
        shape = self.lattice_shape
        tables = []
        for k in range(self.max_levels + 1):
            rows = _np.flatnonzero(self._level == k)
            here = self._origin[rows]
            key = ((here[:, 0] * shape[1] + here[:, 1]) * shape[2]
                   + here[:, 2])
            order = _np.argsort(key, kind="stable")
            tables.append((key[order], rows[order]))
        return tables

    def unbalanced(self, gap=1):
        """Blocks with a neighbour more than `gap` levels finer than they are.

        A block whose own sub-blocks agree is never marked by
        `needs_splitting`, and rightly so -- cutting it would not change the
        answer it gives. But the field can still turn sharply inside it, and
        nothing in the block itself says so. That a *neighbour* was cut twice
        while this block was not cut at all is the evidence, and it lives
        outside the block.

        It matters for what is drawn rather than for what is decided. A
        contour reads a block through its eight corners, so a coarse block
        beside much finer ones is a crude straight guess across a long span
        laid right where the surface runs. Levelling the jump measured three
        times closer to the true surface for 35% more blocks, where refining a
        whole level deeper without it bought almost nothing for 2.6 times as
        many -- deeper refinement widens the jumps as fast as it narrows the
        blocks. `models.refine` therefore cuts these as it goes.

        Blocks already at the finest level are never marked, there being
        nothing to cut them into.

        Parameters
        ----------
        gap : int
            How many levels of difference to tolerate. One is the usual 2:1
            balance: a block may meet blocks one level finer, not two.
        """
        ratio = _np.array(self.discretization, dtype=_np.int64)
        shape = self.lattice_shape
        tables = self._by_level()
        size = self._size
        marked = _np.zeros(self.n_data, dtype=bool)

        # asked from the fine side: a fine block steps one cell past each of
        # its faces and names the coarse block covering that point, which is
        # exact where asking the coarse block what lies beyond its face is not
        # -- a cell of its own size holds blocks that do not touch it
        for fine in range(int(gap) + 1, self.max_levels + 1):
            rows = _np.flatnonzero(self._level == fine)
            if len(rows) == 0:
                continue
            origin, extent = self._origin[rows], size[rows]

            for coarse in range(fine - int(gap)):
                key_of, owner = tables[coarse]
                if len(owner) == 0:
                    continue
                cell = self._coarse_size // ratio ** coarse

                for axis in range(3):
                    for side in (-1, 1):
                        beyond = origin.copy()
                        beyond[:, axis] += (extent[:, axis] if side > 0
                                            else -1)
                        inside = ((beyond[:, axis] >= 0)
                                  & (beyond[:, axis] < shape[axis]))
                        owning = (beyond // cell) * cell
                        key = ((owning[:, 0] * shape[1] + owning[:, 1])
                               * shape[2] + owning[:, 2])
                        at = _np.minimum(_np.searchsorted(key_of, key),
                                         len(key_of) - 1)
                        hit = inside & (key_of[at] == key)
                        marked[owner[at[hit]]] = True

        return marked & (self._level < self.max_levels)

    # ------------------------------------------------------------------ #
    # sample data
    # ------------------------------------------------------------------ #
    def index_data(self, data):
        """Which block each of `data`'s locations falls in.

        One row index per location, `-1` for anything outside the box. Note
        this is **not** what a grid's `index_data` returns -- a cell index per
        axis -- because blocks of several sizes have no per-axis index to
        return. Which block *is* the answer here.

        The lattice makes it cheap: a location's base cell is arithmetic, and
        the block covering that cell is the one whose origin is the cell's
        ancestor at that block's own level, so one `searchsorted` per level
        finds it and every location is settled within `max_levels + 1` of them.
        """
        if data.n_dim != self.n_dim:
            raise DimensionMismatchError(
                "Data dimension mismatch. Expected dimension %d and found %d."
                % (self.n_dim, data.n_dim))

        coordinates = _np.asarray(data.coordinates, dtype=float)
        cell = _np.floor(
            (coordinates - self.box_corner) / self.base_step).astype(_np.int64)
        shape = self.lattice_shape

        found = _np.full(len(coordinates), -1, dtype=_np.int64)
        left = _np.flatnonzero(
            _np.all((cell >= 0) & (cell < shape), axis=1))
        ratio = _np.array(self.discretization, dtype=_np.int64)
        tables = self._by_level()

        for k in range(self.max_levels + 1):
            if len(left) == 0:
                break
            key_of, owner = tables[k]
            if len(owner) == 0:
                continue
            size = self._coarse_size // ratio ** k
            ancestor = (cell[left] // size) * size
            key = ((ancestor[:, 0] * shape[1] + ancestor[:, 1]) * shape[2]
                   + ancestor[:, 2])
            at = _np.minimum(_np.searchsorted(key_of, key), len(key_of) - 1)
            hit = key_of[at] == key
            found[left[hit]] = owner[at[hit]]
            left = left[~hit]

        return found

    @staticmethod
    def _held_by(block, values):
        """`values` against the block holding each, the strays dropped."""
        frame = _pd.DataFrame({"block": block, "value": values})
        return frame[frame["block"] >= 0].dropna(subset=["value"])

    def _dominant(self, block, values):
        """The label most often measured in each block, blank where none was.

        The count decides it, as on a grid; ties go to whichever label sorts
        last, which is arbitrary but has to be something.
        """
        counts = self._held_by(block, values).groupby(
            ["block", "value"]).size().reset_index(name="_n")
        winner = counts.sort_values("_n").groupby("block").tail(1)
        out = _np.full(self.n_data, "", dtype=object)
        out[winner["block"].to_numpy()] = winner["value"].to_numpy()
        return out

    def aggregate_numeric(self, data, variable):
        """The mean of `data`'s measurements over the block holding them."""
        held = self._held_by(
            self.index_data(data),
            data.variables[variable].measurements.values.to_numpy())
        mean = _np.full(self.n_data, _np.nan)
        grouped = held.groupby("block")["value"].mean()
        mean[grouped.index.to_numpy()] = grouped.to_numpy()
        self.variables[variable] = ContinuousVariable(variable, self, mean)

    def aggregate_categorical(self, data, variable):
        """The dominant category in each block.

        Both sides of a contact vote, as on a grid: a boundary measurement
        names the category either side of it and neither is the measurement.
        """
        source = data.variables[variable]
        block = self.index_data(data)
        self.variables[variable] = CategoricalVariable(
            variable, self, source.labels,
            self._dominant(_np.tile(block, 2),
                           _np.concatenate([source.measurements_a.to_numpy(),
                                            source.measurements_b.to_numpy()])))

    def aggregate_binary(self, data, variable):
        """The dominant label in each block."""
        source = data.variables[variable]
        self.variables[variable] = BinaryVariable(
            variable, self, source.labels,
            self._dominant(self.index_data(data),
                           source.measurements.to_numpy()))

    def assign_from_surface(self, surface, name, labels=("above", "below"),
                            fraction=None, uncovered=_np.nan):
        """
        As `Blocks3D.assign_from_surface`, over blocks of several sizes.

        The flag in `name` follows the block centre; naming a `fraction`
        column measures the share of each block below the sheet over the
        sub-blocks `discretization` defines, scaled to each block's own size.
        `crossed_by` asks the same question and answers with the blocks worth
        cutting.
        """
        _blocks_from_surface(self, PointData.assign_from_surface, surface,
                             name, labels, fraction, uncovered)

    def assign_from_solid(self, solid, name, labels=("outside", "inside"),
                          fraction=None):
        """
        As `Blocks3D.assign_from_solid`, over blocks of several sizes.

        `fraction` holds the share of each block's volume inside the body, and
        `crossed_by` turns the same test into the blocks worth cutting.
        """
        _blocks_from_solid(self, PointData.assign_from_solid, solid, name,
                           labels, fraction)

    def crossed_by(self, mesh):
        """Which blocks a mesh passes through, and so which are worth cutting.

        A block is crossed when its sub-blocks fall on **both** sides of the
        mesh -- the question `needs_splitting` asks of a cut-off, asked of
        geometry instead. A topography, a vein wall, a lease boundary: a block
        the surface runs through holds two answers whatever is predicted into
        it, and no amount of prediction will separate them.

        One entirely above or entirely below is left alone however close it
        lies, which is what keeps this from refining a whole domain. Blocks
        already at the finest level are never marked, as elsewhere.

        Hand it to `split`, or give the mesh to `models.refine`, which unions
        this with the other two criteria and repeats until nothing is left to
        cut::

            blocks = blocks.split(blocks.crossed_by(topography))

        A sheet that covers only part of the model counts a sub-block past its
        edge as not below, the way `fraction` does, so a block straddling the
        sheet's own boundary reads as crossed. That is usually wanted -- the
        edge is a real feature of the ground being described -- but it is why
        a sheet should reach across the model when it is not.

        Parameters
        ----------
        mesh : Surface3D or Solid3D
            The sheet or closed body to test against. A `Mesh3D` that is
            neither has no sides and is refused.

        Returns
        -------
        array
            One boolean per block.
        """
        splittable = _np.flatnonzero(self._level < self.max_levels)
        if len(splittable) == 0:
            return _np.zeros(self.n_data, dtype=bool)

        shares = _sub_block_shares(self, _mesh_test(mesh), rows=splittable)
        # nan everywhere else, and a comparison against it is False, so the
        # blocks that were never asked about answer no on their own
        with _np.errstate(invalid="ignore"):
            return (shares > 0) & (shares < 1)

    def unpredicted(self, variable=None):
        """Which blocks have nothing in them yet.

        What `split` leaves behind: hand it to `predict(..., where=...)` and
        only the blocks the refinement created are visited. Without a variable
        it is the blocks the last split made, which is what a container knows
        without having to be told what was predicted into it; naming one reads
        its missing values instead, which stays true however the object was
        arrived at.
        """
        if variable is None:
            return _np.array(self._fresh, dtype=bool)
        return _np.isnan(_np.asarray(
            self.variables[variable].prediction.values))

    # ------------------------------------------------------------------ #
    # what the model asks for
    # ------------------------------------------------------------------ #
    def get_batched_coordinates(self, index=None):
        if index is None:
            index = _np.arange(self._n_data)

        centres = _np.asarray(self.coordinates[index])
        size = self.block_size[index]
        # block-major, one temporary rather than one per block: block b owns
        # rows b*k to (b+1)*k, which is what `_aggregate` averages back. The
        # only difference from a fixed-size block model is that the sub-grid
        # is scaled by each block's own size.
        coords = centres[:, None, :] + self._sub_grid[None, :, :] \
            * size[:, None, :]
        coords = coords.reshape(-1, centres.shape[1])

        splits = None if self.rows_per_location == 1 else len(centres)
        return coords, splits

    def _batch_rows(self, index=None):
        n, _ = _PointBased._batch_rows(self, index)
        k = self.rows_per_location
        return n * k, (None if k == 1 else n)

    def as_data_frame(self, metadata=True, **kwargs):
        df = super().as_data_frame(metadata=metadata, **kwargs)
        for i, label in enumerate(self.coordinate_labels):
            df["_" + label] = self.block_size[:, i]
        return df

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        One hexahedron per block, written out corner by corner, rather than
        the `ImageData` a regular block model exports: implicit geometry can
        only say one spacing, and the point here is that there is more than
        one. The cells are welded, so blocks that touch share the corners they
        meet at -- which is what lets anything be contoured across them.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        mesh = self._hex_mesh(self._origin, self._size, self.base_step)

        for variable in self.variables.values():
            variable.fill_pyvista_cells(mesh, simulations=simulations)
        for name, column in self.metadata.items():
            column.fill_pyvista_cells(mesh, name)
        return mesh

    def _hex_mesh(self, origin, size, step):
        """One welded hexahedron per block, on any lattice of blocks -- the
        object's own, or the finer one a contour is drawn on. `origin` and
        `size` count that lattice's own cells, `step` says how big one is."""
        low = self.box_corner + origin * step
        size = size * step
        points = (low[:, None, :]
                  + _HEX_CORNERS[None, :, :] * size[:, None, :]).reshape(-1, 3)

        connectivity = _np.arange(len(points)).reshape(-1, 8)
        cells = _np.hstack(
            [_np.full((len(connectivity), 1), 8), connectivity]).ravel()
        kinds = _np.full(len(connectivity), _pv.CellType.HEXAHEDRON,
                         dtype=_np.uint8)
        # Welded, and it matters: written corner by corner every cell owns its
        # own eight, so neighbours share nothing, and anything reading values
        # at a corner sees one block there instead of the several that meet.
        # A contour over an unwelded mesh comes back empty for that reason.
        return _pv.UnstructuredGrid(cells, kinds, points).clean(
            tolerance=1e-9, produce_merge_map=False)

    @staticmethod
    def _shared_corners(origin, size, shape):
        """Each block's eight corners, as indices into the distinct lattice
        points -- the welding of `_hex_mesh`, done in integers."""
        corners = (origin[:, None, :]
                   + _HEX_CORNERS[None, :, :] * size[:, None, :])
        span = shape + 1
        key = ((corners[..., 0] * span[1] + corners[..., 1]) * span[2]
               + corners[..., 2])
        return _np.unique(key.ravel(), return_inverse=True)[1].reshape(-1, 8)

    @staticmethod
    def _at_corners(corners, values):
        """The mean over the blocks meeting at each corner, read back per
        block -- what `cell_data_to_point_data` puts there, and so what decides
        where the surface runs. A block holding no value contributes nothing,
        rather than carrying its absence into every corner it touches.
        """
        total = _np.zeros(int(corners.max()) + 1)
        count = _np.zeros(len(total))
        known = _np.isfinite(values)
        _np.add.at(total, corners[known].ravel(), _np.repeat(values[known], 8))
        _np.add.at(count, corners[known].ravel(), 1.0)
        mean = _np.divide(total, count, out=_np.full(len(total), _np.nan),
                          where=count > 0)
        return mean[corners]

    def _cut_to_contour(self, values, value, margin=1, supersample=1):
        """The blocks a surface runs through, cut small -- in the mesh handed
        to VTK, not in the model.

        A hexahedron is contoured from its own eight corners, so where a
        coarse block meets finer ones it cannot see the corners they place in
        the middle of the shared face. The two sides then draw different
        curves there and the surface tears along every interface it crosses.
        Cutting the surface's neighbourhood to one size puts the interfaces
        out of its way, and one ring of margin covers the surface shifting as
        the values are read at the finer size.

        Cutting `supersample` levels past the model's own finest also smooths
        it. What VTK contours is a trilinear reading of the corner averages,
        continuous but with a crease at every face it crosses, and on a field
        with structure at a few blocks those creases are what looks blocky.
        Averaging onto corners again at each finer level composes into a
        rounder reconstruction, so the surface both turns less sharply and
        sits closer to the level set it is meant to be: on a rough test field,
        one level took the mean angle between neighbouring triangles from 9.1
        to 5.7 degrees and halved the distance from the true surface, matching
        a model of 3.4 times as many blocks that had to be predicted.

        Nothing is predicted, at any level. A child takes its parent's value
        plus what the corners say about the shape running across it, and that
        correction averages to zero over the children, so a block's estimate
        is exactly the mean of the children standing in for it.

        Returns
        -------
        origin, size, step, values
            The finer lattice: origins and sizes counted in its own cells,
            and how big one of those is. `(None, None, None, None)` if nothing
            was cut and the blocks as they stand will do.
        """
        ratio = _np.array(self.discretization, dtype=_np.int64)
        offset = _sub_block_index(self.discretization)
        weights = _trilinear_weights(self.discretization)

        # a lattice `supersample` levels finer than the model's own, so that
        # cutting below the base cell is still whole numbers of a cell
        unit = ratio ** int(supersample)
        origin = self._origin * unit
        size = self._size * unit
        step = self.base_step / unit
        shape = self.lattice_shape * unit

        values = _np.asarray(values, dtype=float)
        cut = False

        for _ in range(self.max_levels + int(supersample)):
            corners = self._shared_corners(origin, size, shape)
            at_corner = self._at_corners(corners, values)
            # a corner with no value must not decide anything either way
            low = _np.where(_np.isnan(at_corner), _np.inf, at_corner).min(1)
            high = _np.where(_np.isnan(at_corner), -_np.inf, at_corner).max(1)
            near = _grow(corners, (low < value) & (high > value), margin)
            near &= _np.all(size >= ratio, axis=1)
            if not _np.any(near):
                break

            cut = True
            child_size = size[near] // ratio
            children = (origin[near][:, None, :]
                        + offset[None, :, :] * child_size[:, None, :]
                        ).reshape(-1, 3)
            shaped = at_corner[near] @ weights.T
            shaped += (values[near] - at_corner[near].mean(axis=1))[:, None]
            # a block beside one that was never predicted has no shape to read
            # off its corners, so it hands its own value down unchanged
            shaped = _np.where(_np.isfinite(shaped), shaped,
                               values[near][:, None])

            origin = _np.concatenate([origin[~near], children])
            size = _np.concatenate(
                [size[~near], _np.repeat(child_size, len(offset), axis=0)])
            values = _np.concatenate([values[~near], shaped.ravel()])

        if not cut:
            return None, None, None, None
        return origin, size, step, values

    def _variable_or_component(self, name):
        """The variable called `name`, or the component of a vector one, and
        the name of whatever owns it -- `None` for a variable of its own.

        A composition is held as a single variable, so its parts are not among
        `self.variables` and asking for `Zn` would otherwise mean reaching into
        `Elements` by hand. Only the parts hold a grade -- the variable itself
        carries nothing to contour -- so naming one has to work. The owner
        comes back with it because an export labels a component by both names.
        """
        if name in self.variables:
            return self.variables[name], None

        for variable in self.variables.values():
            components = getattr(variable, "components", None) or {}
            if name in components:
                return components[name], variable.name

        known = set(self.variables)
        for variable in self.variables.values():
            known.update(
                str(label) for label in
                (getattr(variable, "components", None) or {}))
        raise ValueError(
            "no variable or component named %r; found %s"
            % (name, ", ".join(sorted(known)) or "none"))

    def get_contour(self, variable, value, attribute="prediction",
                    supersample=1):
        """
        Isosurface through blocks of more than one size.

        `marching_cubes` wants a rectangular array and there is none to give
        it, so the cells are handed to VTK instead, which contours an
        unstructured grid directly. On a model of one block size the two agree
        exactly; here the answer is the one a regular grid could not have
        produced without carrying every block at the finest size.

        Values live on the cells and an isosurface needs them on the corners,
        so they are averaged onto the corners first -- the blocks meeting at a
        corner are what decide where the surface passes.

        The blocks the surface runs through are cut to the finest size the
        lattice allows before any of that, in the mesh handed to VTK and not
        in the model: a coarse block cannot see the corners its finer
        neighbours place in the middle of the face they share, so the two
        sides draw different curves there and the surface tears along every
        such interface it crosses. Cutting its neighbourhood to one size puts
        the interfaces out of the way. Nothing is predicted -- a child reads
        its parent's value and the shape its corners carry -- and the surface
        comes back the one a model carried at the finest size throughout would
        have given. See `_cut_to_contour`.

        Parameters
        ----------
        variable : str
            Which variable to contour, or which component of a vector one:
            only the components of a composition hold a grade, so `"Zn"` is
            named rather than the `"Elements"` it belongs to.
        value : float
            The value to draw the surface at.
        attribute : str
            Which of the variable's columns to read; the prediction by
            default.
        supersample : int
            How many levels past the model's own finest block to cut the mesh
            to. Costs `prod(discretization)` times the cells per level, around
            the surface only, and buys a rounder and *closer* surface rather
            than merely a prettier one -- what VTK reads between block corners
            is trilinear, and creasing at every face is what looks blocky. One
            level is worth roughly predicting a model several times the size;
            past that it flattens off. Zero to leave the mesh at the model's
            own resolution.

        Returns
        -------
        surf : Solid3D, Surface3D or Mesh3D
            Whichever the geometry calls for, as `get_contour` on a grid.
        """
        named, owner = self._variable_or_component(variable)

        # a component is labelled by its variable's name as well as its own,
        # so the owner is needed to find it again
        label = "%s - %s" % (_export_label(owner, named.name), attribute)
        mesh = self.as_pyvista()
        if label not in mesh.cell_data:
            parts = getattr(named, "components", None) or {}
            if parts:
                raise ValueError(
                    "%r is made of components and holds no %r of its own; "
                    "name one of %s"
                    % (named.name, attribute,
                       ", ".join(str(label) for label in parts)))
            raise ValueError(
                "%r holds nothing under %r to contour; the mesh carries %s"
                % (named.name, attribute,
                   ", ".join(sorted(mesh.cell_data.keys())) or "nothing"))

        value = float(value)
        values = _np.asarray(mesh.cell_data[label], dtype=float)
        span = (float(_np.nanmin(values)), float(_np.nanmax(values)))

        origin, size, step, finer = self._cut_to_contour(
            values, value, supersample=supersample)
        if origin is not None:
            # let the coarse mesh go before the finer one is built: a block
            # model is large enough that holding both is worth avoiding
            mesh = None
            mesh = self._hex_mesh(origin, size, step)
            mesh.cell_data[label] = finer

        surface = mesh.cell_data_to_point_data().contour(
            [value], scalars=label)
        if surface.n_cells == 0:
            raise ValueError(
                "no surface at %g; %r runs from %g to %g"
                % (value, label, span[0], span[1]))

        surface = surface.triangulate()
        verts = _np.asarray(surface.points, dtype=float)
        faces = _np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
        return mesh3d(verts, faces, _gmt.vertex_normals(verts, faces))


def rotate(data, origin, azimuth=0.0, dip=0.0, rake=0.0, reverse=False):
    mat = _gmt.rotation_matrix(azimuth, dip, rake)
    if reverse:
        mat = mat.T

    origin = _np.array(origin)
    if origin.shape != (1, 3):
        origin = _np.squeeze(origin)[None, :]

    data = _copy.deepcopy(data)
    data.coordinates = _np.matmul(data.coordinates - origin, mat) + origin
    return data


class RotatedGrid3D(Grid3D):
    def __init__(self, start, n, step, azimuth=0.0, dip=0.0, rake=0.0, labels=None):
        self.azimuth = azimuth
        self.dip = dip
        self.rake = rake
        super().__init__(start, n, step=step, labels=labels)

        # Keep the coordinates lazy: rotation is a linear map applied per row
        # by the grid's own coordinate generator.
        mat = self.rotation_matrix()
        origin = _np.asarray(self.origin, dtype=float)
        self._transform = (origin, mat)

        # The rotated grid lies in a parallelepiped whose axis-aligned bounding
        # box is attained at its 8 corners, so it needs no full materialization.
        extremes = [axis[[0, -1]] for axis in self.grid]
        corners = _np.array(list(_iter.product(*extremes[::-1])),
                            dtype=float)[:, ::-1]
        corners = _np.matmul(corners - origin, mat) + origin
        self._bounding_box = BoundingBox.from_array(corners)

    def rotate(self, other):
        return rotate(other, self.origin, self.azimuth, self.dip, self.rake)

    def rotation_matrix(self):
        return _gmt.rotation_matrix(self.azimuth, self.dip, self.rake)

    def index_data(self, data):
        return super().index_data(self.rotate(data))

    def aggregate_categorical(self, data, variable):
        raise NotImplementedError
        # super().aggregate_categorical(self.rotate(data), variable)

    def aggregate_binary(self, data, variable):
        raise NotImplementedError
        # super().aggregate_binary(self.rotate(data), variable)

    def aggregate_numeric(self, data, variable):
        raise NotImplementedError
        # super().aggregate_numeric(self.rotate(data), variable)

    def make_interpolator(self, coordinates):
        raise NotImplementedError

    def as_pyvista(self, simulations=False):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        pv_grid = super().as_pyvista(simulations=simulations)

        mat = _gmt.rotation_matrix(self.azimuth, self.dip, self.rake)
        transf = _np.eye(4)
        transf[:3, :3] = mat.T

        pv_grid = pv_grid.translate(- self.origin)
        pv_grid = pv_grid.transform(transf)
        pv_grid = pv_grid.translate(self.origin)

        return pv_grid

    @classmethod
    def from_bounding_box(cls, box, step, margin=0.1, rounding_decimals=0):
        return NotImplementedError

    @classmethod
    def from_points(cls, points, step, margin=0.1, rounding_decimals=0, labels=None):
        if isinstance(points, _PointBased):
            labels = points.coordinate_labels if labels is None else None
            points = points.coordinates
        if points.shape[1] != 3:
            raise DimensionMismatchError('points must be 3-dimensional')
        rotmat = _gmt.rotation_matrix_from_points(points)
        az, dip, rake = _gmt.angles_from_rotation_matrix(rotmat)

        center = _np.mean(points, axis=0, keepdims=True)

        unrotated_points = _np.matmul(points - center, rotmat.T)
        box, _ = bounding_box(unrotated_points)

        margin = _np.array(margin)
        if margin.shape == ():
            margin = _np.full([2, 3], margin)
        elif margin.shape == (2,):
            margin = _np.stack([margin] * 3, axis=1)

        dif = box[1] - box[0]
        new_min = box[0] - dif * margin[0]
        new_max = box[1] + dif * margin[1]
        n = _np.ceil((new_max - new_min) / step).astype(int) + 1

        origin = _np.squeeze(_np.matmul(new_min, rotmat) + center)
        origin = _np.round(origin, rounding_decimals)

        return cls(start=origin, n=n, step=step, azimuth=az, dip=dip, rake=rake, labels=labels)


# --------------------------------------------------------------------------- #
# Zarr persistence (see _SpatialData.to_zarr / _SpatialData.open)
# --------------------------------------------------------------------------- #
_GEOML_ZARR_FORMAT = 1


def _write_container(group, container):
    """Write a container's auxiliary arrays into ``group`` and return its
    JSON-able reconstruction metadata.

    Grid families are rebuilt from their construction parameters; point-based
    containers store their raw arrays. ``DrillholeData`` is not supported (it is
    raw input data, not a prediction target).
    """
    if isinstance(container, (Grid1D, Grid2D, Grid3D, GridND)):
        meta = {
            "class": type(container).__name__,
            "start": [float(axis[0]) for axis in container.grid],
            "n": [int(x) for x in container.grid_size],
            "step": [float(s) for s in _np.atleast_1d(container.step_size)],
            "labels": [str(lb) for lb in container.coordinate_labels],
        }
        if isinstance(container, (Blocks1D, Blocks2D, Blocks3D)):
            meta["discretization"] = [int(d) for d in container.discretization]
        if isinstance(container, RotatedGrid3D):
            meta["azimuth"] = float(container.azimuth)
            meta["dip"] = float(container.dip)
            meta["rake"] = float(container.rake)
        return meta

    def write(name, array):
        # An ArrayStore streams itself chunk-by-chunk; anything else (a plain
        # ndarray of triangles, directions, ...) is wrapped first.
        if not isinstance(array, _storage.ArrayStore):
            array = _storage.ArrayStore.from_numpy(_np.asarray(array))
        array.write_into(group, name)

    if type(container) is PointData:
        write("_coordinates", container.coordinates)
        return {"class": "PointData",
                "labels": [str(lb) for lb in container.coordinate_labels]}
    if type(container) is GaussianData:
        write("_coordinates", container.coordinates)
        write("_variance", container.variance)
        return {"class": "GaussianData",
                "labels": [str(lb) for lb in container.coordinate_labels]}
    if type(container) is DirectionalData:
        write("_coordinates", container.coordinates)
        write("_directions", container.directions)
        return {"class": "DirectionalData",
                "labels": [str(lb) for lb in container.coordinate_labels],
                "direction_labels":
                    [str(lb) for lb in container.direction_labels]}
    if type(container) is Section3D:
        # The construction parameters (center, azimuth, ...) are not kept on
        # the instance, so the rotated coordinates are stored directly.
        write("_coordinates", container.coordinates)
        return {"class": "Section3D",
                "labels": [str(lb) for lb in container.coordinate_labels],
                "grid_shape": [int(x) for x in container.grid_shape]}
    if type(container) is BlockSet3D:
        # the lattice itself, in integers, plus what it is counted in; the
        # centres are derived from those and are not stored twice
        write("_origin", container._origin)
        write("_level", container._level)
        return {"class": "BlockSet3D",
                "labels": [str(lb) for lb in container.coordinate_labels],
                "box_corner": [float(x) for x in container.box_corner],
                "base_step": [float(x) for x in container.base_step],
                "lattice_shape": [int(x) for x in container.lattice_shape],
                "discretization":
                    [int(d) for d in container.discretization],
                "max_levels": int(container.max_levels)}
    if isinstance(container, Mesh3D):
        write("_coordinates", container.coordinates)
        write("_triangles", container.triangles)
        write("_normals", container.normals)
        # the class is recorded rather than inferred on the way back: a mesh
        # that closes could be rebuilt as either a Mesh3D or a Solid3D
        return {"class": type(container).__name__}
    raise NotImplementedError(
        f"to_zarr does not yet support container type "
        f"'{type(container).__name__}'")


def _write_metadata(group, container):
    """Write the container's point-wise metadata columns into ``group``.

    Text columns are stored as the integer codes they already are, with their
    labels going into the JSON description.
    """
    meta = {}
    for name, column in container.metadata.items():
        key = "_metadata/" + name
        column.values.write_into(group, key)
        meta[name] = {"key": key, "labels": column.labels}
    return meta


def _rebuild_metadata(container, group, meta):
    """Reattach the metadata columns, left on disk as the variables are."""
    for name, info in meta.items():
        container.metadata[name] = _Attribute.from_store(
            container, _storage.ArrayStore.wrap_zarr(group[info["key"]]),
            info["labels"])


def _rebuild_container(meta, group):
    cls_name = meta["class"]

    def read(name):
        return _np.asarray(_storage.ArrayStore.wrap_zarr(group[name]))

    def read_store(name):
        # Coordinates and variance stay on disk (read/write), as variables do.
        return _storage.ArrayStore.wrap_zarr(group[name])

    if cls_name == "PointData":
        return PointData.from_array(read_store("_coordinates"), meta["labels"])
    if cls_name == "GaussianData":
        return GaussianData.from_array(
            read_store("_coordinates"), read_store("_variance"),
            meta["labels"])
    if cls_name == "DirectionalData":
        labels = meta["labels"]
        dir_labels = meta["direction_labels"]
        df = _pd.concat([
            _pd.DataFrame(read("_coordinates"), columns=labels),
            _pd.DataFrame(read("_directions"), columns=dir_labels),
        ], axis=1)
        return DirectionalData(df, labels, dir_labels)
    if cls_name == "Section3D":
        # Bypass Section3D.__init__ (which needs the discarded construction
        # parameters) and initialize the PointData layer directly.
        section = Section3D.__new__(Section3D)
        _PointBased.__init__(section)
        section._init_coordinates(read_store("_coordinates"), meta["labels"])
        section.grid_shape = [int(x) for x in meta["grid_shape"]]
        return section
    if cls_name == "BlockSet3D":
        # Bypass __init__, which builds a full coarse grid: the lattice that
        # was saved is whatever it had been refined to since.
        blocks = BlockSet3D.__new__(BlockSet3D)
        _PointBased.__init__(blocks)
        blocks.max_levels = int(meta["max_levels"])
        blocks.discretization = [int(d) for d in meta["discretization"]]
        blocks._sub_grid = _unit_sub_grid(blocks.discretization)
        blocks.base_step = _np.asarray(meta["base_step"], dtype=float)
        blocks.box_corner = _np.asarray(meta["box_corner"], dtype=float)
        blocks.lattice_shape = _np.asarray(meta["lattice_shape"],
                                           dtype=_np.int64)
        blocks._origin = read("_origin").astype(_np.int64)
        blocks._level = read("_level").astype(_np.int64)
        # a saved model is one that was predicted into, not one mid-refinement
        blocks._fresh = _np.zeros(len(blocks._origin), dtype=bool)
        blocks._init_coordinates(blocks._centres(), meta["labels"])
        blocks._bounding_box = BoundingBox(
            blocks.box_corner,
            blocks.box_corner + blocks.lattice_shape * blocks.base_step)
        return blocks

    meshes = {"Mesh3D": Mesh3D, "Surface3D": Surface3D, "Solid3D": Solid3D,
              "DTM3D": DTM3D}
    if cls_name in meshes:
        return meshes[cls_name](read("_coordinates"), read("_triangles"),
                                read("_normals"))

    classes = {"Grid1D": Grid1D, "Grid2D": Grid2D, "Grid3D": Grid3D,
               "GridND": GridND, "Blocks1D": Blocks1D, "Blocks2D": Blocks2D,
               "Blocks3D": Blocks3D, "RotatedGrid3D": RotatedGrid3D}
    if cls_name not in classes:
        raise NotImplementedError(
            f"open does not support container type '{cls_name}'")

    start, n, step = meta["start"], meta["n"], meta["step"]
    if cls_name in ("Grid1D", "Blocks1D"):
        # 1D grids take scalar start/n/step.
        start, n, step = start[0], n[0], step[0]
    kwargs = {"start": start, "n": n, "step": step, "labels": meta["labels"]}
    if "discretization" in meta:
        kwargs["discretization"] = meta["discretization"]
    if cls_name == "RotatedGrid3D":
        kwargs.update(azimuth=meta["azimuth"], dip=meta["dip"], rake=meta["rake"])
    return classes[cls_name](**kwargs)


def _supported_top_variables():
    """Top-level variable classes that ``to_zarr``/``open`` can round-trip.

    The internal ``_Category``/``_Component`` are only persisted recursively as
    components, never at the top level.
    """
    return (ContinuousVariable, VectorVariable, CompositionalVariable,
            RockTypeVariable, CategoricalVariable, OrderedRockType,
            BinaryVariable, AnomalyVariable)


def _write_variable(group, variable):
    """Write a variable into ``group`` and return its reconstruction metadata."""
    if type(variable) not in _supported_top_variables():
        raise NotImplementedError(
            f"to_zarr does not yet support variable type "
            f"'{type(variable).__name__}'")
    return variable._zarr_save(group, variable.name)


def _rebuild_variable(container, group, vmeta):
    cls_name, name = vmeta["class"], vmeta["name"]
    labels = vmeta.get("labels")
    if cls_name == "ContinuousVariable":
        container.add_continuous_variable(name)
    elif cls_name == "VectorVariable":
        container.add_vector_variable(name, labels=labels)
    elif cls_name == "CompositionalVariable":
        container.add_compositional_variable(name, labels=labels)
    elif cls_name == "CategoricalVariable":
        container.add_categorical_variable(name, labels=labels)
    elif cls_name == "RockTypeVariable":
        container.add_rock_type_variable(name, labels=labels)
    elif cls_name == "OrderedRockType":
        container.add_rock_type_variable(name, labels=labels, ordered=True)
    elif cls_name == "BinaryVariable":
        container.add_binary_variable(name, labels=labels)
    elif cls_name == "AnomalyVariable":
        container.add_anomaly_variable(name, label=labels[0])
    else:
        raise NotImplementedError(
            f"open does not support variable type '{cls_name}'")

    container.variables[name]._zarr_load(group, name, vmeta)

