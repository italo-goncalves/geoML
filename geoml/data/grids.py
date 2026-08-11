# geoML - machine learning models for geospatial data
# Copyright (C) 2021  Ítalo Gomes Gonçalves
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
"""
The regular grids: `_GriddedData`, `Grid1D/2D/3D`, `GridND` and
`RotatedGrid3D`, plus `aggregate` (one implementation for every kind of
variable) and the `from_data` box-fitting shared with the block classes.
"""
import copy as _copy
import itertools as _iter

import numpy as _np
import pandas as _pd
import pyvista as _pv

import geoml.math.geometry as _gmt
import geoml.math.interpolate as _gint
from geoml.math.geometry import bounding_box

from geoml.data.base import *
from geoml.data.variables import *
from geoml.data.containers import *
from geoml.data.containers import _PointBased, _SpatialData

def _cover_box(data, step, margin, decimals, n_dim=None, cells=False):
    """`(start, n, step, labels)` for a lattice covering `data`'s box.

    The box is widened by `margin` (a fraction of its extent, per side or per
    axis if given as an array), and the lower corner is then *floored* to
    `decimals` rather than rounded -- a round start reads better on a section,
    and flooring means the cover never shrinks below what the margin asked
    for. The count grows to keep the top covered, so the last node or cell
    always reaches past the margined maximum.

    `cells=True` counts cells instead of nodes: `start` is then the box
    corner, and the caller places the first centre half a step in.
    """
    box = data.bounding_box
    if n_dim is not None and box.n_dim != n_dim:
        raise DimensionMismatchError(
            "%s covers %d-dimensional data; this has %d"
            % (data.__class__.__name__, n_dim, box.n_dim))
    n_dim = box.n_dim

    step = _np.broadcast_to(
        _np.asarray(step, dtype=float).ravel(), (n_dim,))
    if _np.any(step <= 0):
        raise ValueError("step must be positive; got %r" % (step,))

    margin = _np.asarray(margin, dtype=float)
    if margin.shape == ():
        margin = _np.full([2, n_dim], float(margin))
    elif margin.shape == (2,):
        margin = _np.stack([margin] * n_dim, axis=1)
    elif margin.shape == (n_dim,):
        margin = _np.stack([margin] * 2, axis=0)
    elif margin.shape != (2, n_dim):
        raise ValueError(
            "margin is a fraction of the box's extent: one number, one per "
            "side (2,), one per axis (%d,), or one per side and axis "
            "(2, %d); got shape %s" % (n_dim, n_dim, margin.shape))

    low, high = box.min[0].astype(float), box.max[0].astype(float)
    extent = high - low
    low = low - extent * margin[0]
    high = high + extent * margin[1]

    factor = 10.0 ** decimals
    start = _np.floor(low * factor) / factor
    count = _np.ceil((high - start) / step - _TOL_COVER).astype(int)
    n = _np.maximum(count, 1) if cells else count + 1

    return start, n, _np.array(step), getattr(data, "coordinate_labels", None)


# floating arithmetic at a box edge must not buy a whole extra row of cells
_TOL_COVER = 1e-9


def _cell_means(n_cells, cell, values):
    """The mean of `values` over each cell; NaN where nothing fell."""
    frame = _pd.DataFrame({
        "cell": cell, "value": _np.asarray(values, dtype=float)})
    frame = frame[frame["cell"] >= 0].dropna(subset=["value"])
    out = _np.full(n_cells, _np.nan)
    grouped = frame.groupby("cell")["value"].mean()
    out[grouped.index.to_numpy()] = grouped.to_numpy()
    return out


def _dominant_labels(n_cells, cell, values):
    """The label most often seen in each cell; blank where two tie.

    A tie has no dominant label -- picking by sort order, which is what the
    old per-class aggregators did, reports an answer where there is none --
    so an ambiguous cell comes back empty instead.
    """
    frame = _pd.DataFrame({"cell": cell, "value": values})
    frame = frame[(frame["cell"] >= 0) & frame["value"].notna()
                  & (frame["value"] != "")]
    out = _np.full(n_cells, "", dtype=object)
    if len(frame) == 0:
        return out
    counts = frame.groupby(
        ["cell", "value"]).size().reset_index(name="_n")
    best = counts.groupby("cell")["_n"].transform("max")
    winners = counts[counts["_n"] == best].groupby("cell")["value"].agg(
        ["first", "size"])
    settled = winners[winners["size"] == 1]
    out[settled.index.to_numpy()] = settled["first"].to_numpy()
    return out


def _aggregate_onto(target, data, variables=None, metadata=True):
    """The shared body of `aggregate`: one operation per variable kind.

    `target` says which of its cells holds each sample (`_cell_of`); after
    that nothing depends on what the target is -- a grid, a rotated grid or a
    block model of several sizes all aggregate the same way, which is what
    replaced three near-copies of this logic per class.
    """
    cell = target._cell_of(data)
    n_cells = target.n_data

    if variables is None:
        chosen = list(data.variables)
    else:
        chosen = [str(name) for name in _np.atleast_1d(variables)]
    for name in chosen:
        if name not in data.variables:
            raise ValueError(
                "no variable named %r to aggregate; found %s"
                % (name, ", ".join(sorted(data.variables)) or "none"))
        variable = data.variables[name]

        if isinstance(variable, CompositionalVariable):
            # averaged part by part, then closed again: means of parts do not
            # sum to one on their own, and a composition that does not close
            # is not a composition
            parts = _np.stack(
                [_cell_means(n_cells, cell,
                             part.measurements.values.to_numpy())
                 for part in variable.components.values()], axis=1)
            total = parts.sum(axis=1, keepdims=True)
            with _np.errstate(invalid="ignore"):
                parts = _np.where(total > 0, parts / total, _np.nan)
            target.variables[name] = CompositionalVariable(
                name, target, list(variable.labels), parts)
        elif isinstance(variable, VectorVariable):
            parts = _np.stack(
                [_cell_means(n_cells, cell,
                             part.measurements.values.to_numpy())
                 for part in variable.components.values()], axis=1)
            target.variables[name] = VectorVariable(
                name, target, list(variable.labels), parts)
        elif isinstance(variable, RockTypeVariable):
            # both sides of a contact vote, as they always have: a boundary
            # measurement names the category either side of it and neither
            # side is the measurement
            votes = _dominant_labels(
                n_cells, _np.tile(cell, 2),
                _np.concatenate([variable.measurements_a.to_numpy(),
                                 variable.measurements_b.to_numpy()]))
            target.variables[name] = CategoricalVariable(
                name, target, list(variable.labels), votes)
        elif isinstance(variable, BinaryVariable):
            target.variables[name] = BinaryVariable(
                name, target, list(variable.labels),
                _dominant_labels(n_cells, cell,
                                 variable.measurements.to_numpy()))
        elif isinstance(variable, ContinuousVariable):
            target.variables[name] = ContinuousVariable(
                name, target,
                _cell_means(n_cells, cell,
                            variable.measurements.values.to_numpy()))
        else:
            raise TypeError(
                "%r is a %s, which aggregate does not know how to carry"
                % (name, type(variable).__name__))

    if metadata:
        for name, column in data.metadata.items():
            if column.labels is None:
                target.add_metadata(
                    name, _cell_means(n_cells, cell,
                                      column.values.to_numpy()))
            else:
                target.add_metadata(
                    name, _dominant_labels(n_cells, cell, column.to_numpy()))


def _fitted_rotation(data, decimals):
    """Azimuth, dip and rake fitted to `data`'s points, rounded to `decimals`.

    The angles are rounded *before* anything is built from them -- a grid at
    47.3182 degrees is nobody's intention -- so the box is measured in the
    rounded frame and the data stays covered.
    """
    coordinates = getattr(data, "coordinates", None)
    if coordinates is not None:
        points = _np.asarray(coordinates, dtype=float)
    elif hasattr(data, "_desurveyed_points"):
        # drillholes are interval data and expose no `coordinates`; the
        # cloud their bounding box is measured from serves the fit
        points = data._desurveyed_points()
    else:
        raise TypeError(
            "%s carries no coordinates to fit a rotation to"
            % type(data).__name__)
    if points.shape[1] != 3:
        raise DimensionMismatchError(
            "a rotation is fitted to 3-dimensional points; these have %d"
            % points.shape[1])

    fitted = _gmt.rotation_matrix_from_points(points)
    azimuth, dip, rake = _gmt.angles_from_rotation_matrix(fitted)
    azimuth, dip, rake = (float(_np.round(angle, decimals))
                          for angle in (azimuth, dip, rake))
    return points, azimuth, dip, rake


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

    # dimension a subclass covers; None reads it off the data (GridND)
    _GRID_NDIM = None

    @classmethod
    def from_data(cls, data, step, margin=0.1, decimals=0):
        """
        A grid covering another object's bounding box.

        Parameters
        ----------
        data
            Any spatial object, drillholes included -- whatever has a
            bounding box.
        step
            The step size, one number or one per direction.
        margin : float or array
            How far past the data's box to reach, as a fraction of its
            extent: one number, one per side ``(low, high)``, one per axis,
            or one per side and axis with shape ``(2, n_dim)``.
        decimals : int
            The box corner is floored to this many decimals -- round numbers
            read better on a section -- and the number of steps grows to keep
            the far side covered, so the margin is never eaten by the
            rounding.
        """
        start, n, step, labels = _cover_box(
            data, step, margin, decimals, n_dim=cls._GRID_NDIM)
        if len(start) == 1:
            return cls(start=float(start[0]), n=int(n[0]),
                       step=float(step[0]),
                       labels=labels[0] if labels else None)
        return cls(start=start, n=n, step=step, labels=labels)

    def _cell_of(self, data):
        """Which cell each of `data`'s locations falls in, as this object's
        own row index; `-1` outside the grid."""
        ids = self.index_data(data)
        shape = _np.asarray(self.grid_size)
        inside = _np.all((ids >= 0) & (ids < shape), axis=1)
        flat = _np.full(len(ids), -1, dtype=_np.int64)
        # the first axis varies fastest in `_generate`, which is Fortran order
        flat[inside] = _np.ravel_multi_index(
            ids[inside].T, shape, order="F")
        return flat

    def aggregate(self, data, variables=None, metadata=True):
        """Carries another object's measurements onto this object's cells.

        One method instead of one per kind: each variable says what it is and
        the operation follows. Continuous values (and each part of a vector)
        average; categories keep the label most often measured in the cell,
        both sides of a contact voting; a composition is averaged and closed
        again; numeric metadata averages and coded metadata keeps the
        dominant label. What is truly ambiguous comes back empty -- two
        labels tied for a cell name no winner, and a cell nothing fell in
        holds NaN or blank.

        Parameters
        ----------
        data
            A point-based object whose variables are measured.
        variables : str or list, optional
            Which of `data`'s variables to carry; all of them by default.
        metadata : bool
            Whether to carry the metadata columns as well.

        Returns
        -------
        self, so that calls can be chained.
        """
        _aggregate_onto(self, data, variables, metadata)
        return self

    def as_data_frame(self, metadata=True, **kwargs):
        df = _PointBased.as_data_frame(self, metadata=metadata, **kwargs)
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

        new_obj = PointData.from_array(coords[keep], self.coordinate_labels)
        self._subset_metadata(new_obj, keep)
        for name, var in self.variables.items():
            new_obj.variables[name] = var[keep]
            new_obj.variables[name].set_coordinates(new_obj)
        return new_obj


class Grid1D(_GriddedData):
    _GRID_NDIM = 1

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
    _GRID_NDIM = 2

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
    _GRID_NDIM = 3

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

    def make_interpolator(self, coordinates):
        return _gint.cubic_conv_3d(coordinates,
                                   self.grid[0], self.grid[1], self.grid[2])

    def as_pyvista(self, simulations=False, include="**"):
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
        return self._finish_pyvista(pv_grid, "cube", simulations, include)

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
        # imported late: the mesh adapters live beside the mesh classes
        from geoml.data.meshes import (_sheet_interpolator, _check_covered,
                                     _side_codes)
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
        # into the lattice frame, so the *inverse* of the map the grid's own
        # coordinates leave by. It used to apply the forward map -- probed:
        # only the odd node of the grid landed in its own cell -- and nothing
        # noticed, because everything downstream of index_data raised
        # NotImplementedError until the aggregates were unified.
        return super().index_data(
            rotate(data, self.origin, self.azimuth, self.dip, self.rake,
                   reverse=True))

    def make_interpolator(self, coordinates):
        raise NotImplementedError

    def as_pyvista(self, simulations=False, include="**"):
        """
        Converts this object to a pyvista one, carrying its variables.

        Parameters
        ----------
        simulations
            Which simulations to include: `False` for none (the default, since
            each one is a full-length array in the exported object), `True` for
            all of them, an `int` for the first n, or a sequence of indices.
        """
        pv_grid = super().as_pyvista(simulations=simulations, include=include)

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
    def from_data(cls, data, step, margin=0.1, decimals=0):
        """
        A rotated grid fitted to another object's spread.

        The rotation is fitted to the data's own points (a drillhole's
        desurveyed cloud serves where there are no point coordinates), and
        the angles are rounded to `decimals` *before* anything is built from
        them -- a grid at 47.3182 degrees is nobody's intention -- so the box
        is measured in the rounded frame and the data stays covered. The
        world origin is rounded to the same `decimals`.

        Parameters
        ----------
        data
            Any spatial object with 3-dimensional coordinates, drillholes
            included.
        step
            The step size, one number or one per direction.
        margin : float or array
            A fraction of the unrotated box's extent; see `Grid3D.from_data`.
        decimals : int
            Decimals for the origin *and* for the azimuth, dip and rake, in
            degrees.
        """
        points, azimuth, dip, rake = _fitted_rotation(data, decimals)
        mat = _gmt.rotation_matrix(azimuth, dip, rake)

        centre = _np.mean(points, axis=0, keepdims=True)
        unrotated = _np.matmul(points - centre, mat.T)

        margin = _np.asarray(margin, dtype=float)
        if margin.shape == ():
            margin = _np.full([2, 3], float(margin))
        elif margin.shape == (2,):
            margin = _np.stack([margin] * 3, axis=1)

        low = unrotated.min(axis=0)
        high = unrotated.max(axis=0)
        extent = high - low
        low = low - extent * margin[0]
        high = high + extent * margin[1]

        step = _np.broadcast_to(
            _np.asarray(step, dtype=float).ravel(), (3,))
        n = _np.ceil((high - low) / step - _TOL_COVER).astype(int) + 1

        origin = _np.squeeze(_np.matmul(low[None, :], mat) + centre)
        origin = _np.round(origin, decimals)

        labels = getattr(data, "coordinate_labels", None)
        return cls(start=origin, n=n, step=_np.array(step), azimuth=azimuth,
                   dip=dip, rake=rake, labels=labels)

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


