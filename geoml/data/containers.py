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
The point-based containers: `_SpatialData` (what every container is),
`_PointBased`, `PointData`, `GaussianData`, `DirectionalData` and
`Section3D`, with the batching contract the models read.
"""
import copy as _copy
import inspect as _inspect
import json as _json
import warnings as _warnings
from collections.abc import Sequence
from typing import Any as _Any

import numpy as _np
import pandas as _pd
import pyvista as _pv
import tensorflow as _tf
import dask.array as _da
import zarr as _zarr
import scipy.cluster.hierarchy as _hierarchy
import scipy.spatial as _spatial
import scipy.stats as _sstats
from sklearn.cluster import KMeans as _KMeans

import geoml.math.geometry as _gmt
import geoml.math.tf as _tftools
import geoml._types as _types
import geoml.storage as _storage
import geoml.viz.plotly as _py
from geoml.math.geometry import bounding_box

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geoml.data.geoh5 import Workspace as _GeoH5Workspace
    import geoml.transform as _tr

from geoml.data.base import *
from geoml.data.base import _Attribute, _TreeNode, _frame_from_columns
from geoml.data.variables import *
from geoml.data.variables import _Variable, _divisor, _units_per_label


def _balanced_assignment(cluster, sizes, k):
    """Deals clusters to `k` folds, largest first, each to the emptiest.

    `cluster` labels each atom, `sizes` counts each atom's points; the answer
    is a fold per atom. With at least `k` clusters every fold gets one.
    """
    cluster_size = _np.bincount(cluster, weights=sizes)
    fold_of_cluster = _np.zeros(cluster_size.size, dtype=int)
    load = _np.zeros(k)
    for c in _np.argsort(cluster_size)[::-1]:
        f = int(_np.argmin(load))
        fold_of_cluster[c] = f
        load[f] += cluster_size[c]
    return fold_of_cluster[cluster]


def _prepare_composition(values, labels, units, rest, name):
    """The parts of a composition, ready to be stored.

    Handles what a raw assay needs before it can be modelled, in the order
    the steps have to happen -- and returns the parts **in their own
    units**, the closure having been worked out in fractions, since only
    there does a row sum to one:

    1. Every column is divided by its unit, so that the parts become
       fractions of the same whole and can be added up. This is why the
       unit of each part has to be declared.
    2. A row missing any one of its parts is marked missing entirely. The
       parts of a composition only carry information relative to each
       other, so a row that is short of one of them cannot be used.
    3. Non-positive parts are replaced by half the smallest positive value
       of their own column, the usual substitution for values below
       detection. A log-ratio transform cannot take a zero.
    4. With `rest`, a further part is added holding whatever is left of the
       whole. Where the parts leave no room -- they already account for
       everything, or for more than everything -- the rest is held at the
       smallest positive part found anywhere and those samples are scaled
       down to fit. Without it the rows are closed instead.

    A row nothing above touches keeps the numbers it came with, to the last
    bit: it is written back rather than divided and multiplied by its own
    unit, so a table that arrives clean is stored exactly as it was read.
    """
    values = _np.asarray(values, dtype=float)
    given = labels[:-1] if rest else labels
    if values.ndim != 2 or values.shape[1] != len(given):
        raise ValueError(
            "%r has %d part(s) to fill%s but the measurements have shape %s"
            % (name, len(given), " (besides the rest)" if rest else "",
               (values.shape,)))

    divisors = _np.array([_divisor(units[label]) for label in given])
    parts = values / divisors[None, :]
    changed = _np.zeros(values.shape[0], dtype=bool)

    missing = _np.any(_np.isnan(parts), axis=1)
    if missing.any():
        partly = missing.sum() - int(_np.all(_np.isnan(parts), axis=1).sum())
        if partly > 0:
            _warnings.warn(
                f"{partly} sample(s) are missing some but not all parts of "
                f"the composition; they were marked missing entirely")
        parts[missing] = _np.nan
        changed |= missing

    for i, label in enumerate(given):
        with _np.errstate(invalid="ignore"):
            replace = parts[:, i] <= 0
        if not replace.any():
            continue
        positive = parts[parts[:, i] > 0, i]
        if positive.size == 0:
            _warnings.warn(
                f"every value of {label} is non-positive, so there is no "
                f"scale to replace them with; they were left alone")
            continue
        parts[replace, i] = 0.5 * positive.min()
        changed |= replace

    def written(rows):
        """The parts in their own units, the untouched rows kept verbatim."""
        out = values.copy()
        out[rows] = parts[rows] * divisors[None, :]
        return out

    if not rest:
        with _np.errstate(invalid="ignore"):
            total = parts.sum(axis=1, keepdims=True)
            # a row already closed is left alone rather than divided by a
            # number that is one to within rounding
            open_row = ~(_np.abs(total[:, 0] - 1.0) < 1e-10)
            parts = _np.where(open_row[:, None], parts / total, parts)
        return written(changed | open_row)

    with _np.errstate(invalid="ignore"):
        positive = parts[parts > 0]
    if positive.size == 0:
        raise ValueError(
            "the composition has no positive value anywhere, so there is no "
            "room to place a rest in")
    minimum = positive.min()

    total = parts.sum(axis=1)
    residual = 1.0 - total
    with _np.errstate(invalid="ignore"):
        crowded = residual < minimum
    if crowded.any():
        _warnings.warn(
            f"{crowded.sum()} sample(s) leave no room for a rest, their parts "
            f"already accounting for the whole; they were scaled down so that "
            f"the rest could take the smallest part found, {minimum:.3g}. "
            f"Check the units if this affects many samples")
        parts[crowded] *= ((1.0 - minimum) / total[crowded])[:, None]
        residual = _np.where(crowded, minimum, residual)
        changed |= crowded

    rest_divisor = _divisor(units[labels[-1]])
    return _np.concatenate(
        [written(changed), residual[:, None] * rest_divisor], axis=1)


class _SpatialData(_TreeNode):
    """Abstract class for spatial data in general"""

    # Declared for the reader and the checker: `variables` and `metadata`
    # are what a container holds, and `coordinates` is what every concrete
    # one carries under its own arrangement -- a point cloud stores them,
    # a grid generates them, a mesh keeps its vertices.
    variables: "dict[str, _Variable]"
    metadata: "dict[str, _Attribute]"
    coordinates: _Any

    _n_dim: int
    _n_data: int
    _bounding_box: _Any

    def __init__(self):
        # zero rather than None: an empty container has no locations and
        # no dimensions, which is a count, not the absence of one, and
        # every subclass overwrites both before anything reads them
        self._n_dim = 0
        self._bounding_box = None
        self._n_data = 0
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

    def _tree_root_label(self):
        return "%s - %s locations" % (type(self).__name__, self.n_data)

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

    def _finish_pyvista(self, target, kind, simulations=False, include="**"):
        """Fill a freshly built pyvista object and hand it back.

        Every variable through the one enumeration, the metadata columns
        under their bare names (they were only ever exported by the block
        set before -- an air code or a fold is as useful draped over a grid),
        and `field_data["geoml_paths"]`: a JSON table from each array's label
        back to the path that produced it. The label alone cannot be parsed
        back -- `flat` and `pretty` are not invertible -- so anything reading
        an export and wanting to ask geoML about a column reads the table
        instead of guessing. Recorded by the writer rather than enumerated a
        second time, which would read every selected realization twice.
        """
        table = {}
        writer = "fill_pyvista_" + kind
        for variable in self.variables.values():
            for path, attribute in variable._export_leaves(
                    include, simulations):
                label = render(path, "pretty")
                getattr(attribute, writer)(target, label)
                table[label] = str(path)
        for name, column in self.metadata.items():
            getattr(column, writer)(target, name)
            table[name] = METADATA_ROOT + PATH_SEP + name

        if table:
            target.field_data["geoml_paths"] = _np.array(
                [_json.dumps(table)])
        units = self.units()
        if units:
            # what each column is measured in, keyed by the same path the
            # table above maps back to: a viewer reading a grade off an
            # export has no other way to know whether it is a percentage
            target.field_data["geoml_units"] = _np.array(
                [_json.dumps({str(path): unit
                              for path, unit in units.items()})])
        return target

    def units(self) -> "dict[str, _types.Unit]":
        """What each variable is measured in, by path.

        Only the ones that declare a unit; a variable's parts are listed
        under their own paths, since a composition's parts each carry their
        own. The keys are the paths `get` takes.
        """
        found = {}
        for variable in self.variables.values():
            for path, node in variable.walk():
                unit = getattr(node, "unit", None)
                if unit is not None:
                    found[str(path)] = unit
        return found

    @property
    def rows_per_location(self):
        """Rows the model evaluates for each location of this object.

        One, except where a location fans out into several — a block with
        discretization. Prediction divides the batch size by this, so that
        `prediction_batch_size` counts the rows actually handed to the model
        rather than meaning something different for every container.
        """
        return 1

    def add_metadata(self, name: str, values: _types.ArrayLike,
                     labels: _types.Labels | None = None) -> None:
        """
        Attaches point-wise information to this object's locations.

        Metadata is anything known per location that the models do not use —
        an air/solid code, a cross-validation fold, a sample weight. It follows
        the object through subsetting, `as_data_frame()` and `to_zarr()`, but
        is never modeled: use a variable for that.

        Parameters
        ----------
        name
            Column name. An existing column with this name is replaced.
        values
            One value per data location. Text is stored as integer codes.
        labels
            The categories `values` indexes into, when the codes are given
            directly rather than the text they stand for.
        """
        values = _np.asarray(values)
        if labels is None and values.dtype.kind not in "OUS":
            self.metadata[name] = _Attribute(self, values, dtype=values.dtype)
        else:
            self.metadata[name] = _Attribute.encoded(self, values, labels)

    def get_metadata(self, name: str) -> _np.ndarray:
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

    def derive(self, names, function, arguments, units=None):
        """
        Computes new variables from existing ones, realization by realization.

        The function is applied once per realization: for realization `j` it
        receives realization `j` of every simulated argument (metadata
        arguments come in whole, being constant), and its outputs become
        realization `j` of the new variables. Summarizing *after* applying
        the function is what keeps a nonlinear one honest -- the prediction
        column is the mean of the derived realizations, not the function of
        the parents' predictions. If the function accepts a keyword named
        ``simulation``, it also receives the realization's index, which is
        how anything drawn per realization -- a price scenario, say -- knows
        which draw to use.

        The stores are read and written a band of locations at a time, so a
        block model's simulations are never held whole. Deriving a name that
        already exists replaces it: the recipe lives in the calling script,
        and running it again is the refresh.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the variable(s) to create. The function must return
            one array per name (a tuple where there are several).
        function
            Called as ``function(*columns)``, each column an array over the
            locations; must return array(s) of the same length.
        arguments : list of str
            Paths of the inputs, resolved by `get`. Every variable named must
            carry simulations, all with the same count -- `derive` is
            realization-wise, and a variable without realizations has no
            place in it. A per-location constant comes in as metadata:
            ``"_metadata/density"``.
        units : str, float, mapping or sequence
            What the derived values are measured in -- one per name, or a
            mapping naming some of them. A label, as on any variable a
            model reads directly.

        Returns
        -------
        DerivedVariable, or a list of them matching `names`.
        """
        single = isinstance(names, str)
        names = [names] if single else list(names)
        if units is not None and not isinstance(units, (dict, list, tuple)):
            units = [units]
        per_name = _units_per_label(units, names)

        columns = []
        n_sim = None
        for path in arguments:
            parts = VariablePath(path).parts
            if parts[:1] == (METADATA_ROOT,):
                columns.append(
                    ("constant", _np.asarray(self.get_metadata(parts[1]))))
                continue
            node = self.get(path)
            store = getattr(node, "simulations", None)
            if store is None:
                raise ValueError(
                    "%r has no simulations; `derive` is realization-wise, "
                    "so every variable argument needs them (a per-location "
                    "constant can come in as '_metadata/<name>')" % str(path))
            if n_sim is None:
                reference, n_sim = store, int(store.shape[1])
            elif int(store.shape[1]) != n_sim:
                raise ValueError(
                    "%r carries %d realizations where %r carries %d; "
                    "`derive` walks them in step"
                    % (str(path), int(store.shape[1]),
                       str(arguments[0]), n_sim))
            columns.append(("simulated", store))
        if n_sim is None:
            raise ValueError(
                "every argument is metadata; at least one simulated "
                "variable is needed to walk")

        wants_index = "simulation" in _inspect.signature(function).parameters

        new_vars = []
        for name in names:
            variable = DerivedVariable(
                name, self, parents=[str(p) for p in arguments],
                unit=per_name[name])
            variable.allocate_simulations(n_sim)
            self.variables[name] = variable
            new_vars.append(variable)

        for band in reference.row_bands():
            loaded = [data[band] if kind == "constant"
                      else _np.asarray(data[band])
                      for kind, data in columns]
            n_rows = int(band.stop - band.start)
            out = [_np.empty((n_rows, n_sim)) for _ in names]
            for j in range(n_sim):
                args = [c[:, j] if kind == "simulated" else c
                        for (kind, _), c in zip(columns, loaded)]
                kwargs = {"simulation": j} if wants_index else {}
                result = function(*args, **kwargs)
                if not isinstance(result, (tuple, list)):
                    result = (result,)
                if len(result) != len(names):
                    raise ValueError(
                        "the function returned %d output(s) for %d name(s)"
                        % (len(result), len(names)))
                for target, values in zip(out, result):
                    values = _np.asarray(values, dtype=float).reshape(-1)
                    if len(values) != n_rows:
                        raise ValueError(
                            "the function returned %d values for a band of "
                            "%d locations" % (len(values), n_rows))
                    target[:, j] = values
            for variable, block in zip(new_vars, out):
                variable.simulations[band, :] = block
                variable.prediction.values[band] = block.mean(axis=1)

        return new_vars[0] if single else new_vars

    def drop(self, names: "str | Sequence[str]") -> "_SpatialData":
        """
        Removes variables from this container.

        Whole variables only: a component belongs to the variable that built
        it -- a composition without one part is a different composition, and
        a categorical without one class a different classification -- so a
        component's name is refused with its owner named, rather than half a
        variable being left behind.

        Parameters
        ----------
        names : str or list
            The variable(s) to remove.

        Returns
        -------
        self, so that calls can be chained.
        """
        for name in ([names] if isinstance(names, str) else list(names)):
            name = str(name)
            if name in self.variables:
                del self.variables[name]
                continue
            if PATH_SEP in name:
                raise ValueError(
                    "%r is a path; only whole variables can be dropped -- "
                    "one of %s" % (name, ", ".join(sorted(self.variables))
                                   or "none"))
            try:
                _, owner = self._variable_or_component(name)
            except ValueError:
                raise ValueError(
                    "no variable named %r to drop; found %s"
                    % (name, ", ".join(sorted(self.variables)) or "none"))
            raise ValueError(
                "%r is a component of %r and cannot be dropped on its own; "
                "drop %r whole" % (name, owner, owner))
        return self

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
        # imported late: the mesh adapters live beside the mesh classes,
        # which subclass the containers defined here
        from geoml.data.meshes import (_sheet_interpolator, _check_covered,
                                     _side_codes)
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
        from geoml.data.meshes import _closed_body
        self._check_three_dimensional()
        coordinates = _np.asarray(self.coordinates, dtype=float)
        inside = _gmt.inside_solid(_closed_body(solid), coordinates)
        self.add_metadata(name, inside.astype(_np.int8), labels=list(labels))

    def assign_from_data(self, data: "_SpatialData | _types.ArrayLike",
                         distance: "float | None" = None,
                         hull: "float | None" = None,
                         transform: "_tr._Transform | None" = None,
                         name: str = "near_data") -> _np.ndarray:
        """
        Records which locations lie within the data's reach.

        A model speaks for the ground its data informs and no further, and
        any comparison of the two -- a swath plot above all -- is only fair
        over that ground. A location is in reach when it lies within
        `distance` of a sample, or inside the data's concave hull at the
        length `hull`, or both: the hull fills the interior between drill
        fences closer than its length, where a ball around each sample
        would leave a gap, and leaves out a notch wider than it, where a
        convex hull would bridge across. Giving both is a hull with a
        margin.

        The answer is a boolean metadata column, so it survives subsetting
        and Zarr, a block model's children inherit it across `split` and
        `refine`, and `where=` takes it by name.

        Parameters
        ----------
        data
            The samples: a spatial container, or an `(n, n_dim)` array of
            coordinates.
        distance
            Radius around each sample within which a location is in
            reach. In the transformed units when `transform` is given.
        hull
            Length of the concave hull -- the largest circumradius a
            Delaunay simplex may have and still be part of it. In the
            transformed units when `transform` is given. Data that do not
            span their space (all on a line or plane) have no hull, and
            the distance alone decides, with a warning.
        transform
            A spatial transform from `geoml.transform`, applied to the
            samples and to this object's locations before any distance is
            measured -- how drillhole anisotropy is said, dense down the
            hole and sparse across: an `Anisotropy3D` with the ranges of
            the drilling. `None` measures in the coordinates' own units.
        name
            Name of the metadata column to write. An existing column with
            this name is replaced.

        Returns
        -------
        reach : array
            `(n_data,)` booleans, the column just written.

        Raises
        ------
        ValueError
            If neither `distance` nor `hull` is given, or the dimensions
            differ.

        See Also
        --------
        assign_from_surface, assign_from_solid : the same kind of column
            from a surface or a body.
        geoml.math.geometry.concave_hull : the hull itself.
        """
        if distance is None and hull is None:
            raise ValueError("give a distance, a hull length, or both")

        samples = _np.asarray(
            getattr(data, "coordinates", data), dtype=float)
        targets = _np.asarray(self.coordinates, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != targets.shape[1]:
            raise ValueError(
                "the data must have the same dimension as this object "
                "(%d)" % targets.shape[1])
        if transform is not None:
            transform.refresh()
            samples = _np.asarray(transform(samples), dtype=float)
            targets = _np.asarray(transform(targets), dtype=float)

        reach = _np.zeros(targets.shape[0], dtype=bool)
        if distance is not None:
            nearest, _ = _spatial.cKDTree(samples).query(targets)
            reach |= nearest <= distance
        if hull is not None:
            try:
                reach |= _gmt.concave_hull(samples, hull).contains(targets)
            except ValueError as error:
                if distance is None:
                    raise
                import warnings
                warnings.warn(
                    "no concave hull for these data (%s); the distance "
                    "alone decides the reach" % error)
        self.add_metadata(name, reach)
        return reach

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
    def n_data(self) -> int:
        """How many locations this container holds."""
        return self._n_data

    @property
    def diagonal(self) -> float:
        """The length of the bounding box's diagonal."""
        return self._bounding_box.diagonal

    def aspect_ratio(self, vertical_exaggeration: float = 1) -> dict:
        """Plotly layout data keeping the axes in proportion.

        Parameters
        ----------
        vertical_exaggeration
            How much to stretch the vertical axis.

        Returns
        -------
        dict
            A `layout` fragment for a plotly figure.

        Raises
        ------
        ValueError
            In one dimension, where there is nothing to keep in
            proportion.
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

    def to_zarr(self, path: _types.PathLike) -> _types.PathLike:
        """Persist this container to a single on-disk Zarr store.

        Coordinates and every variable's arrays are written into one Zarr group
        at ``path``, streamed chunk-by-chunk (never fully materialized), along
        with the metadata needed to rebuild the container with :meth:`open`.
        All point-based containers and variable types are supported;
        ``DrillholeData`` is not (it is raw input data, not a prediction
        target).
        """
        # imported late: the Zarr writers name every container class, and
        # those classes subclass what this module defines
        from geoml.data.io import (
            _GEOML_ZARR_FORMAT, _write_container, _write_metadata,
            _write_variable)
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
    def open(cls, path: _types.PathLike) -> "_SpatialData":
        """Rebuild a container previously saved with :meth:`to_zarr`.

        The stored container type is honoured regardless of which class ``open``
        is called on. Variable arrays are reopened on disk (read/write), so the
        result can be inspected or predicted into again without recomputing.
        """
        from geoml.data.io import (
            _GEOML_ZARR_FORMAT, _rebuild_container, _rebuild_metadata,
            _rebuild_variable)
        group = _zarr.open_group(path, mode="r+")
        meta = dict(group.attrs["geoml"])
        written = meta.get("geoml_format", 1)
        if written != _GEOML_ZARR_FORMAT:
            # refused outright rather than half-loaded with quiet gaps: the
            # dict families and cut-offs moved when the keys aligned with the
            # paths, and a format-1 store would open with those missing
            raise ValueError(
                "%r was written at geoml store format %d and this version "
                "reads %d; re-create it by predicting again"
                % (path, written, _GEOML_ZARR_FORMAT))
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

    def add_continuous_variable(
            self, name: str,
            measurements: _types.ArrayLike | None = None,
            unit: "_types.Unit | None" = None) -> None:
        """
        Adds a continuous variable to this point set.

        Parameters
        ----------
        name
            Variable name.
        measurements
            The variable's values, one per data location.
        unit
            What the values are measured in -- `"%"`, `"ppm"`, `"g/t"`, or
            a number. A label here: the model reads the values as they
            stand, and the unit travels so that a figure can say what an
            axis is in and an export can record it.
        """
        # self.variables[name] = ContinuousVariable(
        #     name, self, measurements, quantiles=quantiles,
        #     probabilities=probabilities)
        self.variables[name] = ContinuousVariable(
            name, self, measurements, unit=unit)

    def add_vector_variable(
            self, name: str, labels: _types.Labels | None = None,
            measurements: _types.ArrayLike | None = None,
            units: _types.Units = None) -> None:
        """
        Adds a vector variable to this point set.

        Parameters
        ----------
        name
            Variable name.
        labels
            One name per component.
        measurements
            A matrix with one column per component.
        units
            What the components are measured in: one per label, or a
            mapping naming some of them. Labels here, as for a continuous
            variable -- the components go to the model as they stand.
        """
        self.variables[name] = VectorVariable(
            name, self, labels, measurements, units=units)

    def add_categorical_variable(
            self, name: str, labels: _types.Labels | None = None,
            measurements: _types.ArrayLike | None = None) -> None:
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

    def add_binary_variable(
            self, name: str, labels: _types.Labels | None = None,
            measurements: _types.ArrayLike | None = None) -> None:
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

    def add_anomaly_variable(
            self, name: str, label: str,
            measurements: _types.ArrayLike | None = None) -> None:
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

    def add_compositional_variable(
            self, name: str, labels: _types.Labels,
            measurements: _types.ArrayLike | None = None,
            units: _types.Units = None, rest: bool = False) -> None:
        """
        Adds a compositional variable to this data set.

        The parts are kept in the units they were measured in and turned
        into fractions of the whole only where the model reads them, so
        everything reported of them -- predictions, simulations, quantiles,
        the two variances -- comes back in the unit each part was assayed
        in. Declaring the units is what makes that possible: parts in
        different units cannot be added up, and a composition is defined by
        its sum.

        What the values are put through, in this order: a row missing any
        one part is marked missing entirely, the parts of a composition
        carrying information only relative to each other; non-positive
        parts are replaced by half the smallest positive value of their own
        column, since a log-ratio transform cannot take a zero; and the
        rows are closed, either by adding a `rest` part holding whatever is
        left of the whole or by scaling each row to sum to one. Data that
        already arrives closed and positive is left exactly as it is.

        Parameters
        ----------
        name : str
            Variable name.
        labels : tuple
            The labels for each part in the composition.
        measurements : array-like
            A rank 2 matrix containing the compositions, one column per
            label, each in its own unit.
        units : mapping or sequence
            What each part is measured in -- `"%"`, `"ppm"`, `"g/t"`, or a
            number to divide by. One per label, or a mapping naming some of
            them. Undeclared parts are read as fractions, which is what the
            whole composition was before units existed. With `rest`, the
            mapping may name `"rest"` as well; it is a fraction otherwise.
        rest : bool
            Whether to add a further part holding whatever is left of the
            whole. Recommended wherever the parts are a few assayed metals:
            their own numbers then survive untouched, where closing without
            a rest turns them into shares of what was measured.
        """
        labels = list(labels)
        if rest:
            labels = labels + ["rest"]
        per_label = _units_per_label(units, labels)

        if measurements is not None:
            measurements = _prepare_composition(
                measurements, labels, per_label, rest, name)

        self.variables[name] = CompositionalVariable(
            name, self, labels, measurements, units=per_label)

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

    def as_data_frame(self, metadata=True, include="**", simulations=False,
                      columns="flat"):
        """
        Conversion of a spatial object to a data frame.

        Metadata first (bare names, the way `HOLEID` is read back), then the
        coordinates, then every filled column of every variable, named by its
        path -- `assay_Zn_prediction`. `include` chooses what comes
        (`"**/prediction"`, `"assay/**"`), `simulations` how many realizations,
        and `columns="multi"` keeps the path as one `MultiIndex` level per
        segment instead of flattening -- for staying in pandas; written to
        CSV it makes several header rows, which other software reads as data.
        """
        found = []
        if metadata:
            found += [(VariablePath((name,)), column.to_numpy())
                      for name, column in self.metadata.items()]
        coords = _np.asarray(self.coordinates)
        found += [(VariablePath((str(label),)), coords[:, i])
                  for i, label in enumerate(self.coordinate_labels)]
        found += list(self._export_columns(include, simulations))
        return _frame_from_columns(found, columns)

    @staticmethod
    def default_coordinate_labels(n_dim: int) -> list[str]:
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
        # read straight off this container rather than off a `deepcopy` of it:
        # the copy duplicated every store -- simulations included -- and every
        # one of them was then thrown away by the subset that replaced it.
        # Nothing here mutates the source; each variable's `__getitem__`
        # builds its own object.
        new_obj = self._subset_coordinates(item)
        self._subset_metadata(new_obj, item)
        for name, var in self.variables.items():
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
        if not self.n_dim == 3:
            raise ValueError("as_pyvista method is only supported "
                             "for 3-dimensional data")

        pv_points = _pv.PolyData(_np.asarray(self.coordinates))
        return self._finish_pyvista(pv_points, "points", simulations, include)

    @classmethod
    def from_geoh5(cls, workspace: "_types.PathLike | _GeoH5Workspace",
                   name: "str | None" = None) -> "PointData":
        """
        Reads a Points object from a geoh5 workspace.

        The columns come in as measured variables: float data as
        continuous variables and referenced (coded) data as categorical
        ones, named as the file names them, with geoh5's "Unknown" code
        reading as not measured. What the file has no room to say —
        which column was a prediction, whose tree it belonged to — is not
        guessed at: a geoML container round-trips whole through
        `to_zarr`, and this reader is for data somebody else made.

        `name` says which Points object to read, and may be left out when
        the workspace holds exactly one; `geoml.data.geoh5.contents`
        lists what there is to name. Needs the `geoh5py` package:
        `pip install geoml[geoh5]`.

        Parameters
        ----------
        workspace
            Path of the workspace to read, or an open
            `geoml.data.geoh5.Workspace`.
        name
            The Points object to read, when the file holds more than one.

        Returns
        -------
        data : PointData
            The vertices as locations, the columns as variables.

        Raises
        ------
        ValueError
            If the workspace holds no such Points object — the message
            lists what it does hold.
        """
        # late, through the module: the geoh5 machinery must not load,
        # nor its optional dependency be missed, before a file is asked for
        import geoml.data.geoh5 as _geoh5io
        vertices, floats, coded = _geoh5io.read_points(workspace, name)
        data = PointData(_pd.DataFrame(vertices, columns=["X", "Y", "Z"]),
                         ["X", "Y", "Z"])
        for column, values in floats:
            data.add_continuous_variable(column, values)
        for column, labels, values in coded:
            # the empty string is how the reader spells geoh5's "Unknown";
            # None is what pandas reads as a missing category
            values = _np.asarray(values, dtype=object)
            values[values == ""] = None
            data.add_categorical_variable(column, labels=labels,
                                          measurements=values)
        return data

    def to_geoh5(self, workspace: "_types.PathLike | _GeoH5Workspace",
                 name: str = "Points", include: str = "**",
                 simulations: "bool | int | Sequence[int]" = False,
                 replace: bool = True,
                 folder: "str | None" = None) -> None:
        """
        Writes this object into a geoh5 workspace, as a Points object.

        A workspace is what Geoscience ANALYST — a free viewer for the
        format — opens as **one** project, so a model's pieces belong in
        one file: writing into an existing path adds the object beside
        what is already there, and several exports in a row go fastest
        through an open `geoml.data.geoh5.Workspace`, which holds the
        file open across them.

        The columns are the ones every export carries, named as the
        pyvista export names them, with categorical columns as geoh5's
        own *referenced* data; the mapping from each name back to the
        path that produced it rides in the object's metadata under
        ``geoml_paths``, since no rendered name parses back. Needs the
        `geoh5py` package: `pip install geoml[geoh5]`.

        Parameters
        ----------
        workspace
            Path of the workspace to write into, or an open
            `geoml.data.geoh5.Workspace`.
        name
            The name the object gets in the workspace.
        include
            A path pattern selecting the columns to carry, as in
            `as_pyvista`.
        simulations
            Which simulations to include: `False` for none (the default,
            since each one is a full-length array in the file), `True`
            for all of them, an `int` for the first n, or a sequence of
            indices.
        replace
            Whether an existing Points object of this name, in this
            folder, makes way — what a re-run export script means.
            `False` keeps both, and reading that name back then requires
            saying which.
        folder
            Where the object sits in ANALYST's project tree, as a path —
            `"Data/Assays"` — each segment a group, created when it does
            not exist and reused when it does. `None` is the root.
        """
        if self.n_dim != 3:
            raise ValueError(
                "a geoh5 workspace holds 3-dimensional objects; this data "
                "has %d coordinates per location" % self.n_dim)
        import geoml.data.geoh5 as _geoh5io
        _geoh5io.write_points(self, workspace, name, include, simulations,
                              replace, folder)

    def decluster(self, on: "str | None" = None,
                  cell: "float | None" = None,
                  name: str = "declustering"
                  ) -> "tuple[_np.ndarray, float]":
        """
        Computes cell-declustering weights and keeps them, for everything
        downstream to share.

        Samples are rarely laid down evenly — drilling follows the ore —
        and every statistic that gives them equal votes describes the
        sampling rather than the field. This stores one weight per
        location as the metadata column `"declustering"`, which is where
        the declustered consumers look first: the `variogram` figure, and
        the warping initializers a model runs at construction. **One call
        here turns declustering on everywhere**, with one consistent set
        of weights — computed independently, each consumer would sweep
        its own cell size on its own values and quietly disagree.

        The weights are a snapshot of these locations, exactly as a fold
        column is: a subset carries its parent's values, which are then
        not the subset's own weights — recompute after subsetting when it
        matters.

        Parameters
        ----------
        on
            The continuous variable (or component) whose values drive the
            automatic cell-size choice — the sweep keeps the cell whose
            declustered mean departs furthest from the naive one, which
            needs values to mean anything. Left out, the only continuous
            variable is used; holding several, this refuses to guess.
            Not needed when `cell` is given.
        cell
            The cell side, fixing it instead of sweeping.
        name
            The metadata column written. An existing column of this name
            is replaced.

        Returns
        -------
        weights : array
            One per location, summing to the number of locations, as
            `math.geometry.declustering_weights` returns them.
        cell : float
            The cell side used, whether given or chosen.

        See Also
        --------
        math.geometry.declustering_weights : the arithmetic, and the
            cell-sweep rule.
        """
        coordinates = _np.asarray(self.coordinates, dtype=float)
        if cell is None:
            if on is None:
                continuous = [key for key, variable in self.variables.items()
                              if isinstance(variable, ContinuousVariable)]
                if len(continuous) != 1:
                    raise ValueError(
                        "the automatic cell sweep needs values: name the "
                        "variable with `on=` (found %s) or fix the size "
                        "with `cell=`"
                        % (", ".join(sorted(continuous)) or "no continuous "
                           "variable"))
                on = continuous[0]
            variable, _ = self._variable_or_component(str(on))
            measurements = getattr(variable, "measurements", None)
            if measurements is None or measurements.labels is not None:
                raise ValueError(
                    "%r holds no continuous measurements to sweep the "
                    "cell size on" % str(on))
            values = _np.asarray(measurements.values, dtype=float).ravel()
            weights, cell = _gmt.declustering_weights(coordinates, values)
        else:
            weights, cell = _gmt.declustering_weights(
                coordinates, cell=float(cell))
        self.add_metadata(name, weights)
        return weights, float(cell)

    def spatial_k_fold(self, test_data, k=5, groups=None, seed=None,
                       name="fold"):
        """
        Builds cross-validation folds that mimic a prediction task.

        A random fold is answered by its neighbours and flatters every
        score; folds pushed as far from the training data as possible
        overshoot the other way, testing an extrapolation nobody asked
        for. What decides how hard a location is to predict is how far its
        nearest training point sits, so the folds chosen here are the ones
        whose held-out-to-training distances are distributed like the
        distances from `test_data` -- the object the model is actually
        meant to predict -- to this data. This is the nearest-neighbour
        distance matching idea of Linnenbrink et al. (2024), built on
        discrete groups: continuous per-sample weightings can match the
        distributions perfectly while the folds are spatially wrong.

        The data is first gathered into small groups that are never split
        across folds -- the samples of one drill hole stand or fall
        together -- and the candidate partitions come from cutting a Ward
        dendrogram of the group centroids at every count from `k` clusters
        up to one cluster per group, each cut's clusters dealt to the
        emptiest fold largest-first. The cut whose Wasserstein distance to
        the target distribution is smallest wins, and the result is
        written to a metadata column -- ``"fold"`` unless `name` says
        otherwise, which is also what `models.cross_validate` reads by
        default.

        Parameters
        ----------
        test_data
            The spatial object the model is meant to predict -- a grid, a
            block model, or any container with coordinates.
        k : int
            Number of folds.
        groups : str, optional
            Name of a metadata column whose labels must never be split
            across folds (a drill hole id). Without one, the data is
            pre-clustered into many small spatial groups.
        seed : int
            Passed to `sklearn.cluster.KMeans` for a reproducible
            pre-clustering when `groups` is not given. The rest of the
            search is deterministic.
        name : str
            The metadata column to write the folds to. An existing column
            with this name is replaced, so two calls with two names give
            two labellings to compare.

        Returns
        -------
        w : float
            The Wasserstein distance between the two distributions below,
            in coordinate units. Zero is a perfect match.
        target_distances : array
            Distance from each of `test_data`'s locations to its nearest
            data point -- the prediction task.
        fold_distances : array
            Distance from each data point to its nearest training point
            when its fold is held out -- the task the cross-validation
            poses.
        """
        coords = _np.asarray(self.coordinates, dtype=float)
        n = coords.shape[0]
        if k < 2:
            raise ValueError("k must be at least 2")

        if groups is None:
            n_atoms = min(n, 25 * k)
            atom = _KMeans(n_atoms, n_init=10, random_state=seed).fit(
                coords).labels_
        else:
            _, atom = _np.unique(self.get_metadata(groups),
                                 return_inverse=True)
            n_atoms = int(atom.max()) + 1
        if n_atoms < k:
            raise ValueError(f"cannot build {k} folds from {n_atoms} groups")

        atom_sizes = _np.bincount(atom, minlength=n_atoms)
        centroids = _np.stack(
            [coords[atom == i].mean(axis=0) for i in range(n_atoms)])

        # the prediction task's own distances
        test_coords = _np.asarray(test_data.coordinates, dtype=float)
        if test_coords.shape[0] > 10000:
            stride = int(_np.ceil(test_coords.shape[0] / 10000))
            test_coords = test_coords[::stride]
        target = _spatial.cKDTree(coords).query(test_coords, k=1)[0]

        if n_atoms > k:
            link = _hierarchy.linkage(centroids, method="ward")
        candidates = _np.arange(k, n_atoms + 1)
        if candidates.size > 50:
            candidates = _np.unique(
                _np.geomspace(k, n_atoms, 50).round().astype(int))

        best = None
        for n_clusters in candidates:
            if n_clusters == n_atoms:
                cluster = _np.arange(n_atoms)
            else:
                cluster = _hierarchy.fcluster(
                    link, t=n_clusters, criterion="maxclust") - 1
            if cluster.max() + 1 < k:
                continue
            fold = _balanced_assignment(cluster, atom_sizes, k)[atom]

            pooled = _np.concatenate([
                _spatial.cKDTree(coords[fold != f]).query(
                    coords[fold == f], k=1)[0]
                for f in range(k)])
            w = _sstats.wasserstein_distance(pooled, target)
            if best is None or w < best[0]:
                best = (w, fold, pooled)

        assert best is not None    # at least one cut is always tried
        w, fold, pooled = best
        self.add_metadata(name, fold)
        return w, target, pooled


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

    def as_data_frame(self, **kwargs):
        """
        Conversion of a spatial object to a data frame.
        """
        df = _PointBased.as_data_frame(self, **kwargs)
        directions = _pd.DataFrame(self.directions,
                                   columns=self.direction_labels)
        return _pd.concat([directions, df], axis=1)

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
        from geoml.data import Grid2D
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

        return self._finish_pyvista(pv_surf, "points", simulations, include)


