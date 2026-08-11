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
The container tree: the errors, the bounding box, the path grammar
(`VariablePath`, `render`), the traversal (`_TreeNode`) and the leaf it
carries (`_Attribute`). Everything a container or a variable is built on,
and nothing that is one.
"""
import collections as _col
import copy as _copy
import fnmatch as _fnmatch
import warnings as _warnings

import numpy as _np
import pandas as _pd
from skimage import measure as _measure
from skimage import filters as _filters

import geoml.storage as _storage
import geoml.viz.plotly as _py

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


def _frame_from_columns(found, columns):
    """A data frame from `(path, values)` pairs, named by the chosen style.

    `flat` renders each path with underscores, deduplicated with a warning
    where two paths land on one name; `multi` keeps the path as one
    `MultiIndex` level per segment, shorter paths padded with empty strings.
    """
    if not found:
        return _pd.DataFrame({})
    paths = [path for path, _ in found]
    if columns == "flat":
        names = list(render_all(paths, "flat").values())
    elif columns == "multi":
        depth = max(len(path) for path in paths)
        names = _pd.MultiIndex.from_tuples(
            [path.parts + ("",) * (depth - len(path)) for path in paths])
    else:
        raise ValueError(
            "unknown columns style %r; expected 'flat' or 'multi'"
            % (columns,))
    return _pd.DataFrame(
        dict(zip(range(len(found)), (values for _, values in found)))
    ).set_axis(names, axis=1)


def _subset_simulations(store, item):
    """The `item` rows of a simulations store, without holding it whole.

    `np.asarray(store)[item]` reads every realization of every location into
    memory before throwing most of them away, which on a block model is
    hundreds of gigabytes and the end of the session. Dask indexes the chunks
    instead, so what is materialized is the answer rather than the source.
    """
    return _storage.ArrayStore.from_numpy(
        _np.asarray(store.as_dask()[item]))


def _copy_for_subset(variable):
    """A deep copy of a variable, without what the subset is about to replace.

    Two things a plain `deepcopy` drags in, both of them read in full and both
    thrown away a line later. Its own simulations, which are the whole
    `(n_data, n_sim)` array. And, less obviously, *every other variable in the
    container*: an `_Attribute` holds a reference back to its container, so
    the copy walks up to it and down again into everything else it holds.
    Seeding the memo makes the copy share the container instead, and the
    caller points the result at the new one with `set_coordinates`.
    """
    memo = {}
    coordinates = getattr(variable, "coordinates", None)
    if coordinates is not None:
        memo[id(coordinates)] = coordinates

    # the whole subtree, not just this node: a vector variable's realizations
    # hang off its components
    stashed = []
    for _, node in variable.walk():
        store = getattr(node, "simulations", None)
        if store is not None:
            stashed.append((node, store))
            node.simulations = None
    try:
        return _copy.deepcopy(variable, memo)
    finally:
        for node, store in stashed:
            node.simulations = store


def _column_detail(attribute, status=True):
    """What a column shows in a printed tree: its type, or that it is empty."""
    detail = str(_np.dtype(attribute.values.dtype))
    if attribute.labels is not None:
        detail = "%s codes, %d labels" % (detail, len(attribute.labels))
    if status and not attribute._has_content():
        return "empty"
    return detail


def _match_path(pattern, parts):
    """Whether `parts` matches a glob `pattern`, segment by segment.

    `*` stands for one segment and does not cross a `/`; `**` stands for zero
    or more, so `assay/**/prediction` finds `assay/prediction` and
    `assay/Zn/prediction` alike. Within a segment the usual shell wildcards
    apply, and matching is case-sensitive because labels are.
    """
    if not pattern:
        return not parts
    head = pattern[0]
    if head == "**":
        return any(_match_path(pattern[1:], parts[i:])
                   for i in range(len(parts) + 1))
    if not parts:
        return False
    if not _fnmatch.fnmatchcase(parts[0], head):
        return False
    return _match_path(pattern[1:], parts[1:])


def render(path, style="path"):
    """An addressable path as a name in some flat namespace.

    Purely mechanical -- the segments joined, nothing else. That is the point:
    the four spellings this replaced each had rules of their own about which
    role was abbreviated and which was dropped, and no two agreed.

    `path` is what the store and every internal caller use, `flat` is for
    data-frame columns, CSV and mining software (identifier-safe), and
    `pretty` is for pyvista and ParaView, where the name is read by a person.
    """
    parts = VariablePath(path).parts
    if style == "path":
        return PATH_SEP.join(parts)
    if style == "flat":
        return "_".join(parts)
    if style == "pretty":
        return " - ".join(parts)
    raise ValueError(
        "unknown render style %r; expected 'path', 'flat' or 'pretty'"
        % (style,))


def render_all(paths, style="flat"):
    """`{path: name}` for a whole namespace at once, every name distinct.

    A path cannot collide -- `/` is refused inside a segment -- so a collision
    is made by the *join*, and only `flat` makes one readily: `_` is in nearly
    every role name, so a variable `noise` with a component `variance` lands on
    the same column as a leaf called `noise_variance`.

    The rule has to be deterministic or an export changes shape between runs,
    so a colliding group is sorted **by path** and the ones after the first
    take a suffix. Only the group is touched: suffixing all of them would
    penalize the innocent column to spare the pathological one. Adding a
    variable can therefore rename a column inside a colliding group, which is
    why this warns rather than quietly putting it right.
    """
    grouped = _col.OrderedDict()
    order = []
    for path in paths:
        path = VariablePath(path)
        order.append(path)
        grouped.setdefault(render(path, style), []).append(path)

    names = {}
    for name, group in grouped.items():
        if len(group) == 1:
            names[group[0]] = name
            continue
        group = sorted(group, key=lambda p: p.parts)
        chosen = [name] + ["%s_%d" % (name, i)
                           for i in range(2, len(group) + 1)]
        names.update(zip(group, chosen))
        _warnings.warn(
            "%d paths render to the column name %r and were made distinct: %s"
            % (len(group), name,
               ", ".join("%s -> %s" % (p, names[p]) for p in group)),
            stacklevel=2)

    return _col.OrderedDict((path, names[path]) for path in order)


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

    def set_coordinates(self, coordinates):
        """Point this node, and everything under it, at a new container.

        One walk instead of six hand-written lists, four of which had columns
        missing: a component set only its prediction, a rock type set nothing
        but its own reference, and every dict family was missed everywhere.
        A stale reference is quiet until something asks the attribute how many
        locations it has.
        """
        for _, node in self.walk():
            node.coordinates = coordinates
            for _, attribute in node.own_leaves():
                attribute.coordinates = coordinates

    def node_attrs(self):
        """The facts describing this node, as `{name: value}`.

        Cut-offs and the like: not arrays, not rebuilt from anything else, and
        therefore invisible to any code that only knows about columns. Every
        method that builds a fresh variable has to carry them across, and the
        three that forgot are why they are enumerated here.
        """
        return {name: getattr(self, name, None) for name in self._NODE_ATTRS}

    def _export_leaves(self, include="**", simulations=False, prefix=None):
        """`(path, attribute)` for every filled column matching `include`.

        The one enumeration behind every export -- tabular and pyvista alike,
        in place of the thirty-five per-class bodies that each named their
        columns again: a filled leaf is a column, a dict family is one column
        per cut-off, and the realization axis is unrolled only as far as the
        `simulations` selector asks. Metadata stays out -- it has a root of
        its own precisely so a fold over the modelled columns can leave it
        alone.
        """
        pattern = VariablePath(include).parts
        for path, node in self.walk(prefix):
            for leaf_parts, attribute in node.own_leaves():
                full = path / leaf_parts
                if full.parts[:1] == (METADATA_ROOT,):
                    continue
                if not _match_path(pattern, full.parts):
                    continue
                if not attribute._has_content():
                    continue
                yield full, attribute
            store = getattr(node, "simulations", None)
            if store is not None:
                for i, values in _selected_simulations(store, simulations):
                    full = path / "simulations" / str(i)
                    if _match_path(pattern, full.parts):
                        yield full, _Attribute(node.coordinates, values)

    def _export_columns(self, include="**", simulations=False):
        """`(path, values)` for the frame: the leaves, decoded."""
        for path, attribute in self._export_leaves(include, simulations):
            yield path, attribute.to_numpy()

    def as_data_frame(self, include="**", simulations=False, columns="flat",
                      **kwargs):
        """This node's filled columns, one data-frame column each.

        Parameters
        ----------
        include : str
            A path pattern choosing what to export: `"**"` for everything,
            `"**/prediction"` for the predictions alone, `"Zn/**"` for one
            component. See `select`.
        simulations : bool, int or sequence
            Which realizations to include: none by default, `True` for all,
            an `int` for the first n, or a sequence of indices.
        columns : str
            `"flat"` names each column by its path with underscores
            (`assay_Zn_prediction`), deduplicated with a warning if two paths
            land on one name. `"multi"` keeps the path as a `MultiIndex`
            level per segment -- for staying in pandas; written to CSV it
            makes several header rows, which other software reads as data.
        """
        return _frame_from_columns(
            list(self._export_columns(include, simulations)), columns)

    def _copy_attrs_into(self, new):
        """Carry every node's facts into a freshly built copy of this variable.

        `from_variable` builds the structure -- the components, the empty
        columns -- and this walks the two trees in step, copying what each
        node *knows* rather than what it holds. Wholesale, off `_NODE_ATTRS`,
        because carrying them by hand per class is how a composition came to
        lose its components' cut-offs three separate times: whoever adds a
        fact adds it to the declaration, and every rebuild carries it.
        """
        for name, value in self.node_attrs().items():
            if value is not None:
                setattr(new, name, _copy.deepcopy(value))
        theirs = new.child_nodes()
        for label, child in self.child_nodes().items():
            if label in theirs:
                child._copy_attrs_into(theirs[label])

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

    def addressable(self, prefix=None, realizations=False):
        """`(path, thing)` for everything a path can name here -- nodes and
        columns alike, each node before the columns it owns.

        `realizations` unrolls a simulations store into one entry per
        realization. Off by default: on a hundred-realization model that is a
        hundred extra names per variable, and they are wanted only when
        something asked for them by name.
        """
        for path, node in self.walk(prefix):
            yield path, node
            for parts, attribute in node.own_leaves():
                yield path / parts, attribute
            store = getattr(node, "simulations", None)
            if store is not None:
                yield path / "simulations", store
                if realizations:
                    for i in range(store.shape[1]):
                        yield path / "simulations" / str(i), node.simulation(i)

    def select(self, pattern="**", filled=None):
        """`{path: thing}` for everything matching a glob pattern.

        `select("**/prediction")` is every prediction in the tree,
        `select("assay/*")` the components of one variable, `select("assay/**")`
        that variable and everything under it.

        `filled` is the one thing no pattern can express and every export
        needs: `True` keeps only columns that hold something, `False` only the
        empty ones, and either way nodes are left out, since a node is neither.
        A `role=` keyword was considered and rejected -- `role="prediction"` is
        exactly `**/prediction`, and a second way to say the same thing is the
        disease this addressing was built to cure.

        A bare `**` does not unroll the realization axis: the leaf is
        `assay/Zn/simulations`, and `assay/Zn/simulations/7` is reached by
        naming it (`get`) or by asking for it (`**/simulations/*`).
        """
        parts = VariablePath(pattern).parts
        # unrolled only when the pattern names the axis and then asks for
        # something past it: `**/simulations/*` yes, `**` and
        # `**/simulations` no
        realizations = "simulations" in parts[:-1]
        chosen = _col.OrderedDict()
        for path, thing in self.addressable(realizations=realizations):
            if not _match_path(parts, path.parts):
                continue
            if filled is not None:
                if not isinstance(thing, _Attribute):
                    continue
                if thing._has_content() != filled:
                    continue
            chosen[path] = thing
        return chosen

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

    # ------------------------------------------------------------------ #
    # printing the tree
    # ------------------------------------------------------------------ #
    def tree(self, status=True):
        """This node and everything beneath it, as a printable tree.

        Separate from `str`, which stays one line per variable on purpose: a
        container's summary should not grow every time a variable does. This
        is the diagnostic -- what is here, and which of it holds anything.

        `status` reads each column once to say whether it is filled, which is
        the question worth asking (a column allocated and never written is
        the shape of most of the bugs this addressing was built to stop). On
        a disk-backed block model that is a pass over every column, so
        `status=False` prints the structure alone.
        """
        lines = [self._tree_root_label()]
        self._tree_rows(lines, "", status)
        return "\n".join(lines)

    def _tree_root_label(self):
        name = self._node_name
        return "%s%s" % (type(self).__name__, " %r" % name if name else "")

    def _node_detail(self):
        facts = ", ".join(
            "%s=%s" % (name, value)
            for name, value in self.node_attrs().items() if value is not None)
        return type(self).__name__ + ("  " + facts if facts else "")

    def _tree_entries(self, status):
        """`(label, detail, child)` under this node, in printing order."""
        rows = [(str(VariablePath(parts)), _column_detail(attribute, status),
                 None)
                for parts, attribute in self.own_leaves()]
        store = getattr(self, "simulations", None)
        if store is not None:
            rows.append(("simulations", "%s %s" % (tuple(store.shape),
                                                   _np.dtype(store.dtype)),
                         None))
        rows += [(name, child._node_detail(), child)
                 for name, child in self.child_nodes().items()]
        return rows

    def _tree_rows(self, lines, prefix, status):
        rows = self._tree_entries(status)
        width = max(8, 26 - len(prefix))
        for i, (label, detail, child) in enumerate(rows):
            last = i == len(rows) - 1
            lines.append("%s%s%s %s" % (prefix, "`-- " if last else "|-- ",
                                        label.ljust(width), detail))
            if child is not None:
                child._tree_rows(lines, prefix + ("    " if last else "|   "),
                                 status)

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
        # A shallow copy, deliberately. An attribute points back at its
        # container, so `deepcopy` walked up to it and down again into every
        # other variable -- reading every store in the object, simulations
        # included, to build a copy thrown away on the next line. The only
        # thing here worth copying is the values, and those are replaced.
        new_obj = _copy.copy(self)
        if self.labels is not None:
            new_obj.labels = list(self.labels)
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
        # imported late: the concrete containers subclass this tree
        from geoml.data import Grid1D
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
        from geoml.data import Grid2D
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
        from geoml.data import Grid3D
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
        from geoml.data import (Grid1D, Grid2D, Grid3D, Blocks1D, Blocks2D,
                                Blocks3D, RotatedGrid3D)
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
        from geoml.data import mesh3d
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
        from geoml.data import Section3D, Surface3D
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


