# geoML - machine learning models for geospatial data
# Copyright (C) 2026  Ítalo Gomes Gonçalves
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
Mesh sets: every contour of one column of a block model or a grid, keyed
by cut-off -- or by name, for a categorical variable -- the prediction's
and one set per realization, with the reports only a whole set can make.
"""
import collections.abc as _abc
import itertools as _iter
import multiprocessing as _mp
import os as _os
import shutil as _shutil
import tempfile as _tempfile
import warnings as _warnings
import weakref as _weakref

from ezdxf.filemanagement import new as _new_dxf
import numpy as _np
import pandas as _pd
import pyvista as _pv
import zarr as _zarr

import geoml._types as _types
import geoml.math.geometry as _gmt
import geoml.storage as _storage

from typing import TYPE_CHECKING, Any, Iterator, Sequence, cast
if TYPE_CHECKING:
    from geoml.data.geoh5 import Workspace as _GeoH5Workspace

from geoml.data.base import (BoundingBox, InconsistentMeshError,
                             MeshTypeError, NotClosedError, VariablePath,
                             _Attribute, _path_key)
from geoml.data.variables import OrderedRockType, RockTypeVariable, _Category
from geoml.data.containers import PointData
from geoml.data.grids import Grid3D
from geoml.data.meshes import (Mesh3D, Solid3D, Surface3D, _DistanceQueries,
                               _empty_solid, _from_manifold, _ground_under,
                               _joined, _to_manifold, _within_body)
from geoml.data.blocks import BlockSet3D, _contour_column, _sub_block_shares
from geoml.data.io import (_GEOML_ZARR_FORMAT, _rebuild_container,
                           _write_container)

__all__ = ["MeshSet"]

# the layout of a mesh set's own store, recorded beside it
_STORE_FORMAT = 1

# how many bytes of realizations are read at once: the store is chunked by
# rows, so reading one realization visits every chunk, and reading as many
# as fit here costs the same pass
_GROUP_BYTES = 1_000_000_000

# the most processes contouring realizations by default: each holds a
# painted slab of the field and the meshes it has made, several hundred MB
_WORKERS = 8

# what is measured on every realization's mesh, one number per key
_MEASURES = ("volume", "raw", "pieces", "largest", "triangles", "gained",
             "lost", "nudge")

# DXF colour indices for the layers, cycled: red, yellow, green, cyan,
# blue, magenta, then oranges, a green-blue, a violet and two greys
_ACI = (1, 2, 3, 4, 5, 6, 30, 40, 140, 200, 8, 9)


# --------------------------------------------------------------------------- #
# what is contoured
# --------------------------------------------------------------------------- #
def _kept_side(close):
    """`close` as the side it keeps; a set of bodies has no open option."""
    if close is True:
        return "above"
    if close in ("above", "below"):
        return close
    raise ValueError(
        "a mesh set is a set of bodies, so every contour closes: close takes "
        "'above' or 'below' (True meaning 'above'); got %r" % (close,))


def _source(data, path, cutoffs, close):
    """What a set of `path` on `data` contours: its kind, keys, levels, the
    prediction's field for each key, and where the realizations are."""
    if not isinstance(data, (BlockSet3D, Grid3D)):
        raise TypeError(
            "a mesh set contours a block model or a three-dimensional grid; "
            "got %s" % type(data).__name__)
    try:
        node = data.get(VariablePath(str(path)))
    except (KeyError, ValueError):
        node = None

    if isinstance(node, RockTypeVariable):
        if isinstance(node, OrderedRockType):
            raise TypeError(
                "%r is an ordered rock type, read off one implicit field "
                "rather than a field per category; contour its implicit "
                "values with cutoffs= instead" % str(path))
        if cutoffs is not None:
            raise ValueError(
                "a categorical set is keyed by its categories, and has no "
                "use for cutoffs")
        if _kept_side(close) != "above":
            raise ValueError(
                "a category's body is the ground it holds, so there is no "
                "other side for close to keep")
        names = [str(label) for label in node.labels]
        for name in names:
            if "/" in name:
                raise ValueError(
                    "the category %r cannot name a mesh, holding the path "
                    "separator" % name)
        fields = {name: _np.asarray(node.components[name]
                                    .indicator_predicted.values,
                                    dtype=float).ravel()
                  for name in names}
        stores = [node.components[name].simulations for name in names]
        return {"kind": "category", "label": str(VariablePath(str(path))),
                "keys": names, "levels": [0.0] * len(names),
                "fields": fields, "field": None,
                "stores": stores if all(s is not None for s in stores)
                else None,
                "side": "above", "name": str(node.name), "unit": None}
    if isinstance(node, _Category):
        raise TypeError(
            "%r is one category; name its variable, whose set holds a body "
            "for every category" % str(path))

    column_path, column = _contour_column(data, path)
    if not column._has_content():
        raise ValueError("nothing under %r to contour" % str(column_path))
    values = _np.asarray(column.values, dtype=float).ravel()
    owner = None
    if column_path.name == "prediction":
        owner = data.get(column_path.parent)
    if cutoffs is None:
        declared = getattr(owner, "cutoffs", None)
        if not declared:
            raise ValueError(
                "%r declares no cut-offs to contour at; pass cutoffs=, or "
                "declare them with set_cutoffs on the data the model was "
                "trained from" % str(column_path))
        cutoffs = declared
    keys = sorted(set(float(c) for c in _np.atleast_1d(cutoffs)))
    if len(keys) == 0:
        raise ValueError("cutoffs holds nothing to contour at")
    store = getattr(owner, "simulations", None)
    return {"kind": "cutoff", "label": str(column_path), "keys": keys,
            "levels": list(keys), "fields": {key: values for key in keys},
            "field": values, "stores": None if store is None else [store],
            "side": _kept_side(close),
            "name": str(getattr(owner, "name", column_path)),
            "unit": getattr(owner, "unit", None)}


def _numbers(simulations, count):
    """The realizations asked for, as numbers into the variable's
    simulations: `True` all, `False` none, an int the first n, or a list."""
    if simulations is None or simulations is False:
        return []
    if count is None:
        if simulations is True:
            return []
        raise ValueError("this column carries no realizations to contour")
    if simulations is True:
        return list(range(count))
    if isinstance(simulations, (int, _np.integer)):
        return list(range(min(int(simulations), count)))
    numbers = sorted(set(int(i) for i in simulations))
    outside = [i for i in numbers if i < 0 or i >= count]
    if outside:
        raise ValueError(
            "there are %d realizations, numbered from 0; %s are not among "
            "them" % (count, outside))
    return numbers


# How far a contour that will not close is moved off its level to try
# again, as shares of the field's span, nearest first. Where the ground kept
# thins to one cell against the model's edge, the surface can meet the
# closing cap edge-on along one lattice edge -- four triangles sharing it,
# which no winding repair settles and the welded fallback turns into holes:
# measured on the Assen block model, FeO_total at 0.7 came back with one
# such edge, and open at 44 once the fallback had it; contoured 1e-9 lower
# it closes; realization 19 at 0.8 closed only 1e-4 lower, 534 edges open
# at every smaller move. The move is recorded as `nudge` in each mesh's
# provenance; at 1e-4 of the span it is a few millimetres of shell.
_NUDGES = (0.0, -1e-9, 1e-9, -1e-8, 1e-8, -1e-7, 1e-7, -1e-6, 1e-6, -1e-5,
           1e-5, -1e-4, 1e-4)


def _contoured(data, values, level, side, supersample, label):
    if isinstance(data, BlockSet3D):
        return data._contour_values(values, level, label,
                                    supersample=supersample, close=side)
    try:
        return _Attribute(data, _np.asarray(values, dtype=float)) \
            .get_contour(level, close=side)
    except (ValueError, RuntimeError):
        # marching cubes refuses a level the field never reaches, which for
        # a set is an empty answer rather than a failure
        return None


def _shell(data, values, level, side, supersample, label):
    """The body where `values` clear `level` on the side kept -- an empty
    body where they never do -- and how far off `level` it had to be
    contoured to close."""
    finite = _np.asarray(values, dtype=float)
    finite = finite[_np.isfinite(finite)]
    span = float(finite.max() - finite.min()) if finite.size else 0.0
    healed = None
    for share in _NUDGES:
        nudge = share * span
        if share != 0.0 and nudge == 0.0:
            break
        mesh = _contoured(data, values, level + nudge, side, supersample,
                          label)
        if mesh is None or mesh.n_data == 0:
            return _empty_solid(), nudge
        if isinstance(mesh, Solid3D):
            return mesh, nudge
        healed = mesh.heal()
        if isinstance(healed, Solid3D):
            return healed, nudge
    error = NotClosedError if isinstance(healed, Surface3D) \
        else InconsistentMeshError
    raise error(
        "the contour of %r at %g does not close into a body, even healed "
        "and moved a ten-thousandth of its span either side: where the "
        "ground kept thins against the model's edge, its surface can meet "
        "the closing cap edge-on, and a column with missing values leaves "
        "holes a contour runs into" % (label, level))


def _category_fields(draws, rule):
    """Each category's field in one realization: positive where it holds
    the ground, zero along its contacts. `draws` is `(n_rows, n_categories)`.

    `"largest"` is a category's draw against the best of the others -- the
    two row maxima `likelihood._CategoricalLikelihood` takes of the
    probabilities. `"priority"` lets a later category override an earlier
    one wherever its draw is positive, as the hierarchical likelihood does:
    a category holds the ground where its own draw is positive and every
    later one's is negative.
    """
    draws = _np.asarray(draws, dtype=float)
    if rule == "largest":
        best = _np.max(draws, axis=1, keepdims=True)
        winner = draws >= best
        runner_up = _np.max(_np.where(winner, -_np.inf, draws), axis=1,
                            keepdims=True)
        shared = _np.sum(winner, axis=1, keepdims=True) > 1
        return draws - _np.where(winner, _np.where(shared, best, runner_up),
                                 best)
    fields = _np.empty_like(draws)
    later = _np.full(draws.shape[0], -_np.inf)
    for k in range(draws.shape[1] - 1, -1, -1):
        fields[:, k] = _np.minimum(draws[:, k], -later)
        later = _np.maximum(later, draws[:, k])
    return fields


# --------------------------------------------------------------------------- #
# the booleans, in one frame for the whole set
# --------------------------------------------------------------------------- #
# Every mesh of a set is handed to Manifold moved by the same rounded
# corner, so a body converted once -- a limit, the prediction's shell a
# realization is compared against -- serves every operation after it,
# where `Solid3D`'s own booleans convert both sides per call.
def _manifold(mesh, shift):
    """A body in the set's frame, or None for an empty one."""
    if mesh is None or mesh.n_data == 0:
        return None
    return _to_manifold(mesh, shift)


def _volume(body):
    # never below zero: where two bodies only touch, what they share comes
    # back a sliver whose volume is rounding either way
    return 0.0 if body is None else max(0.0, float(body.volume()))


def _mesh(body, shift):
    """A Manifold body back as a mesh, empty where it holds nothing."""
    if body is None or body.is_empty():
        return _empty_solid()
    return _from_manifold(body, shift)


def _difference(a, b):
    """`a` less `b`, as Manifold bodies, either possibly None."""
    if a is None:
        return None
    if b is None:
        return a
    answer = a - b
    return None if answer.is_empty() else answer


def _unchanged(taken, before):
    """Whether cuts taking `taken` from a body of volume `before` left it
    as it was, to rounding."""
    return all(abs(t) <= 1e-12 * max(abs(before), 1.0) for t in taken)


def _prepared(limits, excluded, box):
    """The limits and exclusions as bodies, with the side each keeps.

    A sheet becomes the ground beneath it, built once for the whole set --
    `clip_meshes` does the same -- so keeping a sheet's underneath is an
    intersection and taking it away a difference, like any body.
    """
    bodies = []
    for mapping, keep in ((limits, True), (excluded, False)):
        for name, mesh in mapping.items():
            if isinstance(mesh, Surface3D):
                body = _ground_under(mesh, box) if mesh.n_data > 0 \
                    else _empty_solid()
            elif isinstance(mesh, Solid3D):
                body = mesh
            else:
                raise MeshTypeError(
                    "%r is neither a sheet nor a body, so it has no side to "
                    "keep; got %s" % (name, type(mesh).__name__))
            bodies.append((name, body, keep))
    return bodies


def _limited(mesh, bodies, shift):
    """`mesh` cut by each limit and exclusion in turn, and what each cut
    took. `bodies` holds `(manifold or None, keep)` pairs."""
    if not bodies:
        return mesh, []
    current = _manifold(mesh, shift)
    start = _volume(current)
    taken = []
    for body, keep in bodies:
        before = _volume(current)
        if current is not None:
            if body is None:
                current = None if keep else current
            else:
                current = (current ^ body) if keep else (current - body)
                if current.is_empty():
                    current = None
        taken.append(before - _volume(current))
    if _unchanged(taken, start):
        # nothing was cut away, so the contour stands as it came -- a
        # round trip through Manifold gives back the same body with its
        # triangles welded and renumbered
        return mesh, taken
    return _mesh(current, shift), taken


def _nest(meshes, keys, side, shift):
    """Each shell cut to the one outside it, and what each cut took.

    Above, the shell at a higher cut-off must sit inside the one at the
    lower; below, the other way round.
    """
    order = list(keys) if side == "above" else list(keys)[::-1]
    fixed, removed = {order[0]: meshes[order[0]]}, {order[0]: 0.0}
    outer = _manifold(meshes[order[0]], shift)
    for key in order[1:]:
        inner = _manifold(meshes[key], shift)
        before = _volume(inner)
        if inner is not None:
            inner = None if outer is None else inner ^ outer
        removed[key] = before - _volume(inner)
        if _unchanged([removed[key]], before):
            fixed[key], removed[key] = meshes[key], 0.0
        else:
            fixed[key] = _mesh(inner, shift)
        outer = inner
    return fixed, removed


def _exclusive(meshes, order, shift):
    """Each category's body cut away from the ones before it in `order`,
    and what each cut took."""
    fixed, removed, claimed = {}, {}, None
    for key in order:
        body = _manifold(meshes[key], shift)
        before = _volume(body)
        if body is not None and claimed is not None:
            body = body - claimed
        removed[key] = before - _volume(body)
        if _unchanged([removed[key]], before):
            fixed[key], removed[key] = meshes[key], 0.0
        else:
            fixed[key] = _mesh(body, shift)
        if body is not None:
            claimed = body if claimed is None else claimed + body
    return fixed, removed


def _against(mesh, reference, shift):
    """How much of `mesh` lies outside `reference`, and how much of
    `reference` lies outside `mesh`."""
    body = _manifold(mesh, shift)
    if body is None:
        return 0.0, _volume(reference)
    if reference is None:
        return _volume(body), 0.0
    return float((body - reference).volume()), \
        float((reference - body).volume())


def _pieces(mesh):
    """How many bodies a mesh is made of, and the largest one's share of
    its volume. A cavity's surface is a piece of the mesh but not a body,
    enclosing negative volume, and is not counted."""
    if mesh.n_data == 0:
        return 0, _np.nan
    points = _np.asarray(mesh.coordinates, dtype=float)
    triangles = _np.asarray(mesh.triangles)
    count, labels = _gmt.components(points, triangles)
    corners = (points - points.mean(axis=0))[triangles]
    tetrahedra = _np.einsum("ij,ij->i", corners[:, 0],
                            _np.cross(corners[:, 1], corners[:, 2])) / 6.0
    volumes = _np.bincount(labels, weights=tetrahedra, minlength=count)
    bodies = volumes[volumes > 0]
    total = float(volumes.sum())
    if bodies.size == 0 or total <= 0:
        return 0, _np.nan
    return int(bodies.size), float(bodies.max() / total)


def _measure(mesh):
    pieces, largest = _pieces(mesh)
    return {"volume": float(getattr(mesh, "volume", 0.0)), "pieces": pieces,
            "largest": largest, "triangles": int(len(mesh.triangles))}


def _simplified(mesh, max_error):
    return mesh if mesh.n_data == 0 else mesh.simplify(max_error)


def _relabelled(mesh, provenance):
    """The same mesh under another provenance, its arrays shared: a set
    derived from another hands on meshes it did not change, and relabelling
    them in place would rewrite what the first set says of its own."""
    copy = mesh.__class__.__new__(mesh.__class__)
    copy.__dict__.update(mesh.__dict__)
    copy.variables = dict(mesh.variables)
    copy.metadata = dict(mesh.metadata)
    copy.provenance = provenance
    return copy


def _box_body(corners):
    """The box through eight corners, numbered by bit (x first), as a body."""
    triangles = _np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
                           [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                           [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    corners = _np.asarray(corners, dtype=float)
    return Solid3D(corners, triangles, _gmt.vertex_normals(corners, triangles))


def _corners(data):
    """The eight corners of the ground a set can occupy, in world
    coordinates: a block model's lattice box, turned with the model, or a
    grid's own points -- its contours close between the outermost points
    and the padding past them."""
    bits = _np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1]
                      for i in range(8)], dtype=float)
    if isinstance(data, BlockSet3D):
        extent = _np.asarray(data.lattice_shape, dtype=float) \
            * _np.asarray(data.base_step, dtype=float)
        return data._to_world(_np.asarray(data.box_corner, dtype=float)
                              + bits * extent)
    low = _np.ravel(data.bounding_box.min)
    high = _np.ravel(data.bounding_box.max)
    return low + bits * (high - low)


# --------------------------------------------------------------------------- #
# the realizations, made in worker processes
# --------------------------------------------------------------------------- #
# What every worker reads, set before the pool forks so the children
# inherit it rather than receive it: the block model, the realizations of
# the group being contoured, the limits and the prediction's shells, all
# converted to Manifold bodies in the parent. Module-level because a pool
# can only call what it can import.
_BUILD = {}


def _realization_task(position):
    """One realization's whole set: contoured, cut, measured."""
    state = _BUILD
    number = int(state["numbers"][position])
    draws = state["values"][:, position, :]
    if state["kind"] == "category":
        fields = _category_fields(draws, state["rule"])
        columns = {key: fields[:, j] for j, key in enumerate(state["keys"])}
    else:
        columns = {key: draws[:, 0] for key in state["keys"]}

    meshes, raw, taken, failed, nudges = {}, {}, {}, {}, {}
    for key, level in zip(state["keys"], state["levels"]):
        # One realization's failure -- a shell Manifold refuses, a contour
        # that will not close -- must not throw away the rest of a build
        # that can run for an hour; it is recorded, and reported at the end
        try:
            shell, nudges[key] = _shell(
                state["data"], columns[key], level, state["side"],
                state["supersample"], state["label"])
            raw[key] = shell.volume
            shell, taken[key] = _limited(shell, state["bodies"],
                                         state["shift"])
            if state["simplify"] is not None:
                shell = _simplified(shell, state["simplify"])
            meshes[key] = shell
        except Exception as error:
            failed[key] = "%s: %s" % (type(error).__name__, error)
            meshes[key] = _empty_solid()
            raw[key] = _np.nan
            taken[key] = [_np.nan] * len(state["bodies"])
            nudges[key] = _np.nan

    if state["repair"]:
        if state["kind"] == "category":
            meshes, _ = _exclusive(meshes, state["keys"], state["shift"])
        else:
            meshes, _ = _nest(meshes, state["keys"], state["side"],
                              state["shift"])

    measures = {}
    for key in state["keys"]:
        found = _measure(meshes[key])
        found["raw"] = raw[key]
        found["nudge"] = nudges[key]
        if key in failed:
            found.update(volume=_np.nan, gained=_np.nan, lost=_np.nan)
        else:
            found["gained"], found["lost"] = _against(
                meshes[key], state["reference"][key], state["shift"])
        measures[key] = found
    arrays = {key: _arrays(meshes[key]) for key in state["keys"]}
    return number, arrays, measures, taken, failed


def _each_realization(count, workers):
    """`_realization_task` over positions `0..count-1`, in forked workers
    where there are several, in this process otherwise."""
    if workers > 1 and count > 1 \
            and "fork" in _mp.get_all_start_methods():
        with _warnings.catch_warnings():
            # the reasoning of `_DistanceQueries`: TensorFlow's thread
            # pools are up in the parent, the workers touch VTK, Manifold
            # and numpy alone, and spawned workers would import the package
            # once each
            _warnings.filterwarnings(
                "ignore", message=".*fork\\(\\)", category=DeprecationWarning)
            pool = _mp.get_context("fork").Pool(min(workers, count))
        try:
            yield from pool.imap_unordered(_realization_task, range(count))
            pool.close()
        finally:
            pool.terminate()
            pool.join()
    else:
        for position in range(count):
            yield _realization_task(position)


def _read_realizations(stores, numbers):
    """Realizations `numbers` of every store, `(n_rows, len(numbers),
    len(stores))`, read a band of rows at a time."""
    n_rows = int(stores[0].shape[0])
    values = _np.empty((n_rows, len(numbers), len(stores)))
    for s, store in enumerate(stores):
        for band in store.row_bands():
            values[band, :, s] = _np.asarray(store[band, :])[:, numbers]
    return values


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
def _arrays(mesh):
    """A mesh as the arrays that rebuild it, which a worker can hand back."""
    return (type(mesh).__name__,
            _np.asarray(mesh.coordinates, dtype=float),
            _np.asarray(mesh.triangles, dtype=_np.int64),
            _np.asarray(mesh.normals, dtype=float))


def _group(root, name):
    group = root
    for segment in name.split("/"):
        group = group.require_group(segment)
    return group


def _write_arrays(root, name, arrays, provenance):
    """One mesh into `root/name`, as a container `open` reads on its own."""
    group = _group(root, name)
    kind, points, triangles, normals = arrays
    for label, array in (("_coordinates", points), ("_triangles", triangles),
                         ("_normals", normals)):
        _storage.ArrayStore.from_numpy(array).write_into(group, label)
    meta = {"class": kind}
    if provenance:
        meta["provenance"] = _jsonable(provenance)
    group.attrs["geoml"] = {"geoml_format": _GEOML_ZARR_FORMAT,
                            "container": meta, "metadata": {},
                            "variables": {}}


def _write_mesh(root, name, mesh):
    group = _group(root, name)
    meta = _write_container(group, mesh)
    if "provenance" in meta:
        meta["provenance"] = _jsonable(meta["provenance"])
    group.attrs["geoml"] = {"geoml_format": _GEOML_ZARR_FORMAT,
                            "container": meta, "metadata": {},
                            "variables": {}}


def _read_mesh(path, name):
    group = _zarr.open_group(store=path, path=name, mode="r")
    meta = cast("dict[str, Any]", group.attrs["geoml"])
    return cast(Mesh3D, _rebuild_container(dict(meta["container"]), group))


def _jsonable(value):
    """A value the store's JSON can hold: NaN as None, arrays as lists."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, _np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, _np.bool_)):
        return bool(value)
    if isinstance(value, (_np.integer,)):
        return int(value)
    if isinstance(value, (float, _np.floating)):
        return float(value) if _np.isfinite(value) else None
    return value


def _floats(value):
    """A stored list back as a float array, None read as NaN."""
    array = _np.array(value, dtype=object)
    flat = [_np.nan if v is None else v for v in array.ravel()]
    return _np.array(flat, dtype=float).reshape(array.shape)


class _StoredMeshes(_abc.Mapping):
    """The meshes of one set in its store, each read when first asked for."""

    def __init__(self, path, prefix, keys):
        self._path, self._prefix, self._keys = path, prefix, list(keys)
        self._loaded = {}

    def __getitem__(self, key):
        if key not in self._loaded:
            self._loaded[key] = _read_mesh(
                self._path, "%s/%s" % (self._prefix, _path_key(key)))
        return self._loaded[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


class _Realizations(_abc.Sequence):
    """The realizations a set contoured, each a `MeshSet` of its own,
    read from the store when asked for and never held here."""

    def __init__(self, owner, numbers):
        self._owner, self._numbers = owner, list(numbers)

    @property
    def numbers(self) -> "list[int]":
        """Which of the variable's realizations these are."""
        return list(self._numbers)

    def __len__(self):
        return len(self._numbers)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _Realizations(self._owner, self._numbers[item])
        return self._owner._realization(self._numbers[item])

    def __repr__(self):
        return "%d realization%s of %s" % (
            len(self), "" if len(self) == 1 else "s", self._owner.path)


class _Bands(_abc.Mapping):
    """The bodies between consecutive cut-offs, keyed by `(low, high)` and
    worked out when first asked for."""

    def __init__(self, shells):
        self._shells = shells
        keys = shells._keys
        pairs = list(zip(keys[:-1], keys[1:]))
        if shells.close == "above":
            self._spans = pairs + [(keys[-1], _np.inf)]
        else:
            self._spans = [(-_np.inf, keys[0])] + pairs

    def __getitem__(self, span):
        try:
            low, high = (float(v) for v in span)
        except (TypeError, ValueError):
            raise KeyError(span)
        if (low, high) not in self._spans:
            raise KeyError(
                "no band %r; the bands are %s"
                % (span, ", ".join("(%g, %g)" % s for s in self._spans)))
        cache = self._shells._band_cache
        if (low, high) not in cache:
            cache[(low, high)] = self._shells._band(low, high)
        return cache[(low, high)]

    def __iter__(self):
        return iter(self._spans)

    def __len__(self):
        return len(self._spans)


def _band_label(low, high, unit=None):
    """A band's name: `< 0.3`, `0.3–0.5`, `≥ 0.5`."""
    suffix = "" if unit is None else " %s" % unit
    if not _np.isfinite(low):
        return "< %g%s" % (high, suffix)
    if not _np.isfinite(high):
        return "≥ %g%s" % (low, suffix)
    return "%g–%g%s" % (low, high, suffix)


def _dxf_layer(text):
    """A name DXF takes for a layer: none of `<>/\\":;?*|=,'`."""
    text = text.replace(">=", " ge ").replace("<=", " le ") \
        .replace("≥", " ge ").replace("–", " to ")
    for bad in '<>/\\":;?*|=,\'':
        text = text.replace(bad, "_")
    return " ".join(text.split()) or "mesh"


def _chosen(simulations, numbers):
    """Which of the realizations held to take: `True` all, `False` none,
    an int the first n, or a sequence of realization numbers."""
    if simulations is None or simulations is False:
        return []
    if simulations is True:
        return list(numbers)
    if isinstance(simulations, (int, _np.integer)):
        return list(numbers)[:int(simulations)]
    chosen = [int(n) for n in simulations]
    missing = [n for n in chosen if n not in numbers]
    if missing:
        raise ValueError("realizations %s were not contoured" % missing)
    return chosen


# --------------------------------------------------------------------------- #
# the set
# --------------------------------------------------------------------------- #
class MeshSet(_abc.Mapping):
    """
    Every contour of one column, at every cut-off, as one set.

    Contours a block model or a grid at each of a variable's cut-offs, or a
    categorical variable once per category, and holds the bodies as a
    read-only mapping: `shells[0.5]` is the body where the prediction
    clears 0.5, `shells["BIF"]` the body a category holds. The same is done
    for every realization the variable carries, each realization being a
    set of its own -- `shells.simulations[4][0.5]` -- whose meshes wait in
    a Zarr store until asked for.

    Every mesh is a closed `Solid3D`, cut to the limits the set was given:
    a sheet keeps what lies below it, a body what lies inside it, and the
    exclusions take theirs away. A terrain is extruded into the ground under
    it once for the whole set.

    In principle the shells nest -- the body above a higher cut-off lies
    inside the body above a lower one -- and categories do not overlap.
    `check` measures how far that fails, exactly, and `repair` enforces it.
    The volumes, bands and differences are worked out on the triangles by
    Manifold, in one frame for the whole set.

    Parameters
    ----------
    data
        The block model or grid carrying the column: a `BlockSet3D`, or a
        regular three-dimensional grid.
    path
        What to contour, named the way the tree names it: a variable or
        component, whose prediction is contoured (`"Comp/Fe"`), a column
        (`"Comp/Fe/latent_variance"`), or a categorical variable, for one
        body per category.
    cutoffs
        The levels to contour a continuous column at. The variable's own
        cut-offs when left out; a column that is not a prediction has none
        of its own and needs them here.
    close
        `"above"` keeps the ground where the values clear each cut-off, a
        grade shell; `"below"` the ground under it. A categorical set is
        always the ground each category holds.
    limits
        Sheets and bodies every mesh is cut to, by name: a sheet keeps what
        lies below it, a body what lies inside it.
    exclude
        Sheets and bodies taken away from every mesh, by name: what lies
        below a sheet, or a body's inside.
    simulations
        Which realizations to contour as well: `True` for every one the
        variable carries, `False` for none, an int for the first n, or a
        sequence of realization numbers. Each costs about what the
        prediction's own contours do, spread over `workers`.
    supersample
        How many levels past a block model's finest block each contour is
        cut to, as in `BlockSet3D.get_contour`.
    simplify
        A geometric error budget every mesh is simplified to after it is
        cut, in coordinate units, as in `Mesh3D.simplify`. Everything the
        set reports is measured on the meshes it holds.
    rule
        How a realization of a categorical variable decides which category
        holds a location: `"largest"`, the category with the largest draw,
        which is the rule of `likelihood.CategoricalGaussianIndicator`; or
        `"priority"`, the latest category in the variable's order whose draw
        is positive, which is the rule of
        `likelihood.HierarchicalGaussianIndicator`.
    repair
        Whether to make every set consistent as it is made -- each shell
        cut to the one outside it, each category's body cut away from the
        ones before it -- rather than only reporting what `check` finds.
    workers
        How many processes contour the realizations; 1 contours them in
        this one. Defaults to the number of CPUs, eight at most.
    store
        The path of a Zarr store to keep the set in, which `MeshSet.open`
        reads back. A temporary store, removed with the set, holds the
        realizations when left out.

    Attributes
    ----------
    data
        The container contoured; None for a set read back without one.
    path : str
        The column contoured, or the categorical variable.
    kind : str
        `"cutoff"`, keyed by cut-off, or `"category"`, keyed by name.
    close : str
        The side each body keeps, `"above"` or `"below"`.
    limits, excluded : dict
        The sheets and bodies the meshes were cut to and cut away from, as
        given.
    realization : int or None
        Which realization this set is; None for the prediction's.
    provenance : dict
        What the set was made from and how, which each mesh also carries.
    repairs : pandas.Series or None
        The volume taken from each mesh to make the set consistent, on a
        set that was.

    See Also
    --------
    BlockSet3D.get_contour : one contour, which a set makes many of.
    Solid3D : what every mesh of a set is.

    Examples
    --------
    .. code-block:: python

        shells = geoml.data.MeshSet(blocks, "Comp/FeO_total",
                                    cutoffs=[40, 50, 60],
                                    limits={"topography": topography})
        shells[50.0].volume
        shells.bands[(40.0, 50.0)]
        shells.simulations[4][50.0]
        shells.volume_dispersion()
    """

    data: "BlockSet3D | Grid3D | None"
    path: str
    kind: str
    close: str
    supersample: int
    rule: "str | None"
    limits: "dict[str, Mesh3D]"
    excluded: "dict[str, Mesh3D]"
    realization: "int | None"
    provenance: dict
    repairs: "_pd.Series | None"

    def __init__(self, data: "BlockSet3D | Grid3D", path: str,
                 cutoffs: "_types.Cutoffs | None" = None,
                 close: "bool | str" = "above",
                 limits: "dict[str, Mesh3D] | None" = None,
                 exclude: "dict[str, Mesh3D] | None" = None,
                 simulations: "bool | int | Sequence[int]" = True,
                 supersample: int = 0, simplify: "float | None" = None,
                 rule: str = "largest", repair: bool = False,
                 workers: "int | None" = None,
                 store: "_types.PathLike | None" = None) -> None:
        if rule not in ("largest", "priority"):
            raise ValueError(
                "rule is 'largest' or 'priority'; got %r" % (rule,))
        source = _source(data, path, cutoffs, close)
        stores = source["stores"]
        numbers = _numbers(simulations, None if stores is None
                           else int(stores[0].shape[1]))
        self._build(data, source, limits, exclude, supersample, simplify,
                    rule, repair)
        if numbers:
            self._contour_realizations(
                source, numbers, rule, repair,
                max(1, min(_WORKERS, _os.cpu_count() or 1))
                if workers is None else max(1, int(workers)),
                store)
        elif store is not None:
            self.to_zarr(store)

    # ------------------------------------------------------------------ #
    # building
    # ------------------------------------------------------------------ #
    def _build(self, data, source, limits, excluded, supersample, simplify,
               rule, repair, extra=None):
        """The prediction's set, from a resolved source."""
        self.data = data
        self.path = source["label"]
        self.kind = source["kind"]
        self.close = source["side"]
        self.supersample = int(supersample)
        self._simplify = None if simplify is None else float(simplify)
        self.rule = rule if self.kind == "category" else None
        self.limits = dict(limits or {})
        self.excluded = dict(excluded or {})
        both = sorted(set(self.limits) & set(self.excluded))
        if both:
            raise ValueError(
                "%s names both a limit and an exclusion" % both)
        self.realization = None
        self.repairs = None
        self._keys = list(source["keys"])
        self._levels = list(source["levels"])
        self._name = source["name"]
        self._unit = source["unit"]
        self._field = source["field"]
        self._corner_points = _corners(data)
        # the one frame every boolean of the set is worked out in: rounded,
        # so the shift costs nothing on the way back
        self._shift = _np.round(_np.min(self._corner_points, axis=0))
        self._numbers = []
        self._measures = None
        self._taken = None
        self._failures = []
        self._store = None
        self._finalizer = None
        self._parent = None
        self._band_cache = {}
        self.provenance = {
            "source": self.path, "kind": self.kind, "close": self.close,
            "supersample": self.supersample, "simplify": self._simplify,
            "limits": list(self.limits), "exclude": list(self.excluded)}
        if self.kind == "category":
            self.provenance["rule"] = rule
        self.provenance.update(extra or {})

        converted = [(_manifold(body, self._shift), keep)
                     for _, body, keep in self._bodies()]
        meshes, raw, taken, self._nudge = {}, {}, {}, {}
        for key, level in zip(self._keys, self._levels):
            shell, self._nudge[key] = _shell(
                data, source["fields"][key], level, self.close,
                self.supersample, self.path)
            raw[key] = shell.volume
            shell, taken[key] = _limited(shell, converted, self._shift)
            if self._simplify is not None:
                shell = _simplified(shell, self._simplify)
            meshes[key] = shell
        if repair:
            meshes, removed = self._consistent(meshes)
            self.repairs = _pd.Series(removed, name="removed").reindex(
                self._keys)
            self.provenance["repaired"] = True
        self._meshes = {key: _relabelled(mesh, self._mesh_provenance(key))
                        for key, mesh in meshes.items()}
        self._summary = self._summarized(self._meshes, raw, taken)

    def _bodies(self):
        """The limits and exclusions as bodies, in the order they cut."""
        return _prepared(self.limits, self.excluded,
                         BoundingBox.from_array(self._corner_points))

    def _consistent(self, meshes, priority=None):
        if self.kind == "category":
            order = list(self._keys) if priority is None else list(priority)
            if sorted(order) != sorted(self._keys):
                raise ValueError(
                    "priority must name every category once; this set holds "
                    "%s" % self._keys)
            return _exclusive(meshes, order, self._shift)
        if priority is not None:
            raise ValueError(
                "shells are nested by their cut-offs, and take no priority")
        return _nest(meshes, self._keys, self.close, self._shift)

    def _summarized(self, meshes, raw, taken):
        summary = {name: [] for name in ("volume", "raw", "pieces",
                                         "largest", "triangles")}
        for key in self._keys:
            found = _measure(meshes[key])
            for name in ("volume", "pieces", "largest", "triangles"):
                summary[name].append(found[name])
            summary["raw"].append(float(raw[key]))
        summary["taken"] = [list(map(float, taken[key])) for key in self._keys]
        return summary

    def _mesh_provenance(self, key):
        found = dict(self.provenance)
        found["key"] = key
        found["value"] = self._levels[self._keys.index(key)]
        found["nudge"] = float(self._nudge.get(key, 0.0))
        found["realization"] = self.realization
        return found

    def _contour_realizations(self, source, numbers, rule, repair, workers,
                              store):
        """Every realization's set, contoured in groups, measured, stored."""
        if store is None:
            directory = _tempfile.mkdtemp(prefix="geoml_meshset_")
            store = _os.path.join(directory, "meshes.zarr")
            self._finalizer = _weakref.finalize(
                self, _shutil.rmtree, directory, True)
        store = _os.fspath(store)
        root = _zarr.open_group(store, mode="w")
        self._store = store
        for key in self._keys:
            _write_mesh(root, "prediction/%s" % _path_key(key),
                        self._meshes[key])
        for group, meshes in (("limits", self.limits),
                              ("excluded", self.excluded)):
            for name, mesh in meshes.items():
                _write_mesh(root, "%s/%s" % (group, name), mesh)

        bodies = [(_manifold(body, self._shift), keep)
                  for _, body, keep in self._bodies()]
        reference = {key: _manifold(self._meshes[key], self._shift)
                     for key in self._keys}
        # evaluated here, before any fork: a Manifold body is a lazy tree,
        # and the children should only ever read the finished one
        for body in [b for b, _ in bodies] + list(reference.values()):
            if body is not None:
                body.num_tri()

        n_keys = len(self._keys)
        self._numbers = list(numbers)
        table = {name: _np.full((len(numbers), n_keys), _np.nan)
                 for name in _MEASURES}
        cuts = _np.full((len(bodies), len(numbers), n_keys), _np.nan)
        self._measures, self._taken = table, cuts
        self._failures = []
        where = {number: i for i, number in enumerate(numbers)}
        stores = source["stores"]
        per_realization = 8 * int(stores[0].shape[0]) * len(stores)
        size = max(1, int(_GROUP_BYTES // per_realization))
        for start in range(0, len(numbers), size):
            group = numbers[start:start + size]
            _BUILD.clear()
            _BUILD.update(
                data=self.data, values=_read_realizations(stores, group),
                numbers=group, kind=self.kind, keys=self._keys,
                levels=self._levels, side=self.close,
                supersample=self.supersample, label=self.path, rule=rule,
                bodies=bodies, shift=self._shift, simplify=self._simplify,
                repair=repair, reference=reference)
            try:
                for number, arrays, measures, taken, failed in \
                        _each_realization(len(group), workers):
                    row = where[number]
                    for j, key in enumerate(self._keys):
                        provenance = self._mesh_provenance(key)
                        provenance["realization"] = number
                        provenance["nudge"] = measures[key]["nudge"]
                        _write_arrays(root, "simulations/%d/%s"
                                      % (number, _path_key(key)),
                                      arrays[key], provenance)
                        for name in _MEASURES:
                            table[name][row, j] = \
                                measures[key][name]
                        if len(bodies):
                            cuts[:, row, j] = taken[key]
                    for key, message in failed.items():
                        self._failures.append(
                            {"realization": number, "key": key,
                             "error": message})
            finally:
                _BUILD.clear()
        root.attrs["geoml_meshset"] = self._attrs()
        if self._failures:
            _warnings.warn(
                "%d realization shell%s could not be made and read as "
                "missing; `failures` lists them"
                % (len(self._failures),
                   "" if len(self._failures) == 1 else "s"))

    # ------------------------------------------------------------------ #
    # the mapping
    # ------------------------------------------------------------------ #
    def _key(self, key):
        """`key` as the set holds it, or a KeyError listing what it holds."""
        if self.kind == "category":
            if isinstance(key, str) and key in self._keys:
                return key
        elif not isinstance(key, (str, bool)):
            try:
                number = float(key)
            except (TypeError, ValueError):
                number = None
            if number is not None:
                for held in self._keys:
                    if held == number:
                        return held
        raise KeyError(
            "%r is not in this set, which holds %s"
            % (key, ", ".join(("%g" % k) if self.kind == "cutoff" else k
                              for k in self._keys)))

    def __getitem__(self, key) -> Solid3D:
        return self._meshes[self._key(key)]

    def __iter__(self) -> Iterator:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key) -> bool:
        try:
            self._key(key)
        except KeyError:
            return False
        return True

    def __repr__(self) -> str:
        what = ("realization %d of " % self.realization
                if self.realization is not None else "")
        head = "MeshSet: %s%s, %s" % (
            what, self.path,
            "one body per category" if self.kind == "category"
            else "closed %s, %d cut-off%s" % (
                self.close, len(self), "" if len(self) == 1 else "s"))
        lines = [head]
        width = max(len(str(k)) for k in self._keys)
        for key, volume, pieces in zip(self._keys, self._summary["volume"],
                                       self._summary["pieces"]):
            lines.append("  %s  volume %.6g, %d piece%s" % (
                str(key).rjust(width), volume, pieces,
                "" if pieces == 1 else "s"))
        if self.limits or self.excluded:
            lines.append("  cut to %s" % ", ".join(
                list(self.limits) + ["not %s" % n for n in self.excluded]))
        if self._numbers:
            lines.append("  %d realization%s in %s" % (
                len(self._numbers), "" if len(self._numbers) == 1 else "s",
                self._store))
        return "\n".join(lines)

    @property
    def simulations(self) -> "_Realizations | None":
        """The realizations contoured, each a `MeshSet` of its own.

        A sequence in the order of the realization numbers: when every
        realization was contoured, position and number agree, and `numbers`
        says which they are when only some were. Each set is read from the
        store when asked for, and not kept here.
        """
        if not self._numbers:
            return None
        return _Realizations(self, self._numbers)

    @property
    def unit(self) -> "str | float | None":
        """What the contoured variable is measured in, where it says so."""
        return self._unit

    @property
    def failures(self) -> "list[dict]":
        """The realization shells that could not be made, and why."""
        return list(self._failures)

    @property
    def bands(self) -> _Bands:
        """The bodies between consecutive cut-offs, keyed by `(low, high)`.

        Above, the band from 0.3 to 0.5 is the shell at 0.3 less the shell
        at 0.5, and the last band is the top shell itself, `(0.5, inf)`;
        below, the bands run up from `(-inf, lowest)`. One band per cut-off,
        in the same order. Each is worked out when first asked for.
        """
        if self.kind == "category":
            raise TypeError(
                "a categorical set has no bands; its bodies are the bands")
        return _Bands(self)

    def _band(self, low, high):
        keep, cut = (low, high) if self.close == "above" else (high, low)
        body = self._meshes[keep]
        if not _np.isfinite(cut) or body.n_data == 0:
            return body
        outer = self._meshes[cut]
        if outer.n_data == 0:
            return body
        return _mesh(_difference(_manifold(body, self._shift),
                                 _manifold(outer, self._shift)), self._shift)

    def _realization(self, number):
        """A realization's set, read from the store."""
        position = self._numbers.index(number)
        view = MeshSet.__new__(MeshSet)
        view.__dict__.update({
            name: value for name, value in self.__dict__.items()
            if name not in ("_meshes", "_numbers", "_measures", "_taken",
                            "_failures", "_finalizer", "_band_cache",
                            "_summary", "repairs", "realization",
                            "provenance", "_parent", "_field")})
        view._meshes = _StoredMeshes(self._store, "simulations/%d" % number,
                                     self._keys)
        view._numbers = []
        view._measures = None
        view._taken = None
        view._failures = [f for f in self._failures
                          if f["realization"] == number]
        view._finalizer = None
        view._band_cache = {}
        # the view keeps the set alive, and with it a temporary store
        view._parent = self
        view._field = None
        view.repairs = None
        view.realization = int(number)
        view.provenance = dict(self.provenance, realization=int(number))
        measures = self._measured()
        view._nudge = dict(zip(self._keys, measures["nudge"][position]))
        taken = self._taken[:, position, :] if self._taken is not None \
            else _np.zeros((0, len(self._keys)))
        view._summary = {
            "volume": list(measures["volume"][position]),
            "raw": list(measures["raw"][position]),
            "pieces": [int(v) if _np.isfinite(v) else 0
                       for v in measures["pieces"][position]],
            "largest": list(measures["largest"][position]),
            "triangles": [int(v) if _np.isfinite(v) else 0
                          for v in measures["triangles"][position]],
            "taken": [list(map(float, taken[:, j]))
                      for j in range(len(self._keys))]}
        return view

    def _measured(self) -> "dict[str, _np.ndarray]":
        """What was measured on every realization, which only a set that
        contoured some holds."""
        if self._measures is None:
            raise ValueError(
                "no realizations were contoured for this set; build it with "
                "simulations=True from a variable that carries them")
        return self._measures

    # ------------------------------------------------------------------ #
    # reports
    # ------------------------------------------------------------------ #
    def table(self, density: "float | str | None" = None,
              grade: "str | None" = None) -> _pd.DataFrame:
        """
        What each mesh holds, one row per cut-off or per category.

        `volume` is the mesh as held, `raw_volume` the contour before any
        limit cut it, and one `removed: <name>` column per limit and
        exclusion says what each took, in the order they cut. `pieces` and
        `largest` count the bodies a mesh is in and the largest one's share.
        For a cut-off set, `band` and `band_volume` give the ground from
        each cut-off to the next on the side kept, and `crossing` how much
        of each shell lies outside the one around it; a categorical set
        gives each body's `overlap` with the others instead.
        `blocks_volume` adds up the blocks whose own value is on the kept
        side of the cut-off, or whose predicted category it is, for
        comparison with the raw contour.

        With a `density` or a `grade`, each band or body is also measured
        against the blocks: `tonnage`, the weighted `mean` of the grade and
        the `metal` it comes to, and -- where the grade carries realizations
        -- `metal_p10`, `metal_p50` and `metal_p90`, every realization's
        grade filling the same bands. A block a surface passes through
        counts the share of its sub-blocks inside.

        Parameters
        ----------
        density
            A number, the name of a metadata column, or the path of a
            variable whose prediction is the density; realizations of a
            density are paired one to one with the grade's.
        grade
            The variable or column to measure in each band or body, by
            path.

        Returns
        -------
        pandas.DataFrame
        """
        rows = {"volume": self._summary["volume"],
                "raw_volume": self._summary["raw"]}
        names = list(self.limits) + list(self.excluded)
        for i, name in enumerate(names):
            rows["removed: %s" % name] = [
                taken[i] if i < len(taken) else _np.nan
                for taken in self._summary["taken"]]
        rows["pieces"] = self._summary["pieces"]
        rows["largest"] = self._summary["largest"]
        blocks = self._blocks_volume()
        if blocks is not None:
            rows["blocks_volume"] = blocks

        checked = self.check()
        if self.kind == "cutoff":
            spans = list(self.bands)
            rows["band"] = [_band_label(low, high, self._unit)
                            for low, high in spans]
            bodies = [self.bands[span] for span in spans]
            rows["band_volume"] = [body.volume for body in bodies]
            crossing = dict(zip(checked["first"], checked["volume"]))
            rows["crossing"] = [crossing.get(key, _np.nan)
                                for key in self._keys]
        else:
            overlap = {key: 0.0 for key in self._keys}
            for kind, first, second, volume in zip(
                    checked["kind"], checked["first"], checked["second"],
                    checked["volume"]):
                if kind == "overlap":
                    overlap[first] += volume
                    overlap[second] += volume
            rows["overlap"] = [overlap[key] for key in self._keys]
            bodies = [self[key] for key in self._keys]

        if density is not None or grade is not None:
            rows.update(self._against_blocks(bodies, density, grade))
        return _pd.DataFrame(rows, index=_pd.Index(
            self._keys,
            name="category" if self.kind == "category" else "cutoff"))

    def _blocks_volume(self):
        """The blocks on the kept side of each cut-off, or holding each
        category, added up -- what the raw contour approximates."""
        if self.data is None or self.realization is not None:
            return None
        volume = _block_volumes(self.data)
        if self.kind == "category":
            node = self.data.get(VariablePath(self.path))
            if not isinstance(node, RockTypeVariable):
                return None
            codes = _np.asarray(node.predicted.values).ravel()
            labels = [str(label) for label in node.predicted.labels or []]
            return [float(_np.sum(volume[codes == labels.index(key)]))
                    if key in labels else _np.nan for key in self._keys]
        if self._field is None:
            return None
        field = self._field
        with _np.errstate(invalid="ignore"):
            return [float(_np.sum(volume[field >= key
                                         if self.close == "above"
                                         else field <= key]))
                    for key in self._keys]

    def _against_blocks(self, bodies, density, grade):
        """Tonnage, mean grade and metal in each body, from the blocks."""
        if self.data is None:
            raise ValueError(
                "this set was read back without its block model; open it "
                "with MeshSet.open(path, data=blocks) to measure against "
                "the blocks")
        data = self.data
        volume = _block_volumes(data)
        shares = _np.stack([_block_shares(data, body) for body in bodies],
                           axis=1)
        mass, mass_store = _densities(data, density)
        weights = volume[:, None] * shares * _np.where(
            _np.isfinite(mass), mass, 0.0)[:, None]
        found = {}
        if density is not None:
            found["tonnage"] = list(weights.sum(axis=0))
        if grade is None:
            return found
        path, column = _contour_column(data, grade)
        values = _np.asarray(column.values, dtype=float).ravel()
        finite = _np.isfinite(values)
        amount = _np.where(finite[:, None], weights, 0.0)
        metal = amount.T @ _np.where(finite, values, 0.0)
        with _np.errstate(invalid="ignore", divide="ignore"):
            found["mean"] = list(metal / amount.sum(axis=0))
        found["metal"] = list(metal)
        owner = data.get(path.parent) if path.name == "prediction" else None
        store = getattr(owner, "simulations", None)
        if store is not None:
            metals = _realized_metal(store, volume[:, None] * shares,
                                     mass if density is not None
                                     else _np.ones(data.n_data),
                                     mass_store)
            for q in (10, 50, 90):
                found["metal_p%d" % q] = list(
                    _np.nanpercentile(metals, q, axis=1))
        return found

    def check(self) -> _pd.DataFrame:
        """
        Where the set fails its own promise, measured exactly.

        For a cut-off set, one row per pair of consecutive shells: the
        volume of the inner shell lying outside the outer one, which nested
        shells would hold none of. For a categorical set, one row per pair
        of categories -- the volume both bodies claim -- and one for the
        gap, the ground within the model and its limits that no body holds.
        Part of a gap is always the model's own edges: a body closed
        against the box has its caps on the faces but rounds the edges
        where two faces meet by about half a boundary block, so bodies
        that fill the box between them still leave those strips.

        Returns
        -------
        pandas.DataFrame
            `kind` (`"crossing"`, `"overlap"` or `"gap"`), `first`,
            `second`, `volume`, and `share`: of the inner shell, of the
            smaller body, or of the ground.
        """
        rows = []
        if self.kind == "cutoff":
            order = list(self._keys) if self.close == "above" \
                else list(self._keys)[::-1]
            for outer, inner in zip(order[:-1], order[1:]):
                body = self._meshes[inner]
                crossing = 0.0 if body.n_data == 0 else _volume(_difference(
                    _manifold(body, self._shift),
                    _manifold(self._meshes[outer], self._shift)))
                own = body.volume if body.n_data else 0.0
                rows.append({"kind": "crossing", "first": inner,
                             "second": outer, "volume": crossing,
                             "share": crossing / own if own > 0 else 0.0})
        else:
            bodies = {key: _manifold(self._meshes[key], self._shift)
                      for key in self._keys}
            for first, second in _iter.combinations(self._keys, 2):
                a, b = bodies[first], bodies[second]
                shared = 0.0 if a is None or b is None else _volume(a ^ b)
                smaller = min(_volume(a), _volume(b))
                rows.append({"kind": "overlap", "first": first,
                             "second": second, "volume": shared,
                             "share": shared / smaller if smaller > 0
                             else 0.0})
            ground = self._ground()
            union = None
            for body in bodies.values():
                if body is not None:
                    union = body if union is None else union + body
            total = _volume(ground)
            covered = 0.0 if union is None or ground is None \
                else _volume(ground ^ union)
            gap = total - covered
            rows.append({"kind": "gap", "first": "(all)", "second": "",
                         "volume": gap,
                         "share": gap / total if total > 0 else 0.0})
        return _pd.DataFrame({name: [row[name] for row in rows]
                              for name in ("kind", "first", "second",
                                           "volume", "share")})

    def _ground(self):
        """The model's box cut by the limits, as a body in the set frame."""
        box = _box_body(self._corner_points)
        converted = [(_manifold(body, self._shift), keep)
                     for _, body, keep in self._bodies()]
        cut, _ = _limited(box, converted, self._shift)
        return _manifold(cut, self._shift)

    def volume_dispersion(self) -> _pd.DataFrame:
        """
        The realizations' mesh volumes against the prediction's.

        One row per cut-off or category. `prediction` is the volume of the
        prediction's mesh; `mean`, `sd`, `p10`, `p50` and `p90` are of the
        realizations'. `rank` is the share of realizations whose mesh is
        smaller than the prediction's, ties counted half: far from one half,
        the prediction's mesh is not a typical realization's, which is what
        a smooth field does at a cut-off in either tail. `gained` and `lost`
        are the mean volume a realization's mesh adds outside the
        prediction's and leaves out of it, as shares of the prediction's --
        large while the volumes agree means the same volume in different
        places. `pieces` and `pieces_p50` compare how fragmented they are.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If no realization was contoured.
        """
        measures = self._measured()
        volumes = measures["volume"]
        rows = []
        for j, key in enumerate(self._keys):
            predicted = self._summary["volume"][j]
            held = volumes[:, j]
            held = held[_np.isfinite(held)]
            below = _np.sum(held < predicted) + 0.5 * _np.sum(
                held == predicted)
            scale = predicted if predicted > 0 else _np.nan
            rows.append({
                "prediction": predicted,
                "mean": float(_np.mean(held)) if held.size else _np.nan,
                "sd": float(_np.std(held, ddof=1)) if held.size > 1
                else _np.nan,
                "p10": _quantile(held, 10), "p50": _quantile(held, 50),
                "p90": _quantile(held, 90),
                "rank": float(below / held.size) if held.size else _np.nan,
                "gained": _mean(measures["gained"][:, j]) / scale,
                "lost": _mean(measures["lost"][:, j]) / scale,
                "pieces": self._summary["pieces"][j],
                "pieces_p50": _quantile(measures["pieces"][:, j], 50)})
        return _pd.DataFrame(rows, index=_pd.Index(
            self._keys,
            name="category" if self.kind == "category" else "cutoff"))

    def realization_volumes(self) -> _pd.DataFrame:
        """
        Every realization's mesh volumes, as measured when it was made.

        Returns
        -------
        pandas.DataFrame
            One row per realization, one column per cut-off or category.
        """
        volumes = self._measured()["volume"]
        return _pd.DataFrame({key: volumes[:, j]
                              for j, key in enumerate(self._keys)},
                             index=_pd.Index(self._numbers,
                                             name="realization"))

    def realization_table(self, density: "float | str | None" = None,
                          grade: "str | None" = None,
                          simulations: "bool | int | Sequence[int]" = True
                          ) -> _pd.DataFrame:
        """
        What each realization's own meshes hold, measured against the
        blocks.

        `table` fills the prediction's bands with every realization's grade,
        which is what mining the prediction's shells would recover; this is
        the other spread, each realization's own bands filled with that
        realization's grade and density -- how much material there is. A
        block a band's surface passes through counts the share of its
        sub-blocks inside, so each band of each realization costs about what
        one band of `table(grade=)` does.

        Parameters
        ----------
        density
            A number, the name of a metadata column, or the path of a
            variable whose prediction is the density; a density with
            realizations is read realization by realization.
        grade
            The variable or column to measure in each band or body, by path;
            one with realizations is read realization by realization, one
            without is the same in all of them.
        simulations
            Which realizations: `True` for all, an int for the first n, or a
            sequence of realization numbers.

        Returns
        -------
        pandas.DataFrame
            One row per realization and band or body: `volume`, and with a
            density `tonnage`, with a grade `mean` and `metal`.
        """
        if self.data is None:
            raise ValueError(
                "this set was read back without its block model; open it "
                "with MeshSet.open(path, data=blocks) to measure against "
                "the blocks")
        numbers = _chosen(simulations, self._numbers)
        if not numbers:
            raise ValueError("no realizations were contoured for this set")
        data = self.data
        volume = _block_volumes(data)
        mass, mass_store = _densities(data, density)
        masses = None
        if mass_store is not None and int(mass_store.shape[1]) > max(numbers):
            masses = _read_realizations([mass_store], numbers)[:, :, 0]
        grades = values = None
        if grade is not None:
            path, column = _contour_column(data, grade)
            values = _np.asarray(column.values, dtype=float).ravel()
            owner = data.get(path.parent) if path.name == "prediction" \
                else None
            store = getattr(owner, "simulations", None)
            if store is not None and int(store.shape[1]) > max(numbers):
                grades = _read_realizations([store], numbers)[:, :, 0]

        rows, index = [], []
        for position, number in enumerate(numbers):
            view = self._realization(number)
            if self.kind == "category":
                labels = list(self._keys)
                bodies = [view[key] for key in self._keys]
            else:
                labels = list(self._keys)
                bodies = [view.bands[span] for span in view.bands]
            here = mass if masses is None else masses[:, position]
            here = _np.where(_np.isfinite(here), here, 0.0)
            grade_here = values if grades is None else grades[:, position]
            for label, body in zip(labels, bodies):
                share = _block_shares(data, body)
                row = {"volume": body.volume}
                weight = volume * share * here
                if density is not None:
                    row["tonnage"] = float(weight.sum())
                if grade_here is not None:
                    finite = _np.isfinite(grade_here)
                    amount = _np.where(finite, weight, 0.0)
                    metal = float(_np.sum(amount * _np.where(
                        finite, grade_here, 0.0)))
                    row["mean"] = metal / float(amount.sum()) \
                        if amount.sum() > 0 else _np.nan
                    row["metal"] = metal
                rows.append(row)
                index.append((number, label))
        return _pd.DataFrame(rows, index=_pd.MultiIndex.from_tuples(
            index, names=["realization", "category" if self.kind ==
                          "category" else "cutoff"]))

    def connectivity(self) -> _pd.DataFrame:
        """
        How many pieces each mesh is in, and the largest one's share.

        Read along the cut-offs, the largest piece's share is a connectivity
        curve: where it drops, the ground above the cut-off breaks into
        pods. With realizations, `largest_p10` to `largest_p90` and
        `pieces_p50` say the same of theirs.

        Returns
        -------
        pandas.DataFrame
        """
        frame = _pd.DataFrame(
            {"pieces": self._summary["pieces"],
             "largest": self._summary["largest"]},
            index=_pd.Index(self._keys, name="category"
                            if self.kind == "category" else "cutoff"))
        if self._measures is not None:
            measures = self._measures
            largest = measures["largest"]
            for q in (10, 50, 90):
                frame["largest_p%d" % q] = [_quantile(largest[:, j], q)
                                            for j in range(len(self._keys))]
            frame["pieces_p50"] = [
                _quantile(measures["pieces"][:, j], 50)
                for j in range(len(self._keys))]
        return frame

    def spacing(self) -> _pd.DataFrame:
        """
        How far apart consecutive shells sit.

        For each pair, the distance from every vertex of the inner shell to
        the surface of the outer one: tight means the grade climbs fast, a
        sharp contact; wide, a gradational one.

        Returns
        -------
        pandas.DataFrame
            One row per pair: `inner`, `outer`, and the distances' `min`,
            `p10`, `p50`, `p90` and `max`.
        """
        if self.kind == "category":
            raise TypeError(
                "categories are not nested, so there is no inner and outer "
                "to measure between")
        order = list(self._keys) if self.close == "above" \
            else list(self._keys)[::-1]
        rows = []
        for outer, inner in zip(order[:-1], order[1:]):
            distances = _distances(self._meshes[inner], self._meshes[outer],
                                   self._shift)
            rows.append({"inner": inner, "outer": outer,
                         **_spread(distances)})
        return _pd.DataFrame(rows)

    def compare(self, other: "MeshSet") -> _pd.DataFrame:
        """
        How another set's meshes differ from these, key by key.

        `gained` is the volume of the other's mesh lying outside this one's,
        `lost` the volume of this one's the other leaves out -- exact, both
        ways -- and `moved_p50`, `moved_p90` and `moved_max` how far the
        other's vertices sit from this one's surface. Two models of the same
        ground, or one model before and after a drilling campaign.

        Parameters
        ----------
        other
            A set with keys in common with this one.

        Returns
        -------
        pandas.DataFrame
            One row per key the two have in common.
        """
        common = [key for key in self._keys if key in other]
        if not common:
            raise ValueError("the two sets have no key in common")
        rows = []
        for key in common:
            mine, theirs = self[key], other[key]
            gained, lost = _against(theirs, _manifold(mine, self._shift),
                                    self._shift)
            spread = _spread(_distances(theirs, mine, self._shift))
            rows.append({"volume": mine.volume,
                         "other_volume": theirs.volume,
                         "gained": gained, "lost": lost,
                         "moved_p50": spread["p50"],
                         "moved_p90": spread["p90"],
                         "moved_max": spread["max"]})
        return _pd.DataFrame(rows, index=_pd.Index(
            common, name="category" if self.kind == "category"
            else "cutoff"))

    def section(self, axis: "int | str",
                value: float) -> "dict[Any, list[_np.ndarray]]":
        """
        Where each mesh crosses a plane across one axis.

        Parameters
        ----------
        axis
            The coordinate held fixed, by index or by label (`"X"`).
        value
            Where along it the plane sits.

        Returns
        -------
        dict
            For every key, a list of polylines, each an `(n, 3)` array of
            points in order; a closed line repeats its first point last.
        """
        index = _axis_index(self.data, axis)
        normal = _np.zeros(3)
        normal[index] = 1.0
        origin = self._shift.copy()
        origin[index] = float(value)
        lines = {}
        for key in self._keys:
            mesh = self._meshes[key]
            lines[key] = [] if mesh.n_data == 0 \
                else _cut(mesh, normal, origin, self._shift)
        return lines

    # ------------------------------------------------------------------ #
    # new sets
    # ------------------------------------------------------------------ #
    def _derived(self, meshes, limits=None, excluded=None, taken=None,
                 **provenance):
        """A new prediction's set holding `meshes`, sharing everything
        else, without realizations."""
        new = MeshSet.__new__(MeshSet)
        new.__dict__.update({
            name: value for name, value in self.__dict__.items()
            if name not in ("_meshes", "_numbers", "_measures", "_taken",
                            "_failures", "_finalizer", "_band_cache",
                            "_summary", "repairs", "_store", "_parent",
                            "provenance", "limits", "excluded")})
        new.limits = dict(self.limits if limits is None else limits)
        new.excluded = dict(self.excluded if excluded is None else excluded)
        new._numbers, new._measures, new._taken = [], None, None
        new._failures, new._finalizer, new._store = [], None, None
        new._band_cache = {}
        new._parent = self._parent
        new.repairs = None
        new.provenance = dict(self.provenance)
        new.provenance.update(limits=list(new.limits),
                              exclude=list(new.excluded), **provenance)
        new._meshes = {key: _relabelled(mesh, new._mesh_provenance(key))
                       for key, mesh in meshes.items()}
        raw = dict(zip(self._keys, self._summary["raw"]))
        if taken is None:
            taken = dict(zip(self._keys, self._summary["taken"]))
        new._summary = new._summarized(new._meshes, raw, taken)
        return new

    def repair(self, priority: "Sequence[str] | None" = None) -> "MeshSet":
        """
        The set made consistent, and what that took.

        Each shell is cut to the one outside it, so the shells nest; or
        each category's body is cut away from the ones before it in
        `priority`, so no ground is claimed twice. What each mesh lost is
        in `repairs` on the set returned. A gap is not filled -- that would
        be inventing ground.

        Parameters
        ----------
        priority
            For a categorical set, the order in which categories keep their
            ground, first first. The set's own order when left out.

        Returns
        -------
        MeshSet
            The prediction's meshes, repaired; realizations are not carried.
        """
        meshes, removed = self._consistent(dict(self.items()), priority)
        new = self._derived(meshes, repaired=True)
        new.repairs = _pd.Series(removed, name="removed").reindex(self._keys)
        return new

    def clip(self, mesh: Mesh3D, name: "str | None" = None) -> "MeshSet":
        """
        Every mesh cut to one more limit: what lies below a sheet, or
        inside a body.

        Parameters
        ----------
        mesh
            The sheet or body.
        name
            What to call it among the limits.

        Returns
        -------
        MeshSet
            The prediction's meshes, cut; realizations are not carried --
            build the set with the limit to have them cut too.
        """
        return self._cut(mesh, name, keep=True)

    def exclude(self, mesh: Mesh3D, name: "str | None" = None) -> "MeshSet":
        """
        Every mesh with one more piece taken away: what lies below a sheet,
        or inside a body.

        Parameters
        ----------
        mesh
            The sheet or body.
        name
            What to call it among the exclusions.

        Returns
        -------
        MeshSet
            The prediction's meshes, cut; realizations are not carried.
        """
        return self._cut(mesh, name, keep=False)

    def _cut(self, mesh, name, keep):
        limits, excluded = dict(self.limits), dict(self.excluded)
        target = limits if keep else excluded
        if name is None:
            name = "%s %d" % ("limit" if keep else "exclusion",
                              len(target) + 1)
        if name in limits or name in excluded:
            raise ValueError("%r already names a limit or an exclusion"
                             % name)
        target[name] = mesh
        ((_, body, _),) = _prepared({name: mesh} if keep else {},
                                    {} if keep else {name: mesh},
                                    BoundingBox.from_array(
                                        self._corner_points))
        converted = [(_manifold(body, self._shift), keep)]
        meshes, taken = {}, {}
        for key, previous in zip(self._keys, self._summary["taken"]):
            meshes[key], cut = _limited(self._meshes[key], converted,
                                        self._shift)
            # the new cut is reported where its kind goes: the limits
            # first, then the exclusions
            before = list(previous)
            before.insert(len(self.limits) if keep else len(before), cut[0])
            taken[key] = before
        return self._derived(meshes, limits=limits, excluded=excluded,
                             taken=taken)

    def simplify(self, max_error: float) -> "MeshSet":
        """
        Every mesh on as few triangles as `max_error` allows, still nested.

        Each mesh is simplified on its own (`Mesh3D.simplify`), which can
        move two shells closer than twice the budget across each other, so
        a cut-off set is nested again afterwards, each shell cut to the one
        outside it; `repairs` says what that took.

        Parameters
        ----------
        max_error
            The largest distance a simplified surface may sit from its
            original, in coordinate units.

        Returns
        -------
        MeshSet
            The prediction's meshes, simplified; realizations are not
            carried.
        """
        meshes = {key: _simplified(mesh, max_error)
                  for key, mesh in self.items()}
        removed = None
        if self.kind == "cutoff":
            meshes, removed = _nest(meshes, self._keys, self.close,
                                    self._shift)
        new = self._derived(meshes, simplify=float(max_error))
        new._simplify = float(max_error)
        if removed is not None:
            new.repairs = _pd.Series(removed, name="removed").reindex(
                self._keys)
        return new

    def drop_pieces(self, min_volume: float) -> "MeshSet":
        """
        Every mesh without the pieces smaller than `min_volume`.

        For pods below a mining unit, say. A cut-off set is nested again
        afterwards: dropping a piece of an outer shell takes whatever of the
        inner shells it held.

        Parameters
        ----------
        min_volume
            The smallest body to keep, in cubic coordinate units.

        Returns
        -------
        MeshSet
            The prediction's meshes, without the small pieces;
            realizations are not carried.
        """
        meshes = {}
        for key, mesh in self.items():
            if mesh.n_data == 0:
                meshes[key] = mesh
                continue
            pieces = mesh.split()
            kept = [piece for piece in pieces
                    if isinstance(piece, Solid3D)
                    and piece.volume >= min_volume]
            if len(kept) == len(pieces):
                meshes[key] = mesh
            else:
                meshes[key] = _joined(kept) if kept else _empty_solid()
        if self.kind == "cutoff":
            meshes, _ = _nest(meshes, self._keys, self.close, self._shift)
        return self._derived(meshes, min_volume=float(min_volume))

    @classmethod
    def probability(cls, data: "BlockSet3D | Grid3D", path: str,
                    cutoff: float, levels: Sequence[float] = (0.1, 0.5, 0.9),
                    side: str = "above",
                    limits: "dict[str, Mesh3D] | None" = None,
                    exclude: "dict[str, Mesh3D] | None" = None,
                    supersample: int = 0,
                    simplify: "float | None" = None) -> "MeshSet":
        """
        The bodies where a variable clears a cut-off with given probability.

        For each level, the ground where at least that share of the
        realizations clears `cutoff` on `side`: at 0.9 the ground the model
        is sure of, at 0.1 the ground it cannot rule out. Worked out from
        the realizations a band of blocks at a time, and never written to
        the container. The set is keyed by the levels, and nests like any
        cut-off set.

        Parameters
        ----------
        data
            The block model or grid.
        path
            The variable or component, by path.
        cutoff
            The cut-off the probability is of.
        levels
            The probabilities to contour at.
        side
            `"above"`, the probability of exceeding the cut-off, or
            `"below"`, of falling under it.
        limits, exclude, supersample, simplify
            As for `MeshSet`.

        Returns
        -------
        MeshSet
        """
        column_path, _ = _contour_column(data, path)
        owner = data.get(column_path.parent) \
            if column_path.name == "prediction" else None
        store = getattr(owner, "simulations", None)
        if store is None:
            raise ValueError(
                "%r carries no realizations to take a probability from"
                % str(column_path))
        kept = _kept_side(side)
        field = _exceedance(store, float(cutoff), kept)
        keys = sorted(set(float(level) for level in levels))
        sign = ">" if kept == "above" else "<"
        source = {"kind": "cutoff",
                  "label": "P(%s %s %g)" % (column_path, sign, float(cutoff)),
                  "keys": keys, "levels": list(keys),
                  "fields": {key: field for key in keys}, "field": field,
                  "stores": None, "side": "above",
                  "name": "P(%s %s %g)" % (getattr(owner, "name", column_path),
                                           sign, float(cutoff)),
                  "unit": None}
        new = cls.__new__(cls)
        new._build(data, source, limits, exclude, supersample, simplify,
                   None, False,
                   extra={"probability_of": str(column_path),
                          "cutoff": float(cutoff), "side": kept})
        return new

    # ------------------------------------------------------------------ #
    # back to the data
    # ------------------------------------------------------------------ #
    def assign(self, container: PointData, name: str,
               fraction: "str | None" = None) -> None:
        """
        Writes which band or body each location falls in, as metadata.

        A coded column named `name`: for a cut-off set the band, counted by
        how many shells hold the location, labelled `< 0.3`, `0.3–0.5`,
        `≥ 0.5`; for a categorical set the first body in the set's order
        holding it, missing where none does. Grade-shell domaining for
        drillholes, say.

        Parameters
        ----------
        container
            Anything with coordinates: points, drillhole composites, a grid,
            blocks.
        name
            The metadata column to write.
        fraction
            For blocks, a prefix: one more column per band or body, named
            `"<fraction> <label>"`, holding the share of each block's
            sub-blocks inside it.
        """
        coordinates = _np.asarray(container.coordinates, dtype=float)
        if self.kind == "category":
            codes = _np.full(len(coordinates), -1, dtype=int)
            for i, key in enumerate(self._keys):
                mesh = self._meshes[key]
                if mesh.n_data == 0:
                    continue
                inside = _within_body(mesh)(coordinates)
                codes[(codes < 0) & inside] = i
            labels = list(self._keys)
            bodies = [self._meshes[key] for key in self._keys]
        else:
            held = _np.zeros(len(coordinates), dtype=int)
            for key in self._keys:
                mesh = self._meshes[key]
                if mesh.n_data:
                    held += _within_body(mesh)(coordinates)
            spans = list(self.bands)
            bands = [_band_label(low, high, self._unit)
                     for low, high in spans]
            first, last = self._keys[0], self._keys[-1]
            if self.close == "above":
                # nothing holds the ground below the lowest cut-off, which
                # is a band of the table but not of the set
                labels = [_band_label(-_np.inf, first, self._unit)] + bands
                codes = held
            else:
                labels = bands + [_band_label(last, _np.inf, self._unit)]
                codes = len(self._keys) - held
            bodies = [self.bands[span] for span in spans]
        container.add_metadata(name, codes, labels=labels)
        if fraction is not None:
            if not hasattr(container, "rows_per_location"):
                raise TypeError(
                    "a fraction is a share of a block's sub-blocks, and %s "
                    "has no blocks" % type(container).__name__)
            named = labels if self.kind == "category" else bands
            for label, body in zip(named, bodies):
                container.add_metadata("%s %s" % (fraction, label),
                                       _block_shares(container, body))

    def crossed_by(self, blocks: BlockSet3D) -> _np.ndarray:
        """
        Which blocks any mesh of the set passes through.

        `BlockSet3D.crossed_by` asked of every mesh at once, to refine a
        model against the whole set:
        `blocks.split(shells.crossed_by(blocks))`.

        Returns
        -------
        array
            One boolean per block.
        """
        crossed = _np.zeros(blocks.n_data, dtype=bool)
        for mesh in self.values():
            if mesh.n_data:
                crossed |= blocks.crossed_by(mesh)
        return crossed

    # ------------------------------------------------------------------ #
    # exports
    # ------------------------------------------------------------------ #
    def _object_name(self, key):
        """What a mesh is called outside the package."""
        if self.kind == "category":
            return str(key)
        if "probability_of" in self.provenance:
            return "%s >= %g" % (self._name, key)
        unit = "" if self._unit is None else " %s" % self._unit
        return "%s %s %g%s" % (self._name,
                               ">=" if self.close == "above" else "<=",
                               key, unit)

    def to_geoh5(self, workspace: "_types.PathLike | _GeoH5Workspace",
                 folder: "str | None" = None, replace: bool = True,
                 simulations: "bool | int | Sequence[int]" = False) -> None:
        """
        Writes every mesh into a geoh5 workspace, one Surface each.

        Named after the variable and the cut-off, with its unit where the
        variable declares one, or after the category, each carrying its
        provenance. The limits and exclusions go in a `limits` folder beside
        them, and each realization asked for in `simulations/<n>`. Empty
        meshes are left out. Needs `geoh5py`: `pip install geoml[geoh5]`.

        Parameters
        ----------
        workspace
            The path of the workspace, or an open
            `geoml.data.geoh5.Workspace`.
        folder
            Where the set goes in ANALYST's project tree, `"Shells/Fe"`.
        replace
            Whether objects of the same names in the same folders make way.
        simulations
            Which realizations to write as well: `True` for all, an int for
            the first n, or a sequence of realization numbers.
        """
        import geoml.data.geoh5 as _geoh5io
        own = isinstance(workspace, (str, _os.PathLike))
        wrapper = _geoh5io.Workspace(workspace) if own else workspace

        def inside(sub):
            return sub if folder is None else "%s/%s" % (folder, sub)

        try:
            self._write_geoh5(wrapper, folder, replace)
            for name, mesh in list(self.limits.items()) \
                    + list(self.excluded.items()):
                if mesh.n_data:
                    mesh.to_geoh5(wrapper, name=name, replace=replace,
                                  folder=inside("limits"))
            for number in _chosen(simulations, self._numbers):
                self._realization(number)._write_geoh5(
                    wrapper, inside("simulations/%d" % number), replace)
        finally:
            if own:
                wrapper.close()

    def _write_geoh5(self, wrapper, folder, replace):
        for key in self._keys:
            mesh = self[key]
            if mesh.n_data:
                mesh.to_geoh5(wrapper, name=self._object_name(key),
                              replace=replace, folder=folder)

    def export_dxf(self, filename: _types.PathLike,
                   offset: "_types.ArrayLike | None" = None) -> None:
        """
        Writes every mesh into one DXF file, a layer each.

        Each mesh is a `MESH` entity on a layer named after the variable
        and the cut-off, or after the category, coloured in turn. Only the
        geometry travels, as in `Mesh3D.export_dxf`.

        Parameters
        ----------
        filename
            The file to write.
        offset
            Added to every coordinate on the way out, and not recorded.
        """
        document = _new_dxf()
        space = document.modelspace()
        shift = _np.zeros(3) if offset is None \
            else _np.asarray(offset, dtype=float).reshape(3)
        for i, key in enumerate(self._keys):
            mesh = self[key]
            if mesh.n_data == 0:
                continue
            layer = _dxf_layer(self._object_name(key))
            if not document.layers.has_entry(layer):
                document.layers.add(layer, color=_ACI[i % len(_ACI)])
            entity = space.add_mesh(dxfattribs={"layer": layer})
            with entity.edit_data() as edited:
                edited.vertices = (_np.asarray(mesh.coordinates, dtype=float)
                                   + shift).tolist()
                edited.faces = _np.asarray(mesh.triangles,
                                           dtype=int).tolist()
        document.saveas(_os.fspath(filename))

    def as_pyvista(self) -> "_pv.MultiBlock":
        """Every mesh as one pyvista `MultiBlock`, named as `to_geoh5`
        names them."""
        blocks = _pv.MultiBlock()
        for key in self._keys:
            mesh = self[key]
            blocks[self._object_name(key)] = mesh._polydata() \
                if mesh.n_data else _pv.PolyData()
        return blocks

    def plot(self, plotter: "_pv.Plotter | None" = None,
             opacity: float = 0.35, colors: "Sequence[str] | None" = None,
             show_limits: bool = False, **kwargs) -> "_pv.Plotter":
        """
        Adds every mesh to a pyvista scene, translucent, a colour each.

        Parameters
        ----------
        plotter
            The scene to add to; a new `pyvista.Plotter` when left out.
        opacity
            How opaque each mesh is: nested shells need to be seen through.
        colors
            One colour per mesh, in order. The package palette by default.
        show_limits
            Whether to draw the limits and exclusions too, as wireframes.
        **kwargs
            Passed to `add_mesh`.

        Returns
        -------
        pyvista.Plotter
        """
        # late: the palette is the plots', and the data layer loads no
        # plotting code until a scene is asked for
        import geoml.plots.style as _style
        scene = _pv.Plotter() if plotter is None else plotter
        for i, key in enumerate(self._keys):
            mesh = self[key]
            if mesh.n_data == 0:
                continue
            color = colors[i % len(colors)] if colors else _style.color(i)
            scene.add_mesh(mesh._polydata(), color=color, opacity=opacity,
                           label=self._object_name(key), **kwargs)
        if show_limits:
            for mesh in list(self.limits.values()) \
                    + list(self.excluded.values()):
                if mesh.n_data:
                    scene.add_mesh(mesh._polydata(), style="wireframe",
                                   color="#7f7f7f", opacity=0.5)
        return scene

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def _attrs(self):
        unit = self._unit if isinstance(self._unit, (str, int, float,
                                                     type(None))) \
            else str(self._unit)
        return _jsonable({
            "format": _STORE_FORMAT, "kind": self.kind, "path": self.path,
            "keys": self._keys, "levels": self._levels, "close": self.close,
            "supersample": self.supersample, "simplify": self._simplify,
            "rule": self.rule, "limits": list(self.limits),
            "excluded": list(self.excluded), "name": self._name,
            "unit": unit, "provenance": self.provenance,
            "realization": self.realization,
            "corners": self._corner_points, "shift": self._shift,
            "summary": self._summary, "numbers": self._numbers,
            "nudge": [self._nudge.get(key, 0.0) for key in self._keys],
            "measures": self._measures, "taken": self._taken,
            "failures": self._failures,
            "repairs": None if self.repairs is None
            else {"keys": list(self.repairs.index),
                  "values": list(self.repairs.values)}})

    def to_zarr(self, path: _types.PathLike) -> _types.PathLike:
        """
        Writes the set into a Zarr store, which `MeshSet.open` reads back.

        Every mesh goes in a group of its own -- `prediction/0.5`,
        `simulations/4/0.5`, `limits/topography` -- each a container that
        `Solid3D.open` also reads on its own, with what the set measured
        beside them. The realizations are copied from the set's store.

        Parameters
        ----------
        path
            Where to write the store.

        Returns
        -------
        The path written.
        """
        target = _os.fspath(path)
        if self._store is not None and self.realization is None \
                and _os.path.abspath(target) \
                == _os.path.abspath(self._store):
            root = _zarr.open_group(target, mode="r+")
            root.attrs["geoml_meshset"] = self._attrs()
            return path
        root = _zarr.open_group(target, mode="w")
        for key in self._keys:
            _write_mesh(root, "prediction/%s" % _path_key(key), self[key])
        for group, meshes in (("limits", self.limits),
                              ("excluded", self.excluded)):
            for name, mesh in meshes.items():
                _write_mesh(root, "%s/%s" % (group, name), mesh)
        if self._numbers and self._store is not None:
            _shutil.copytree(_os.path.join(self._store, "simulations"),
                             _os.path.join(target, "simulations"))
        root.attrs["geoml_meshset"] = self._attrs()
        return path

    @classmethod
    def open(cls, path: _types.PathLike,
             data: "BlockSet3D | Grid3D | None" = None) -> "MeshSet":
        """
        A set written by `to_zarr`, its meshes read when asked for.

        Parameters
        ----------
        path
            The store.
        data
            The block model the set was contoured from, for the reports
            that measure against the blocks; they are left out without it.

        Returns
        -------
        MeshSet
        """
        path = _os.fspath(path)
        root = _zarr.open_group(path, mode="r")
        meta = dict(cast("dict[str, Any]", root.attrs["geoml_meshset"]))
        if meta.get("format") != _STORE_FORMAT:
            raise ValueError(
                "%r was written at mesh set format %s and this version reads "
                "%d" % (path, meta.get("format"), _STORE_FORMAT))
        new = cls.__new__(cls)
        kind = meta["kind"]
        keys = [str(k) for k in meta["keys"]] if kind == "category" \
            else [float(k) for k in meta["keys"]]
        new.data = data
        new.path = meta["path"]
        new.kind = kind
        new.close = meta["close"]
        new.supersample = int(meta["supersample"])
        new._simplify = meta["simplify"]
        new.rule = meta["rule"]
        new.limits = {name: _read_mesh(path, "limits/%s" % name)
                      for name in meta["limits"]}
        new.excluded = {name: _read_mesh(path, "excluded/%s" % name)
                        for name in meta["excluded"]}
        new.realization = meta["realization"]
        new.provenance = dict(meta["provenance"])
        new._keys = keys
        new._levels = [float(v) for v in meta["levels"]]
        new._name = meta["name"]
        new._unit = meta["unit"]
        new._field = None
        if data is not None and kind == "cutoff" \
                and "probability_of" not in new.provenance:
            _, column = _contour_column(data, new.path)
            new._field = _np.asarray(column.values, dtype=float).ravel()
        new._corner_points = _floats(meta["corners"])
        new._shift = _floats(meta["shift"])
        summary = meta["summary"]
        new._summary = {
            "volume": list(_floats(summary["volume"])),
            "raw": list(_floats(summary["raw"])),
            "pieces": [int(v) for v in summary["pieces"]],
            "largest": list(_floats(summary["largest"])),
            "triangles": [int(v) for v in summary["triangles"]],
            "taken": [list(_floats(t)) for t in summary["taken"]]}
        new._nudge = dict(zip(keys, _floats(meta["nudge"])))
        new._numbers = [int(n) for n in meta["numbers"]]
        new._measures = None if meta["measures"] is None else {
            name: _floats(values) for name, values in meta["measures"].items()}
        # an empty list keeps no shape, so the cuts are reshaped by what
        # they count: every limit and exclusion, realization and key
        new._taken = None if meta["taken"] is None \
            else _floats(meta["taken"]).reshape(
                len(new.limits) + len(new.excluded), len(new._numbers),
                len(keys))
        new._failures = list(meta["failures"])
        new._store = path
        new._finalizer = None
        new._parent = None
        new._band_cache = {}
        new._meshes = _StoredMeshes(path, "prediction", keys)
        new.repairs = None
        if meta["repairs"] is not None:
            new.repairs = _pd.Series(
                _floats(meta["repairs"]["values"]),
                index=meta["repairs"]["keys"], name="removed")
        return new


# --------------------------------------------------------------------------- #
# the arithmetic behind the reports
# --------------------------------------------------------------------------- #
def _mean(values):
    values = _np.asarray(values, dtype=float)
    values = values[_np.isfinite(values)]
    return float(values.mean()) if values.size else _np.nan


def _quantile(values, q):
    values = _np.asarray(values, dtype=float)
    values = values[_np.isfinite(values)]
    return float(_np.percentile(values, q)) if values.size else _np.nan


def _spread(distances):
    if distances.size == 0:
        return {"min": _np.nan, "p10": _np.nan, "p50": _np.nan,
                "p90": _np.nan, "max": _np.nan}
    return {"min": float(distances.min()),
            "p10": float(_np.percentile(distances, 10)),
            "p50": float(_np.percentile(distances, 50)),
            "p90": float(_np.percentile(distances, 90)),
            "max": float(distances.max())}


def _distances(mesh, surface, shift):
    """How far each vertex of `mesh` sits from the surface of `surface`."""
    if mesh.n_data == 0 or surface.n_data == 0:
        return _np.zeros(0)
    points = _np.asarray(mesh.coordinates, dtype=float) - shift
    with _DistanceQueries(surface._polydata().translate(-shift)) as measure:
        return _np.abs(measure.query(points))


def _axis_index(data, axis):
    if isinstance(axis, (int, _np.integer)):
        if not 0 <= int(axis) < 3:
            raise ValueError("axis is 0, 1 or 2; got %d" % axis)
        return int(axis)
    labels = [str(label) for label in
              (getattr(data, "coordinate_labels", None) or ("X", "Y", "Z"))]
    lowered = [label.lower() for label in labels]
    if str(axis).lower() in lowered:
        return lowered.index(str(axis).lower())
    if str(axis).lower() in ("x", "y", "z"):
        return "xyz".index(str(axis).lower())
    raise ValueError("no axis named %r; the axes are %s" % (axis, labels))


def _cut(mesh, normal, origin, shift):
    """A mesh's crossing with a plane, as ordered polylines."""
    sliced = mesh._polydata().translate(-shift).slice(
        normal=tuple(normal), origin=tuple(origin - shift))
    if sliced.n_points == 0:
        return []
    stripped = sliced.strip()
    points = _np.asarray(stripped.points, dtype=float) + shift
    lines, i = _np.asarray(stripped.lines), 0
    polylines = []
    while i < len(lines):
        count = int(lines[i])
        polylines.append(points[lines[i + 1:i + 1 + count]])
        i += count + 1
    return polylines


def _block_volumes(data):
    volume = getattr(data, "block_volume", None)
    if volume is not None:
        return _np.asarray(volume, dtype=float).ravel()
    return _np.full(data.n_data, float(_np.prod(data.step_size)))


def _half_diagonals(data):
    size = getattr(data, "block_size", None)
    if size is not None:
        return 0.5 * _np.linalg.norm(_np.asarray(size, dtype=float), axis=1)
    return _np.full(data.n_data,
                    0.5 * float(_np.linalg.norm(data.step_size)))


def _block_shares(data, body):
    """The share of each block inside `body`.

    Exact where a block lies wholly to one side of the surface, which is
    what a block farther from it than half its own diagonal does, so only
    the blocks the surface passes near are asked about their sub-blocks --
    on a real model a small part of them.
    """
    shares = _np.zeros(data.n_data)
    if body.n_data == 0:
        return shares
    centres = _np.asarray(data.coordinates, dtype=float)
    half = _half_diagonals(data)
    low = _np.ravel(body.bounding_box.min)
    high = _np.ravel(body.bounding_box.max)
    near = _np.all((centres + half[:, None] >= low)
                   & (centres - half[:, None] <= high), axis=1)
    rows = _np.flatnonzero(near)
    if rows.size == 0:
        return shares
    test = _within_body(body)
    shares[rows] = test(centres[rows])
    if getattr(data, "rows_per_location", 1) == 1:
        return shares
    shift = _np.round(low)
    with _DistanceQueries(body._polydata().translate(-shift)) as measure:
        distance = _np.abs(measure.query(centres[rows] - shift))
    straddling = rows[distance < half[rows]]
    if straddling.size:
        shares[straddling] = _sub_block_shares(data, test,
                                               rows=straddling)[straddling]
    return shares


def _densities(data, density):
    """A density per block, and its realizations where it has them."""
    n = data.n_data
    if density is None:
        return _np.ones(n), None
    if isinstance(density, (int, float, _np.number)):
        return _np.full(n, float(density)), None
    if density in data.metadata:
        return _np.asarray(data.get_metadata(density),
                           dtype=float).ravel(), None
    path, column = _contour_column(data, density)
    owner = data.get(path.parent) if path.name == "prediction" else None
    return _np.asarray(column.values, dtype=float).ravel(), \
        getattr(owner, "simulations", None)


def _realized_metal(store, volume_shares, mass, mass_store):
    """Every realization's grade, and density where it has them, summed
    into each body: `(n_bodies, n_realizations)`."""
    n_sim = int(store.shape[1])
    paired = mass_store is not None and int(mass_store.shape[1]) == n_sim
    total = _np.zeros((volume_shares.shape[1], n_sim))
    for band in store.row_bands():
        grades = _np.asarray(store[band, :], dtype=float)
        weight = volume_shares[band]
        if paired and mass_store is not None:
            densities = _np.asarray(mass_store[band, :], dtype=float)
            product = grades * densities
            total += weight.T @ _np.where(_np.isfinite(product), product, 0.0)
        else:
            amount = _np.where(_np.isfinite(grades), grades, 0.0)
            scaled = weight * _np.where(_np.isfinite(mass[band]),
                                        mass[band], 0.0)[:, None]
            total += scaled.T @ amount
    return total


def _exceedance(store, cutoff, side):
    """The share of each location's realizations on `side` of `cutoff`."""
    share = _np.full(int(store.shape[0]), _np.nan)
    for band in store.row_bands():
        values = _np.asarray(store[band, :], dtype=float)
        finite = _np.isfinite(values)
        counted = _np.sum(finite, axis=1)
        with _np.errstate(invalid="ignore"):
            hits = (values > cutoff) if side == "above" else (values < cutoff)
        found = _np.sum(hits & finite, axis=1) / _np.maximum(counted, 1)
        share[band] = _np.where(counted > 0, found, _np.nan)
    return share
