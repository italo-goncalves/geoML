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
Container persistence: the Zarr writers and rebuilders behind
`_SpatialData.to_zarr` / `open`. Containers are recorded by bare class
name and rebuilt by the dispatch here, so moving a class between modules
never invalidates a store.
"""
import numpy as _np
import pandas as _pd

import geoml.math.geometry as _gmt
import geoml.storage as _storage

from geoml.data.base import *
from geoml.data.base import _Attribute
from geoml.data.variables import *
from geoml.data.variables import _Category, _Component
from geoml.data.containers import *
from geoml.data.containers import _PointBased, _SpatialData
from geoml.data.grids import *
from geoml.data.meshes import *
from geoml.data.blocks import *

# --------------------------------------------------------------------------- #
# Zarr persistence (see _SpatialData.to_zarr / _SpatialData.open)
# --------------------------------------------------------------------------- #
# 2: the dict families and the node facts persist off the declarations, with
# the Zarr key spelling the same string `get` takes (`quantiles/0.5`, not
# `quantile_0.5`). Stores written at format 1 are not read back -- agreed at
# the time of the change, everything in use being refreshed after it.
_GEOML_ZARR_FORMAT = 2


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
    if type(container) in (BlockSet3D, RotatedBlockSet3D):
        # the lattice itself, in integers, plus what it is counted in; the
        # centres are derived from those and are not stored twice
        write("_origin", container._origin)
        write("_level", container._level)
        meta = {"class": type(container).__name__,
                "labels": [str(lb) for lb in container.coordinate_labels],
                "box_corner": [float(x) for x in container.box_corner],
                "base_step": [float(x) for x in container.base_step],
                "lattice_shape": [int(x) for x in container.lattice_shape],
                "discretization":
                    [int(d) for d in container.discretization],
                "max_levels": int(container.max_levels)}
        if type(container) is RotatedBlockSet3D:
            meta.update(azimuth=float(container.azimuth),
                        dip=float(container.dip),
                        rake=float(container.rake),
                        pivot=[float(x) for x in container._pivot])
        return meta
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
    if cls_name in ("BlockSet3D", "RotatedBlockSet3D"):
        # Bypass __init__, which builds a full coarse grid: the lattice that
        # was saved is whatever it had been refined to since.
        chosen = RotatedBlockSet3D if cls_name == "RotatedBlockSet3D" \
            else BlockSet3D
        blocks = chosen.__new__(chosen)
        _PointBased.__init__(blocks)
        blocks.max_levels = int(meta["max_levels"])
        blocks.discretization = [int(d) for d in meta["discretization"]]
        blocks._sub_grid = _gmt.unit_sub_grid(blocks.discretization)
        blocks.base_step = _np.asarray(meta["base_step"], dtype=float)
        blocks.box_corner = _np.asarray(meta["box_corner"], dtype=float)
        blocks.lattice_shape = _np.asarray(meta["lattice_shape"],
                                           dtype=_np.int64)
        if cls_name == "RotatedBlockSet3D":
            # before the centres: `_centres` turns through these
            blocks.azimuth = float(meta["azimuth"])
            blocks.dip = float(meta["dip"])
            blocks.rake = float(meta["rake"])
            blocks._pivot = _np.asarray(meta["pivot"], dtype=float)
        blocks._origin = read("_origin").astype(_np.int64)
        blocks._level = read("_level").astype(_np.int64)
        # a saved model is one that was predicted into, not one mid-refinement
        blocks._fresh = _np.zeros(len(blocks._origin), dtype=bool)
        blocks._init_coordinates(blocks._centres(), meta["labels"])
        blocks._bounding_box = blocks._lattice_bounding_box()
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
    return (ContinuousVariable, DerivedVariable, VectorVariable,
            CompositionalVariable, RockTypeVariable, CategoricalVariable,
            OrderedRockType, BinaryVariable, AnomalyVariable)


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
    elif cls_name == "DerivedVariable":
        # the recipe (the function) lives in the script that derived it;
        # what is reloaded is the values, plus `parents` via node_attrs
        container.variables[name] = DerivedVariable(name, container)
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

