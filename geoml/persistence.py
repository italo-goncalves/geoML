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

"""
Saving and loading trained models.

A model is saved by writing down how it was built — the class of every object
in it and the arguments it was given — and then rebuilding it with the same
calls when loading. `Parametric` records those arguments on its own (see
`parameter.Parametric.__init_subclass__`), so nothing needs to be declared per
class. The trained parameter values are written alongside the description and
assigned to the rebuilt model.

The result is a complete model: it keeps its training data, its variables and
their types, and can be used for prediction or trained further. Pass `data` to
`load_model()` to rebuild the same architecture around a different data set.

Everything goes into a single Zarr store, laid out as

    <path>/
        arrays/         numeric arguments and the trained parameter values
        containers/     data objects, each written with `_SpatialData.to_zarr`

with the description itself in the store's attributes.
"""

__all__ = ["save_model", "load_model"]

import os as _os
import importlib as _importlib
import inspect as _inspect
from typing import TYPE_CHECKING

import numpy as _np
import tensorflow as _tf
import zarr as _zarr

import geoml._types as _types
import geoml.data as _data
import geoml.parameter as _gpr
import geoml.storage as _storage

if TYPE_CHECKING:
    # only for the annotations: importing the models at run time would close
    # the circle, since that is the module calling these functions
    import geoml.models as _models


_GEOML_MODEL_FORMAT = 1


class ModelFormatError(Exception):
    """Raised when a model store cannot be read or rebuilt."""
    pass


def _class_name(obj):
    cls = type(obj)
    return cls.__module__ + "." + cls.__qualname__


def _resolve(name):
    """Finds a class from the name written in the store.

    Only classes from `geoml` itself are accepted: rebuilding a model means
    calling the constructors named in the file, and a model store should not be
    able to reach arbitrary code.
    """
    module_name, _, cls_name = name.rpartition(".")
    if module_name != "geoml" and not module_name.startswith("geoml."):
        raise ModelFormatError(
            "refusing to load '%s': only geoml classes can be rebuilt" % name)

    module = _importlib.import_module(module_name)
    cls = module
    for part in cls_name.split("."):
        cls = getattr(cls, part, None)
        if cls is None:
            raise ModelFormatError("class '%s' not found" % name)
    return cls


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
class _Writer(object):
    """Turns a model into a JSON-able description, storing arrays on the way."""

    def __init__(self, group, path):
        self.group = group
        self.path = path
        self.arrays = group.create_group("arrays")
        self.memo = {}          # id(object) -> description already written
        self.held = []          # keeps the described objects alive, so that
                                # their ids are not reused while writing
        self.n_array = 0
        self.n_container = 0

    def put_array(self, values):
        name = "a%d" % self.n_array
        self.n_array += 1
        _storage.ArrayStore.from_numpy(values).write_into(self.arrays, name)
        return name

    def put_container(self, container):
        name = "c%d" % self.n_container
        self.n_container += 1
        container.to_zarr(_os.path.join(self.path, "containers", name))
        return name


def _encode(obj, writer):
    """Describes one constructor argument."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)

    if isinstance(obj, list):
        return [_encode(item, writer) for item in obj]

    if isinstance(obj, tuple):
        return {"$": "tuple", "items": [_encode(item, writer) for item in obj]}

    if isinstance(obj, dict):
        if not all(isinstance(key, str) for key in obj.keys()):
            raise ModelFormatError(
                "only dictionaries with string keys can be saved")
        return {"$": "map",
                "items": {k: _encode(v, writer) for k, v in obj.items()}}

    # objects appearing more than once (a shared transform, a node feeding two
    # children) must stay shared: the parameters are matched by position, and
    # duplicating an object would shift every position after it
    if id(obj) in writer.memo:
        return {"$": "ref", "id": writer.memo[id(obj)]["id"]}

    if isinstance(obj, _tf.Tensor):
        obj = obj.numpy()
    if isinstance(obj, _storage.ArrayStore):
        obj = _np.asarray(obj)

    if isinstance(obj, _np.ndarray):
        node = {"$": "array", "name": writer.put_array(obj)}
    elif isinstance(obj, _data._SpatialData):
        node = {"$": "container", "name": writer.put_container(obj)}
    elif isinstance(obj, _gpr.Parametric):
        node = _encode_parametric(obj, writer)
    else:
        import geoml.models as _mdl
        if isinstance(obj, _mdl._ModelOptions):
            node = {"$": "options", "class": _class_name(obj),
                    "values": dict(vars(obj))}
        else:
            raise ModelFormatError(
                "cannot save an argument of type %s" % type(obj).__name__)

    node["id"] = len(writer.memo)
    writer.memo[id(obj)] = node
    writer.held.append(obj)
    return node


def _encode_parametric(obj, writer):
    if "_init_args" not in obj.__dict__:
        raise ModelFormatError(
            "%s did not record its arguments; it was probably built before "
            "`geoml.parameter.Parametric` was imported"
            % type(obj).__name__)

    return {
        "$": "object",
        "class": _class_name(obj),
        "args": [_encode(a, writer) for a in obj._init_args],
        "kwargs": {k: _encode(v, writer)
                   for k, v in obj._init_kwargs.items()},
    }


def _write_parameters(group, model):
    """Writes the trained values, along with the bounds and fixed flags.

    The bounds and the fixed flags are part of the trained state too: they can
    be changed after construction (`set_limits`, `fix`), and rebuilding only
    restores whatever the constructors set up.
    """
    values, min_val, max_val, shapes, fixed = [], [], [], [], []

    for parameter in model.all_parameters:
        values.append(_np.reshape(parameter.variable.numpy(), [-1]))
        min_val.append(_np.reshape(parameter.min_transformed.numpy(), [-1]))
        max_val.append(_np.reshape(parameter.max_transformed.numpy(), [-1]))
        shapes.append([int(s) for s in parameter.variable.shape])
        fixed.append(bool(parameter.fixed))

    params = group.create_group("parameters")
    for name, arrays in (("value", values), ("min", min_val), ("max", max_val)):
        flat = _np.concatenate(arrays, axis=0) if arrays else _np.zeros([0])
        _storage.ArrayStore.from_numpy(flat).write_into(params, name)

    return {"shapes": shapes, "fixed": fixed}


def save_model(model: "_models._GPModel",
               path: _types.PathLike) -> _types.PathLike:
    """
    Saves a trained model to a single Zarr store.

    The model's structure, its trained parameters and its training data are all
    written, so that `load_model()` returns a model that can predict or be
    trained further.

    Parameters
    ----------
    model
        A model object from the `models` module.
    path : str
        Directory to write to. It is created if necessary and overwritten if it
        already exists.

    Returns
    -------
    path : str
        The path that was written to.
    """
    group = _zarr.open_group(path, mode="w")
    writer = _Writer(group, path)

    meta = {
        "geoml_format": _GEOML_MODEL_FORMAT,
        "class": _class_name(model),
        "spec": _encode(model, writer),
        "parameters": _write_parameters(group, model),
    }

    training_log = [float(x) for x in getattr(model, "training_log", [])]
    if len(training_log) > 0:
        _storage.ArrayStore.from_numpy(
            _np.array(training_log)).write_into(group, "training_log")
        meta["training_log"] = True

    group.attrs["geoml_model"] = meta
    return path


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
class _Reader(object):
    def __init__(self, group, path):
        self.group = group
        self.path = path
        self.objects = {}       # description id -> rebuilt object
        self.building = set()   # ids being built, to catch a cyclic file


def _decode(node, reader):
    if node is None or isinstance(node, (bool, int, float, str)):
        return node

    if isinstance(node, list):
        return [_decode(item, reader) for item in node]

    kind = node["$"]

    if kind == "tuple":
        return tuple(_decode(item, reader) for item in node["items"])

    if kind == "map":
        return {k: _decode(v, reader) for k, v in node["items"].items()}

    if kind == "ref":
        if node["id"] not in reader.objects:
            raise ModelFormatError("the model description is cyclic")
        return reader.objects[node["id"]]

    if node["id"] in reader.objects:
        return reader.objects[node["id"]]
    if node["id"] in reader.building:
        raise ModelFormatError("the model description is cyclic")
    reader.building.add(node["id"])

    if kind == "array":
        obj = _np.asarray(reader.group["arrays"][node["name"]][...])
    elif kind == "container":
        obj = _data.PointData.open(
            _os.path.join(reader.path, "containers", node["name"]))
    elif kind == "options":
        cls = _resolve(node["class"])
        obj = cls.__new__(cls)
        vars(obj).update(node["values"])
    elif kind == "object":
        cls = _resolve(node["class"])
        args = [_decode(a, reader) for a in node["args"]]
        kwargs = {k: _decode(v, reader) for k, v in node["kwargs"].items()}
        obj = cls(*args, **kwargs)
    else:
        raise ModelFormatError("unknown entry '%s' in the model" % kind)

    reader.building.discard(node["id"])
    reader.objects[node["id"]] = obj
    return obj


def _read_parameters(group, model, meta):
    shapes = [tuple(sh) for sh in meta["shapes"]]
    fixed = meta["fixed"]

    if len(shapes) != len(model.all_parameters):
        raise ModelFormatError(
            "the rebuilt model has %s parameters, but the file has %s"
            % (len(model.all_parameters), len(shapes)))

    params = group["parameters"]
    sizes = _np.cumsum([int(_np.prod(sh)) for sh in shapes])[:-1]
    blocks = {name: _np.split(_np.asarray(params[name][...]), sizes)
              for name in ("value", "min", "max")}

    for i, parameter in enumerate(model.all_parameters):
        if tuple(parameter.variable.shape) != shapes[i]:
            raise ModelFormatError(
                "parameter '%s' has shape %s, but the file has %s"
                % (parameter.name, tuple(parameter.variable.shape), shapes[i]))

        # the bounds come first: assigning a value clamps it to them
        parameter.min_transformed.assign(
            _np.reshape(blocks["min"][i], shapes[i]))
        parameter.max_transformed.assign(
            _np.reshape(blocks["max"][i], shapes[i]))
        parameter.set_value(_np.reshape(blocks["value"][i], shapes[i]),
                            transformed=True)
        parameter.fixed = bool(fixed[i])


def _substitute_data(spec, data, reader):
    """Points the model's `data` argument at another container."""
    cls = _resolve(spec["class"])
    names = list(_inspect.signature(cls.__init__).parameters)[1:]

    if "data" not in names:
        raise ModelFormatError(
            "%s does not take a `data` argument" % spec["class"])

    reader.objects["new_data"] = data
    node = {"$": "ref", "id": "new_data"}

    position = names.index("data")
    if position < len(spec["args"]):
        spec["args"][position] = node
    else:
        spec["kwargs"]["data"] = node


def load_model(path: _types.PathLike,
               data: "_data._SpatialData | None" = None
               ) -> "_models._GPModel":
    """
    Loads a model saved with `save_model()`.

    The model is rebuilt by calling the same constructors it was built with, and
    the trained parameters are assigned to it. It keeps its variables and their
    types, so it can predict on new data objects right away.

    Parameters
    ----------
    path : str
        A directory written by `save_model()`.
    data
        A data object to build the model around, in place of the one it was
        trained on. The variables the model uses must be present in it. Use this
        to fit an already designed model to a new data set; note that the
        trained parameters are kept, acting as the starting point for training.

    Returns
    -------
    model
        The model, of whatever class was saved.
    """
    group = _zarr.open_group(path, mode="r")
    if "geoml_model" not in group.attrs:
        raise ModelFormatError("'%s' does not contain a geoml model" % path)

    meta = dict(group.attrs["geoml_model"])
    if meta["geoml_format"] != _GEOML_MODEL_FORMAT:
        raise ModelFormatError(
            "model format %s is not supported" % meta["geoml_format"])

    reader = _Reader(group, path)
    spec = meta["spec"]
    if data is not None:
        _substitute_data(spec, data, reader)

    model = _decode(spec, reader)
    _read_parameters(group, model, meta["parameters"])

    if meta.get("training_log", False):
        model.training_log = [float(x) for x in group["training_log"][...]]

    return model
