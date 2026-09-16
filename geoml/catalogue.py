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
The catalogue: every class and function a model or a script may use,
described as data, for programs that build geoML models without reading its
code.

`build` returns it as a dictionary and `dumps` as JSON text;
``python -m geoml.catalogue [path]`` writes that text to a file or to the
standard output. Signatures, defaults and descriptions come from the code.
What introspection cannot know -- a class's category, how many parents it
takes, how its output size follows from its arguments, whether inducing
points pass through it -- each class declares beside its code, in a
`_catalogue` class attribute.
"""
import argparse as _argparse
import importlib as _importlib
import inspect as _inspect
import json as _json
import sys as _sys
import types as _pytypes
import typing as _typing
from typing import Any as _Any

__all__ = ["CATALOGUE_FORMAT", "build", "dumps"]

# the format this module writes; raised when a reader of the old one would
# misread the new, an added field keeping it
CATALOGUE_FORMAT = 1

# the modules whose every public class is catalogued
MODULES = ("geoml.latent.network", "geoml.latent.fourier", "geoml.kernels",
           "geoml.transform", "geoml.warping", "geoml.likelihood")

# the functions and methods a script calls, by the path they are reached
# at, and the category each belongs to
FUNCTIONS = {
    "geoml.data.inducing.from_kmeans": "inducing",
    "geoml.data.inducing.from_grid": "inducing",
    "geoml.data.inducing.combine": "inducing",
    "geoml.data.inducing.grid_experts": "inducing",
    "geoml.data.inducing.experts": "inducing",
    "geoml.stats.random.set_seed": "workflow",
    "geoml.models.GPOptions": "workflow",
    "geoml.models.VGPNetwork": "workflow",
    "geoml.models.VGPNetwork.train_full": "workflow",
    "geoml.models.VGPNetwork.train_svi": "workflow",
    "geoml.models.VGPNetwork.predict": "workflow",
    "geoml.models.VGPNetwork.save": "workflow",
    "geoml.models.VGPNetwork.open": "workflow",
    "geoml.models.refine": "workflow",
    "geoml.models.cross_validate": "workflow",
    "geoml.data.containers.PointData.assign_from_data": "workflow",
    "geoml.data.variables.ContinuousVariable.set_cutoffs": "workflow",
    "geoml.data.meshsets.MeshSet": "workflow",
    "geoml.data.meshsets.MeshSet.to_zarr": "workflow",
    # the module is private so that `geoml.progress` is the context manager
    # rather than a module holding one of that name; the path below is what
    # an import resolves, and `geoml.progress` is what a script writes
    "geoml._progress.progress": "workflow",
    "geoml.data.containers.PointData.unpredicted": "workflow",
}

WORKFLOW = {
    "fit": {
        "network": "geoml.models.VGPNetwork",
        "options": "geoml.models.GPOptions",
        "seed": "geoml.stats.random.set_seed",
        "train": ["geoml.models.VGPNetwork.train_full",
                  "geoml.models.VGPNetwork.train_svi"],
        "validate": "geoml.models.cross_validate",
        "save": "geoml.models.VGPNetwork.save",
        "load": "geoml.models.VGPNetwork.open",
        "leaves": "one_per_likelihood",
    },
    "predict": {
        "predict": "geoml.models.VGPNetwork.predict",
        "refine": "geoml.models.refine",
        "reach": "geoml.data.containers.PointData.assign_from_data",
        "cutoffs": "geoml.data.variables.ContinuousVariable.set_cutoffs",
        # what a cancelled prediction is finished with: hand the answer
        # back as `where` and the rest is visited
        "resume": "geoml.data.containers.PointData.unpredicted",
    },
    "contour": {
        "mesh_set": "geoml.data.meshsets.MeshSet",
        "save": "geoml.data.meshsets.MeshSet.to_zarr",
    },
    # spans the others: every long call reports inside the block, and the
    # callback raising is the cancel
    "progress": {
        "report": "geoml._progress.progress",
        "cancel": "callback_raises",
    },
}

# the variable types a likelihood can be bound to: the class, and what
# beyond `length` -- the columns it hands a likelihood -- a size rule may
# read; `n_classes` and `n_components` count its labels
VARIABLES = {
    "continuous": ("ContinuousVariable", ()),
    "vector": ("VectorVariable", ("n_components",)),
    "compositional": ("CompositionalVariable", ("n_components",)),
    "rock_type": ("RockTypeVariable", ("n_classes",)),
    "categorical": ("CategoricalVariable", ("n_classes",)),
    "ordered_rock_type": ("OrderedRockType", ("n_classes",)),
    "binary": ("BinaryVariable", ()),
    "anomaly": ("AnomalyVariable", ()),
}

_KINDS = {_inspect.Parameter.POSITIONAL_ONLY: "positional",
          _inspect.Parameter.KEYWORD_ONLY: "keyword",
          _inspect.Parameter.VAR_POSITIONAL: "variadic"}


def build() -> "dict[str, _Any]":
    """The catalogue of the geoML in use, as a dictionary.

    Returns
    -------
    dict
        `catalogue_format`, `geoml_version`, `persistence_format` (the
        format number a saved model carries), `variable_types`, `classes`
        and `functions`, both keyed by the dotted path of what they
        describe, and `workflow`. Every value is JSON.
    """
    import geoml
    import geoml.persistence as _persistence
    return {
        "catalogue_format": CATALOGUE_FORMAT,
        "geoml_version": geoml.__version__,
        "persistence_format": _persistence._GEOML_MODEL_FORMAT,
        "variable_types": _variable_types(),
        "classes": {path(cls): class_entry(cls) for cls in classes()},
        "functions": {name: _function_entry(name, category)
                      for name, category in FUNCTIONS.items()},
        "workflow": WORKFLOW,
    }


def dumps() -> str:
    """The catalogue as JSON text, keys sorted: the same bytes for the same
    geoML."""
    return _json.dumps(build(), sort_keys=True, indent=1,
                       ensure_ascii=False) + "\n"


def path(cls: type) -> str:
    """The dotted path a class is catalogued, and saved, under."""
    return cls.__module__ + "." + cls.__qualname__


def classes() -> "list[type]":
    """Every public class defined in a catalogued module, the exceptions
    left out, in the order of `MODULES` and then of name."""
    found = []
    for module_name in MODULES:
        module = _importlib.import_module(module_name)
        for label, cls in sorted(vars(module).items()):
            if _inspect.isclass(cls) and not label.startswith("_") \
                    and cls.__module__ == module_name \
                    and not issubclass(cls, BaseException):
                found.append(cls)
    return found


def class_entry(cls: type) -> "dict[str, _Any]":
    """One class's entry: what it declares, its parameters read from its
    constructor, and the first paragraph of its docstring."""
    declared = dict(cls.__dict__["_catalogue"])
    overrides = declared.pop("params", {})
    entry: "dict[str, _Any]" = {"label": cls.__name__,
                                "summary": _summary(cls),
                                "stability": "public"}
    entry.update(declared)
    entry["params"] = _parameters(getattr(cls, "__init__"), overrides,
                                  _parameter_docs(cls))
    return entry


def _function_entry(name, category):
    obj = resolve(name)
    function = getattr(obj, "__init__") if _inspect.isclass(obj) else obj
    overrides = obj.__dict__.get("_catalogue", {}).get("params", {}) \
        if _inspect.isclass(obj) else {}
    entry: "dict[str, _Any]" = {
        "category": category, "summary": _summary(obj),
        "params": _parameters(function, overrides, _parameter_docs(obj))}
    if not _inspect.isclass(obj):
        returns = _signature(function).return_annotation
        if returns is not _inspect.Signature.empty:
            entry["returns"] = _describe(returns)["type"]
    return entry


def resolve(name: str) -> _Any:
    """The object a dotted path names: a module's attribute, or an
    attribute of one."""
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = _importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError("nothing is importable along %r" % name)


def _variable_types():
    import geoml.data.variables as _variables
    return {name: {"class": path(getattr(_variables, cls)),
                   "properties": dict({"length": "int"},
                                      **{p: "int" for p in extra})}
            for name, (cls, extra) in VARIABLES.items()}


def _summary(obj):
    """The first paragraph of the object's own docstring -- a class's, or
    its constructor's where the class has none."""
    doc = obj.__doc__
    if _inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__") or getattr(obj, "__init__").__doc__
    doc = _inspect.cleandoc(doc or "")
    return " ".join(doc.split("\n\n")[0].split())


def _parameter_docs(obj):
    """`{name: text}` from the `Parameters` section of a numpydoc
    docstring: the class's, then its constructor's."""
    docs = {}
    for source in (obj, getattr(obj, "__init__", None)):
        lines = (_inspect.getdoc(source) or "").splitlines()
        heads = [i for i in range(len(lines) - 1)
                 if lines[i].strip() == "Parameters"
                 and set(lines[i + 1].strip()) == {"-"}]
        if not heads:
            continue
        name, text = None, []
        for i in range(heads[0] + 2, len(lines)):
            line = lines[i]
            if i + 1 < len(lines) and lines[i + 1].strip() \
                    and set(lines[i + 1].strip()) == {"-"}:
                break        # the heading of the next section
            if line and not line[0].isspace():
                if name:
                    docs.setdefault(name, " ".join(text))
                name, text = line.split(":")[0].strip().lstrip("*"), []
            elif line.strip():
                text.append(line.strip())
        if name:
            docs.setdefault(name, " ".join(text))
    return docs


def _signature(function):
    """With the string annotations evaluated where they were written,
    unless one names something imported for the checker alone."""
    try:
        return _inspect.signature(function, eval_str=True)
    except NameError:
        return _inspect.signature(function)


def _parameters(function, overrides, docs):
    params = []
    for name, p in _signature(function).parameters.items():
        if name in ("self", "cls") or p.kind is p.VAR_KEYWORD:
            continue
        entry: "dict[str, _Any]" = {
            "name": name,
            "kind": _KINDS.get(p.kind, "positional"
                               if p.default is p.empty else "keyword")}
        entry.update(_describe(p.annotation))
        if p.kind is p.VAR_POSITIONAL:
            entry["required"] = False
        elif p.default is p.empty:
            entry["required"] = True
        else:
            entry["required"] = False
            entry["default"] = _json_default(p.default, name)
        # where null is an answer of its own, not only the default's stand-in
        entry["nullable"] = p.default is None or type(None) in \
            _typing.get_args(p.annotation)
        if name in docs:
            entry["doc"] = docs[name]
        declared = overrides.get(name, {})
        constraints = dict(entry.get("constraints", {}),
                           **declared.get("constraints", {}))
        entry.update(declared)
        if constraints:
            entry["constraints"] = constraints
        params.append(entry)
    return params


def _json_default(value, name):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError("the default of %r, %r, is no JSON number"
                            % (name, value))
        return value
    if isinstance(value, (tuple, list)):
        return [_json_default(v, name) for v in value]
    raise TypeError("the default of %r is an object, %r, which JSON cannot "
                    "carry: default to None and build it in the constructor"
                    % (name, value))


def _describe(annotation):
    """The catalogue's type for an annotation, with its choices where it
    names them; `json` where there is no better word."""
    if annotation is _inspect.Parameter.empty or isinstance(annotation, str):
        return {"type": "json"}
    origin = _typing.get_origin(annotation)
    args = [a for a in _typing.get_args(annotation) if a is not type(None)]
    if origin in (_typing.Union, _pytypes.UnionType):
        return _describe(args[0]) if len(args) == 1 else {"type": "json"}
    if origin is _typing.Literal:
        return {"type": "enum", "constraints": {"choices": list(args)}}
    if origin in (list, tuple) or getattr(origin, "__name__", "") \
            == "Sequence":
        inner = _describe(args[0])["type"] if args else "json"
        return {"type": inner + "[]" if inner != "json" else "json"}
    for plain, name in ((bool, "bool"), (int, "int"), (float, "float"),
                        (str, "str")):
        if annotation is plain:
            return {"type": name}
    if _inspect.isclass(annotation):
        for base, name in _references():
            if issubclass(annotation, base):
                return {"type": name}
    return {"type": "json"}


def _references():
    import geoml.data as _data
    import geoml.kernels as _kernels
    import geoml.latent.network as _network
    import geoml.likelihood as _likelihood
    import geoml.transform as _transform
    import geoml.warping as _warping
    return ((_kernels._Kernel, "ref:kernel"),
            (_kernels._AbstractCovariance, "ref:covariance"),
            (_transform._Transform, "ref:transform"),
            (_warping._Warping, "ref:warping"),
            (_likelihood._Likelihood, "ref:likelihood"),
            (_network._LatentVariable, "node"),
            (_data.DirectionalData, "data:DirectionalData"),
            (_data.BlockSet3D, "data:BlockSet3D"),
            (_data.PointData, "data:PointData"))


def main(argv: "list[str] | None" = None) -> None:
    """``python -m geoml.catalogue [path]``."""
    parser = _argparse.ArgumentParser(
        prog="python -m geoml.catalogue",
        description="Write the catalogue of the geoML in use as JSON.")
    parser.add_argument("path", nargs="?",
                        help="the file to write; the standard output if "
                             "left out")
    target = parser.parse_args(argv).path
    text = dumps()
    if target is None:
        _sys.stdout.write(text)
    else:
        with open(target, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)


if __name__ == "__main__":
    main()
