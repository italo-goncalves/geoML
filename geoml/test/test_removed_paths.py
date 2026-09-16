"""The 0.6.0 paths, removed in 0.7.0, and why removing them was safe.

The package moved into subpackages in 0.6.0 and left a one-line shim at ten
old flat paths; 0.6.4 had them warn, and 0.7.0 removed them. A path a saved
model can name has to stay importable forever -- persistence rebuilds a
model by importing the dotted path it recorded for each class -- so what is
pinned here besides the removal is the argument that allowed it: every class
a store can record resolves from the path it records, and none of those
paths is one of the ten.
"""
import importlib

import pytest

import geoml
import geoml.latent.fourier  # noqa: F401 -- unadvertised, but saveable
import geoml.models
import geoml.parameter
import geoml.persistence as persistence

REMOVED = ("drillhole", "geometry", "graphviz", "inducing", "interpolation",
           "plotly", "probability", "pyvista", "random", "tftools")


@pytest.mark.parametrize("name", REMOVED)
def test_the_old_module_is_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("geoml." + name)


@pytest.mark.parametrize("name", REMOVED)
def test_the_old_attribute_is_gone(name):
    assert not hasattr(geoml, name)


def _recordable():
    """Every package class a model store can name.

    Persistence writes a `Parametric` or an options object by its class
    path and refuses to save anything else. Classes the tests define are
    left out, and so is anything defined inside a function, which has no
    path to record.
    """
    found = set()
    stack = [geoml.parameter.Parametric, geoml.models._ModelOptions]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub in found:
                continue
            stack.append(sub)
            if sub.__module__.startswith("geoml") \
                    and not sub.__module__.startswith("geoml.test") \
                    and "<locals>" not in sub.__qualname__:
                found.add(sub)
    return sorted(found, key=lambda c: (c.__module__, c.__qualname__))


def test_every_recordable_class_resolves_from_the_path_it_records():
    classes = _recordable()
    assert len(classes) > 50
    unresolved = []
    for cls in classes:
        recorded = cls.__module__ + "." + cls.__qualname__
        try:
            if persistence._resolve(recorded) is not cls:
                unresolved.append(recorded)
        except persistence.ModelFormatError:
            unresolved.append(recorded)
    assert unresolved == []


def test_no_recordable_class_lives_under_a_removed_path():
    homes = {cls.__module__ for cls in _recordable()}
    assert not homes & {"geoml." + name for name in REMOVED}
