"""The 0.6.0 aliases: still working, and now saying they are on the way out.

The package moved into subpackages in 0.6.0 and left a one-line shim at
every old flat path. They promised "one release" in a docstring and said
nothing at runtime, so 0.6.4 gave them a `DeprecationWarning` ahead of
removing them in 0.7.0. What is pinned here is the shape of that notice:
loud for anyone on an old path, silent for everyone else.
"""

import importlib
import os
import subprocess
import sys
import warnings

import pytest

import geoml
from geoml import _deprecated


def _fresh(name):
    """Forgets a shim, module and parent attribute alike.

    Importing a submodule binds it on the parent package, so a second
    `geoml.geometry` never reaches the module-level `__getattr__` -- which
    is correct (one notice per session) and would make this test measure
    nothing if the attribute were left in place.
    """
    sys.modules.pop("geoml." + name, None)
    geoml.__dict__.pop(name, None)


@pytest.mark.parametrize("name", sorted(_deprecated.MOVED))
def test_the_old_attribute_warns_once_and_still_works(name):
    _fresh(name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = getattr(geoml, name)

    ours = [w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "deprecated alias" in str(w.message)]
    assert len(ours) == 1, "expected exactly one notice, got %d" % len(ours)
    assert "0.7.0" in str(ours[0].message)
    assert _deprecated.MOVED[name] in str(ours[0].message)
    # attributed to this file rather than to the shim: a DeprecationWarning
    # blamed on library code is one Python's default filter hides
    assert ours[0].filename == __file__
    assert module.__name__ == "geoml." + name


@pytest.mark.parametrize("name", sorted(_deprecated.MOVED))
def test_importing_the_old_module_directly_warns(name):
    _fresh(name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("geoml." + name)

    assert any("deprecated alias" in str(w.message) for w in caught)


def test_a_plain_import_is_silent_and_loads_no_shim():
    """The reason the shims are resolved lazily, and that nothing inside
    the package reaches for one.

    Importing all ten up front would fire ten notices at every `import
    geoml`, for the majority of users who never touch an old path -- which
    is how a warning teaches people to ignore warnings. `kernels` also
    reached three helpers through the `tftools` shim until 0.6.4, which
    would have fired one notice for everyone on its own.

    A subprocess rather than a re-import: `geoml` is a package whose
    submodules hold references to it, so dropping it from `sys.modules`
    and importing it again raises on the half-built parent. A fresh
    interpreter is the only honest way to ask what a first import does.
    """
    program = (
        "import warnings, sys\n"
        "warnings.simplefilter('always')\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    import geoml\n"
        "print(sum('deprecated alias' in str(w.message) for w in caught))\n"
        "print(sorted(n for n in sys.modules if n.count('.') == 1 and "
        "n.startswith('geoml.') and n.split('.')[1] in "
        "__import__('geoml')._deprecated.MOVED))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    assert result.returncode == 0, result.stderr
    notices, loaded = result.stdout.strip().splitlines()[-2:]
    assert notices == "0", "a plain import emitted %s notice(s)" % notices
    assert loaded == "[]", "a plain import loaded shims: %s" % loaded


def test_an_unknown_attribute_still_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        geoml.definitely_not_a_module


def test_dir_lists_the_aliases():
    assert set(_deprecated.MOVED).issubset(set(dir(geoml)))
