"""The package skill names only what exists, and never what is internal.

The skill in `plugins/geoml/skills/geoml/` tells an agent what to build. It
went stale for three releases with nothing to notice: at 0.8.0 it still
named shims 0.7.0 had removed. Every class, function and module path it
mentions -- in prose, in code, dotted or behind a notebook alias -- must
resolve against the package; none may be internal in the catalogue, since
those are kept for old saves only; and an experimental one must be called
experimental where it is named. What the skill's code does is the release
test's to check (`test_skill_release.py`).
"""
import builtins
import importlib
import inspect
import pathlib
import pkgutil
import re

import pytest

import geoml
import geoml.catalogue as catalogue

ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILL = ROOT / "plugins" / "geoml" / "skills" / "geoml"
TEXTS = [SKILL / "SKILL.md", SKILL / "references" / "notebook-style.md"]
# the notebook aliases the skill teaches
ALIASES = {"gl": "geoml.latent", "kr": "geoml.kernels", "tr": "geoml.transform",
           "lk": "geoml.likelihood", "wp": "geoml.warping"}
DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
CAMEL = re.compile(r"\b[A-Z][a-z0-9]+[A-Z0-9]\w*\b|\b[A-Z][a-z]{3,}\b")
MISSING = object()
# a file named in the text, not a module path
FILES = {"md", "py", "csv", "png", "json", "ipynb", "txt", "zarr", "io", "be"}
STRINGS = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
# the variables of the bundled data, named as data rather than as code
DATA_NAMES = {"Elements", "Landuse", "Rock"}
# a comment, or a heading shown inside a code block, is prose
COMMENTS = re.compile(r"#.*")
# a tree path (`Elements/Cd/prediction`) names columns, not code
PATHS = re.compile(r"[\w.<>]+(?:/[\w.<>]+)+")


@pytest.fixture(scope="module")
def built():
    return catalogue.build()


@pytest.fixture(scope="module")
def everything():
    """Every public name of every geoML module, to what it names."""
    found = {}
    for info in pkgutil.walk_packages(geoml.__path__, "geoml."):
        if ".test" in info.name:
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue
        for name, obj in vars(module).items():
            if not name.startswith("_"):
                found.setdefault(name, []).append(obj)
    return found


def _paragraphs(text):
    """`(paragraph, [code spans])` for each blank-line-separated block, a
    fenced block counting as one paragraph and all of it as code."""
    out, blocks = [], re.split(r"\n\s*\n", text)
    fenced = False
    for block in blocks:
        opens = block.count("```") % 2 == 1
        if fenced or block.lstrip().startswith("```"):
            out.append((block, [block]))
        else:
            out.append((block, re.findall(r"`([^`\n]+)`", block)))
        fenced = fenced != opens
    return out


def _mentions():
    """`(file, paragraph, name)` for every name the skill mentions in code."""
    for path in TEXTS:
        for paragraph, spans in _paragraphs(path.read_text(encoding="utf-8")):
            for span in spans:
                span = PATHS.sub("", COMMENTS.sub("", STRINGS.sub("", span)))
                for name in DOTTED.findall(span):
                    if name.rsplit(".", 1)[1] not in FILES:
                        yield path.name, paragraph, name
                for name in CAMEL.findall(DOTTED.sub("", span)):
                    if not hasattr(builtins, name) \
                            and name not in DATA_NAMES:
                        yield path.name, paragraph, name


def _resolve_dotted(name):
    """The object a dotted name means, or None when it is not a geoML path
    (`model.predict`, `grid.unpredicted`: a variable's methods)."""
    head, *rest = name.split(".")
    if head in ALIASES:
        start = ALIASES[head]
    elif head == "geoml":
        start = "geoml"
    elif hasattr(geoml, head) and inspect.ismodule(getattr(geoml, head)):
        start = "geoml." + head
    elif head[0].isupper():
        return "camel"
    else:
        return None
    obj = importlib.import_module(start)
    for part in rest:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            try:
                obj = importlib.import_module("%s.%s" % (obj.__name__, part))
            except (ImportError, AttributeError):
                return MISSING
    return obj


def _stabilities(built, objects):
    """The catalogue's stability for each object it describes."""
    known = catalogue.entries(built)
    found = []
    for obj in objects:
        key = "%s.%s" % (getattr(obj, "__module__", ""),
                         getattr(obj, "__qualname__", ""))
        if key in known:
            found.append(known[key].get("stability", "public"))
    return found


def test_the_skill_mentions_something():
    assert len(list(_mentions())) > 50


def test_every_name_the_skill_mentions_exists(everything):
    missing = []
    for file, _, name in _mentions():
        if "." in name:
            obj = _resolve_dotted(name)
            if obj is MISSING:
                missing.append((file, name))
            if obj is not MISSING and obj != "camel":
                continue
            name = name.split(".")[0]
        if name not in everything:
            missing.append((file, name))
    assert missing == []


def test_the_skill_names_nothing_internal(built, everything):
    internal = []
    for file, _, name in _mentions():
        obj = _resolve_dotted(name) if "." in name else "camel"
        objects = [obj] if obj not in (None, "camel", MISSING) else \
            everything.get(name.split(".")[0], [])
        stability = _stabilities(built, objects)
        if stability and all(s == "internal" for s in stability):
            internal.append((file, name))
    assert internal == []


def test_an_experimental_name_is_called_experimental(built, everything):
    unflagged = []
    for file, paragraph, name in _mentions():
        obj = _resolve_dotted(name) if "." in name else "camel"
        objects = [obj] if obj not in (None, "camel", MISSING) else \
            everything.get(name.split(".")[0], [])
        stability = _stabilities(built, objects)
        if stability and all(s == "experimental" for s in stability) \
                and "experimental" not in paragraph:
            unflagged.append((file, name))
    assert unflagged == []


def test_the_version_line_is_the_package_s():
    text = TEXTS[0].read_text(encoding="utf-8")
    assert "**Version:** %s." % geoml.__version__ in text
