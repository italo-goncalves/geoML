# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "geoML"
copyright = "2025, Ítalo Gomes Gonçalves"
author = "Ítalo Gomes Gonçalves"
# One of the three places the version is written; bump it with
# geoml/__init__.py and pyproject.toml at every release.
release = "0.6.8"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # NumPy-style docstring sections
    "sphinx.ext.viewcode",      # links from the reference to the source
    "sphinx.ext.intersphinx",   # links to numpy/pandas/python objects
    "sphinx.ext.githubpages",   # .nojekyll, for publishing on Pages
    "myst_parser",              # Markdown pages: the manual, the records
]

templates_path = ["_templates"]
exclude_patterns = []

# The manual and the design records are Markdown; the reference pages are rst.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Autodoc -----------------------------------------------------------------

# The annotation is the single source of type information: it is moved into
# the parameter description rather than repeated in the signature, and the
# docstrings carry descriptions alone.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Instance attributes are documented in the class docstring's Attributes
# section, so autodoc need not repeat what it can see.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
# Off by default: the reference is built without network access in CI.
intersphinx_disabled_reftypes = ["*"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

# -- HTML --------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "geoML %s" % release


# -- Shared implementations, documented under each class that offers them ----
#
# Several classes here deliberately share one function object: the
# `_blockdata` decorator hands the block fan-out to `Blocks1D/2D/3D`, and a
# few latent nodes and transforms reuse a sibling's method rather than
# reimplement it. Autodoc identifies a member by its qualified name, which is
# the name of whichever class defined the function, so documenting two
# classes that share one reads as documenting the same object twice.
#
# The build runs under `-W`, and that warning is the one kind this project
# produces on purpose, so it is filtered by message rather than by turning
# the gate off. Nothing else is suppressed: a broken reference, a malformed
# docstring or an unimportable module still fails the build.
import logging as _logging


class _AllowSharedMembers(_logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(
            "duplicate object description of")


# -- The manual ---------------------------------------------------------------
#
# The chapters live in `docs/manual/`, where they are written and where
# `run_blocks.py` executes them, and they link their figures relatively
# (`figures/02-x.png`). Sphinx resolves an image against the document that
# carries it, so including the chapters from elsewhere in the tree would
# break every figure. They are copied into `docs/source/manual/` instead,
# just before the sources are read -- one copy in git, working images, and
# nothing to keep in step by hand. The copy is gitignored.
import pathlib as _pathlib
import shutil as _shutil


def _copy_manual(app):
    source = _pathlib.Path(app.srcdir).parent / "manual"
    target = _pathlib.Path(app.srcdir) / "manual"
    if not source.is_dir():
        return
    if target.exists():
        _shutil.rmtree(target)
    target.mkdir()
    for chapter in sorted(source.glob("[0-9][0-9]-*.md")):
        _shutil.copy2(chapter, target / chapter.name)
    # The manual's README is its front page: its chapter list reads as a
    # table of contents already, and the links in it work. Sphinx needs the
    # chapters in a toctree as well -- for the sidebar, and so that "next"
    # and "previous" lead somewhere -- so a hidden one is appended to the
    # copy. The README in the repository stays as it is, since it is also
    # read on its own.
    readme = source / "README.md"
    if readme.exists():
        chapters = sorted(path.stem for path in source.glob("[0-9][0-9]-*.md"))
        hidden = "\n\n```{toctree}\n:hidden:\n\n%s\n```\n" % "\n".join(chapters)
        (target / "index.md").write_text(
            readme.read_text(encoding="utf-8") + hidden, encoding="utf-8")
    figures = source / "figures"
    if figures.is_dir():
        _shutil.copytree(figures, target / "figures")


def setup(app):
    _logging.getLogger("sphinx").addFilter(_AllowSharedMembers())
    app.connect("builder-inited", _copy_manual)
    return {"parallel_read_safe": True}
