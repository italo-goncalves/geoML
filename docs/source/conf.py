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
release = "0.6.2"

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
