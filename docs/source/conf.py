# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'geoML'
copyright = '2025, Ítalo Gomes Gonçalves'
author = 'Ítalo Gomes Gonçalves'
release = '0.3.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # or '../..' if needed

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # Google/NumPy style support
    'sphinx.ext.viewcode',      # Adds links to source code
    'sphinx.ext.githubpages',   # Enables publishing on GitHub Pages
]


napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
