# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

from sphinx.ext.autodoc.importer import import_module

# Verify that the generated stub file exists
stub_path = os.path.abspath("../../pysplashsurf.pyi")
if not os.path.exists(stub_path):
   print(f"### Error: Stub file not found at {stub_path}. Please ensure stub_gen produced pysplashsurf.pyi.")
   sys.exit(1)

# Import the stub file
print(f"### Prepend to path: {os.path.abspath('../../')}")
sys.path.insert(0, os.path.abspath("../../"))
pysplashsurf = import_module("pysplashsurf")

#import pysplashsurf

# -- Project information -----------------------------------------------------

project = "pySplashsurf"
copyright = "2025, Interactive Computer Graphics"
author = "Interactive Computer Graphics"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "numpydoc",
    "myst_parser",
    "sphinx_rtd_theme",
    #'sphinx_autodoc_typehints'
]

source_suffix = [".rst", ".md"]

numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

autodoc_typehints = "both"

always_document_param_types = True
always_use_bars_union = True

napoleon_use_rtype = False
napoleon_include_special_with_doc = True

# typehints_document_rtype = False
# typehints_use_rtype = False
# typehints_use_signature = True
# typehints_use_signature_return = True
