# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'dPBE: Discrete Population Balance Equations'
copyright = '2024, Frank Rhein, Haoran Ji'
author = 'Frank Rhein, Haoran Ji'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", 
              "sphinx.ext.viewcode", 
              "sphinx.ext.autodoc",
              "myst_parser"]

myst_enable_extensions = ["dollarmath", "amsmath"]

#templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Include all external packages (should not be included in the docs)
autodoc_mock_imports = ['matplotlib', 'scipy', 'numba', 'plotter','statsmodels',
                        'bayes_opt', 'sklearn', 'pandas', 'SALib', 'numdifftools']
autodoc_default_options = {'member-order': 'bysource'}
nitpick_ignore = [
    ("py:class", "ray.tune.trainable.trainable.Trainable"),
    ("py:class", "ray.tune.trainable.Trainable"),
    ("py:class", "libcst.CSTTransformer"),
    ("py:class", "libcst._visitors.CSTTransformer")
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
