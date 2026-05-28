"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Climb up two directory levels to map the absolute path to your 'src' folder
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "RetailGenius customer churn prediction"
current_year = datetime.now().year
copyright = f"{current_year}, Krishanth Dev, Thin Sapal Yu, Duy Thai, George Ebaidallah"
author = "Krishanth Dev, Thin Sapal Yu, Duy Thai, George Ebaidallah"

# -- General configuration ---------------------------------------------------
# Core extensions to auto-extract docstrings and read NumPy layouts
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Adds interactive links to your source code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# Overrides 'alabaster' with the production-standard clean layout theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]