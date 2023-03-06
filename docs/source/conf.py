# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DataCheese'
copyright = '2023, Zahin Zaman'
author = 'Zahin Zaman'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# preserve default argument values as in source code
autodoc_preserve_defaults = True

# don't show type hints
autodoc_typehints = 'none'

# order documentation by definition in source code
autodoc_member_order = 'bysource'

# -- Options for PyData theme ------------------------------------------------

html_logo = '../img/logo_full.png'

html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/alvii147/DataCheese',
            'icon': 'fab fa-github-square',
            'type': 'fontawesome',
        },
    ],
    'external_links': [
        {
            'name': 'Donate to Turkey & Syria',
            'url': 'https://www.islamicreliefcanada.org/emergencies/turkey-syria-earthquake-appeal/',
        },
    ],
    'show_nav_level': 1,
    'navigation_depth': 2,
    'collapse_navigation': False,
    'show_prev_next': True,
    'use_edit_page_button': False,
}
