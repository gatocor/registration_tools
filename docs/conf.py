import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Registration Tools'
author = 'Gabriel Torregrosa Cortés'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # Add this line to support Markdown files
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']

# Add the new Markdown file to the toctree
master_doc = 'index'
