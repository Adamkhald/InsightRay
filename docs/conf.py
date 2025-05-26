# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Insight Ray'
copyright = '2025, Adam KHALD'
author = 'Adam KHALD'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    'myst_parser',  # For Markdown support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'analytics_id': '',  # Provided by Google Analytics
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# The master toctree document.
master_doc = 'index'

# Logo and favicon
html_logo = '_static/logo.png'  # Add your logo here
html_favicon = '_static/favicon.ico'  # Add your favicon here

# Project's home page
html_context = {
    "display_github": True,
    "github_user": "your-username",  # Replace with your GitHub username
    "github_repo": "insight-ray",    # Replace with your repository name
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
