from datetime import date
import sys
import os

sys.path.insert(0, os.path.abspath("../../"))

from reeffit import __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
]
templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'REEFFIT'
copyright = u'2013-%s Pablo Cordero G., Leland Stanford Junior University. All Rights Reserved' % date.today().year
author = u'Pablo Cordero, Rhiju Das'
language = 'en'

version = __version__
release = version


pygments_style = 'sphinx'
add_function_parentheses = True
add_module_names = True
show_authors = False
todo_include_todos = True
man_show_urls = True
modindex_common_prefix = []

html_domain_indices = True
html_use_smartypants = False
html_use_index = True
html_use_modindex = False
html_split_index = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True
html_copy_source = True
html_compact_lists = True
html_last_updated_fmt = '%b %d, %Y'
html_search_language = 'en'

html_title = project + ' ' + version + ' : RNA Ensemble Extraction From Footprinting Insights Technique'
html_short_title = project + ' Documentation'
html_logo = None
html_extra_path = []
html_favicon = html_static_path[0] + '/ribokit.gif'

html_sidebars = {}
html_context = {}
html_style = []
html_theme = 'ribokit-Sphinx-theme'
html_theme_path = ['_theme']
html_theme_options = {
    'description': '<u>R</u>NA <u>E</u>nsemble <u>E</u>xtraction <u>F</u>rom <u>F</u>ootprinting <u>I</u>nsights <u>T</u>echnique',
    'author': 'Siqi Tian',
    'github_repo': 'ribokit/REEFFIT',
    'ga_tracker': 'UA-36037648-4'
}
html_additional_pages = {'404': '404.html'}

# Output file base name for HTML help builder.
htmlhelp_basename = 'REEFFIT_doc'


napoleon_google_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_rtype = False
autodoc_member_order = 'groupwise'
autodoc_default_flags = ['members', 'show-inheritance']
autosummary_generate = True

