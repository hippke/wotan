import os
import sys
#sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./wotan/'))
#sys.path.insert(0, os.path.abspath('../wotan/'))

html_show_sourcelink = False
project = 'wotan'
copyright = '2019, Michael Hippke'
author = 'Michael Hippke'

version = '1'
release = '1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
]
source_suffix = '.rst'

# Napoleon settings
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

# The master toctree document.
master_doc = 'index'
language = None
exclude_patterns = []
pygments_style = None
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
htmlhelp_basename = 'wotan_doc'
man_pages = [
    (master_doc, 'wotan', 'wotan Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'wotan', 'wotan Documentation',
     author, 'wotan', 'One line description of project.',
     'Miscellaneous'),
]
