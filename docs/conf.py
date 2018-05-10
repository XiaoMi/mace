# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#

import recommonmark.parser
import sphinx_rtd_theme

project = u'MiAI Compute Engine'
author = u'%s Developers' % project
copyright = u'2018, %s' % author

source_parsers = {
    '.md': recommonmark.parser.CommonMarkParser,
}
source_suffix = ['.rst', '.md']

master_doc = 'index'

exclude_patterns = [u'_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
