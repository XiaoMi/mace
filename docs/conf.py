# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#

import recommonmark.parser
import sphinx_rtd_theme

project = u'MACE'
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
smartquotes = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
]

latex_elements = {
    # Additional stuff for the LaTeX preamble, to generate Chinese correctly.
    'preamble':
    r'''
        \hypersetup{unicode=true}
        \usepackage{CJKutf8}
        \DeclareUnicodeCharacter{00A0}{\nobreakspace}
        \DeclareUnicodeCharacter{2203}{\ensuremath{\exists}}
        \DeclareUnicodeCharacter{2200}{\ensuremath{\forall}}
        \DeclareUnicodeCharacter{2286}{\ensuremath{\subseteq}}
        \DeclareUnicodeCharacter{2713}{x}
        \DeclareUnicodeCharacter{27FA}{\ensuremath{\Longleftrightarrow}}
        \DeclareUnicodeCharacter{221A}{\ensuremath{\sqrt{}}}
        \DeclareUnicodeCharacter{221B}{\ensuremath{\sqrt[3]{}}}
        \DeclareUnicodeCharacter{2295}{\ensuremath{\oplus}}
        \DeclareUnicodeCharacter{2297}{\ensuremath{\otimes}}
        \begin{CJK}{UTF8}{gbsn}
        \AtEndDocument{\end{CJK}}
        ''',
}
