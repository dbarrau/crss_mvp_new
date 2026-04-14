"""
Parser Registry Module (moved to parse/base)
"""

from domain.legislation_catalog import LEGISLATION
from ..universal_eurlex_parser import parse_eurlex_html

# All currently supported legislation uses the universal parser.
# To assign a custom parser to a legal act, add a "parser" key to its
# entry in domain/legislation_catalog.py and dispatch on it here.
PARSER_REGISTRY = {celex: parse_eurlex_html for celex in LEGISLATION}
