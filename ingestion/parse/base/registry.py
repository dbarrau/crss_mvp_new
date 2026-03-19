"""
Parser Registry Module (moved to parse/base)
"""

from domain.regulations_catalog import REGULATIONS
from ..universal_eurlex_parser import parse_eurlex_html

# All currently supported regulations use the universal parser.
# To assign a custom parser to a regulation, add a "parser" key to its
# entry in domain/regulations_catalog.py and dispatch on it here.
PARSER_REGISTRY = {celex: parse_eurlex_html for celex in REGULATIONS}
