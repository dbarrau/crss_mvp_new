"""
Parser Registry Module (moved to parse/base)
"""

from ..universal_eurlex_parser import parse_eurlex_html

PARSER_REGISTRY = {
    "32017R0745": parse_eurlex_html,
    "32024R1689": parse_eurlex_html,
}
