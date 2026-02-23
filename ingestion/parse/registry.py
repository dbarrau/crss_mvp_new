"""
Parser Registry Module
======================

This module serves as the central dispatching hub for the CRSS pipeline.
It maps official European Union CELEX identifiers to their respective
parsing implementation functions.

The Registry pattern is utilized here to ensure the pipeline remains
decoupled; the core execution engine does not need to know the internal
logic of a parser, only its associated identifier.

Attributes:
    PARSER_REGISTRY (dict): A mapping where keys are CELEX IDs (str)
        and values are callable parsing functions.
"""

## Legacy parser imports removed; only universal parser is used

from .universal_eurlex_parser import parse_eurlex_html

PARSER_REGISTRY = {
    "32017R0745": parse_eurlex_html,
    "32024R1689": parse_eurlex_html,
}
