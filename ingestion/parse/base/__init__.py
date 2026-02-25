"""Base parsing utilities package

Exports core helper modules for regulation parsers. This module keeps
backwards compatibility by re-exporting small helpers from the
`semantic_layer` package.
"""
from ..semantic_layer.requirement_patterns import *  # noqa: F401,F403
## Legacy lang_keywords import removed; handled by universal parser
