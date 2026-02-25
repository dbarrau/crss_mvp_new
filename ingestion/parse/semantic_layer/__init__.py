"""Semantic parsing helpers for the parsing pipeline.

This package is intended to host small, parser-level semantic detectors
such as requirement/obligation pattern matchers.
"""

from .requirement_patterns import is_requirement_text, classify_requirement_type  # noqa: F401
