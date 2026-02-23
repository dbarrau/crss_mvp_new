"""Parsing package public API for ingestion.parse.

Re-exports :func:`parse_document` from :mod:`ingestion.parse.dispatcher`
so callers can simply ``from ingestion.parse import parse_document``
without depending on the internal module layout.
"""

from .dispatcher import parse_document  # noqa: F401
