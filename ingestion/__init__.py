"""CRSS ingestion package.

Contains CLI entrypoints, scraping helpers and parser runners used to
produce structured `graph.json` artifacts from EUR-Lex HTML.
"""

from . import parse, scrape

__all__ = ["parse", "scrape"]
