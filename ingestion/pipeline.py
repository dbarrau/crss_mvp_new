"""Backward-compatible wrapper around :mod:`ingestion.run_pipeline`."""
from __future__ import annotations

from .run_pipeline import DEFAULT_DOC, DEFAULT_CELEX, DEFAULT_LANG, run

__all__ = ["run", "DEFAULT_DOC", "DEFAULT_CELEX", "DEFAULT_LANG"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=DEFAULT_DOC)
    # Deprecated alias kept for backward compatibility
    parser.add_argument("--celex", dest="doc", help=argparse.SUPPRESS)
    parser.add_argument("--lang", default=DEFAULT_LANG)
    args = parser.parse_args()

    run(args.doc, args.lang)
