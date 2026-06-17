#!/usr/bin/env python3
"""Embed provisions and store vectors in Neo4j.

Quick start::

    python scripts/embed_provisions.py                  # embed everything
    python scripts/embed_provisions.py --doc 32026R0977 # single document
    python scripts/embed_provisions.py --doc 32017R0745 32017R0746  # multiple
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from infrastructure.embeddings.batch_embedder import run

parser = argparse.ArgumentParser(description="Embed provisions into Neo4j.")
parser.add_argument(
    "--doc",
    nargs="+",
    metavar="CELEX",
    help="One or more CELEX IDs to embed (default: all documents).",
)
args = parser.parse_args()

n = run(celex_filter=args.doc)
print(f"\nEmbedded {n} provisions.")
