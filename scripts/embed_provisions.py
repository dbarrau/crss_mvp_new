#!/usr/bin/env python3
"""Embed all provisions and store vectors in Neo4j.

Quick start::

    python scripts/embed_provisions.py
"""
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

n = run()
print(f"\nEmbedded {n} provisions.")
