#!/usr/bin/env python3
"""Quick smoke test for the retriever + agent pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from retrieval.graph_retriever import GraphRetriever
from application.agent import ask

print("=== Test 1: Retriever ===")
retriever = GraphRetriever()

results = retriever.retrieve(
    "What are the requirements for technical documentation under MDR?", k=3
)
print(f"Retrieved {len(results)} provisions")
for r in results:
    ref = r.get("article_ref", "?")
    reg = r.get("regulation", "")
    score = r.get("score", 0)
    n_children = len(r.get("children") or [])
    print(f"  {ref} ({reg})  score={score:.3f}  children={n_children}")

print()
print("=== Test 2: Full Agent (Mistral) ===")
answer = ask(
    "What are the post-market surveillance obligations under MDR Article 83?",
    retriever,
    k=5,
)
print(f"Answer ({len(answer)} chars):")
print(answer[:2000])

retriever.close()
print()
print("=== ALL TESTS PASSED ===")
