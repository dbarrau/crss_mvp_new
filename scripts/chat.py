#!/usr/bin/env python3
"""Interactive CLI to chat with the EU regulation knowledge graph.

Usage::

    python scripts/chat.py

Commands inside the chat:
    quit   — exit
    debug  — toggle display of retrieved provisions before the answer
    k=N    — change number of retrieved provisions (default 5)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

logging.basicConfig(level=logging.WARNING)

from retrieval.graph_retriever import GraphRetriever
from application.agent import ask


def main() -> None:
    print("╔══════════════════════════════════════════════╗")
    print("║       CRSS — Regulatory Compliance Agent     ║")
    print("║  Neo4j graph · multilingual-e5 · Mistral     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("Commands:  quit · debug · k=N")
    print()

    retriever = GraphRetriever()
    show_debug = False
    k = 5

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                break
            if question.lower() == "debug":
                show_debug = not show_debug
                print(f"  [Debug mode: {'ON' if show_debug else 'OFF'}]")
                continue
            if question.lower().startswith("k="):
                try:
                    k = int(question.split("=", 1)[1])
                    print(f"  [Retrieval k set to {k}]")
                except ValueError:
                    print("  [Invalid k value]")
                continue

            if show_debug:
                provisions = retriever.retrieve(question, k=k)
                print(f"\n  [{len(provisions)} provisions retrieved]")
                for p in provisions:
                    score = p.get("score", 0)
                    ref = p.get("article_ref", "?")
                    reg = p.get("regulation", "")
                    n_children = len(p.get("children") or [])
                    n_cited = len(p.get("cited_provisions") or [])
                    print(
                        f"    {ref} ({reg})  "
                        f"score={score:.3f}  "
                        f"children={n_children}  cited={n_cited}"
                    )
                print()

            answer = ask(question, retriever, k=k)
            print(f"\nAgent: {answer}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
