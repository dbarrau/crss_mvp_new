"""Shared hierarchy level ordering for regulation parsers.

Defines a common numeric ranking for structural elements used by
parsers to manage stack-based hierarchy logic. Keep this synchronized
across all parsers by importing `LEVEL_ORDER` from here.
"""
from __future__ import annotations

LEVEL_ORDER = {
    "title": 0,        # Top‑level document title
    "recital": 1,      # Preamble recitals
    "chapter": 2,      # Chapters in the enacting part
    "section": 3,      # Sections inside chapters
    "article": 4,      # Articles under sections or chapters
    "paragraph": 5,    # Numbered paragraphs inside articles
    "letter": 6,       # letter item within paragraphs (e.g., (a), (b))
    "subpoint": 7,     # Further nested levels (e.g., (i), (ii) under letters)
    "annex": 0,        # Annexes treated as top-level like title
}
