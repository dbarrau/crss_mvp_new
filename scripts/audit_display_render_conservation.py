#!/usr/bin/env python
"""Audit that "show me Article/Annex X" RENDERS everything the graph holds.

The three sibling sweeps prove the *graph* is complete:

  * audit_render_coverage.py      — no provision renders fully EMPTY.
  * audit_text_conservation.py    — raw HTML content survives into parsed.json.
  * audit_orphan_subparagraphs.py — no trailing <p> dropped at parse time.

None of them exercises the actual DISPLAY RENDER. "Show me Article X" goes through
``application/_display.render_provision_display`` — retrieve the node's ordered
HAS_PART subtree, then render it. Historic bugs lived exactly HERE, downstream of
a complete graph: a 12-child cap, a scrambled ``collect()``, dropped paragraph
numbers. This sweep closes that last link end-to-end.

Method (runtime, no LLM): for every article and annex in every loaded regulation,
call the real ``render_provision_display`` and compare its output against the
node's full HAS_PART subtree text taken DIRECTLY from Neo4j. The comparison is an
order-independent multiset difference of meaningful content tokens (mirrors
audit_text_conservation) — only tokens present in the graph subtree but ABSENT
from the render are flagged, so the render's own boilerplate (title, "verbatim
source text", amendment footer) never trips it and reordering is ignored.

Because deletions are removed from the consolidated graph and replacements carry
their new text, the subtree is current law — no husk false positives.

    python scripts/audit_display_render_conservation.py
    python scripts/audit_display_render_conservation.py --min-absent 4 --docs 32024R1689
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from application._display import render_provision_display  # noqa: E402
from retrieval.graph_retriever import GraphRetriever  # noqa: E402

# Same tokenizer contract as audit_text_conservation.py (kept in sync on purpose).
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the of to a and or in for on by with that this as be is are shall may not "
    "an at from which such it its any other article annex".split()
)


def _ctoks(text: str) -> list[str]:
    return [w for w in _TOK.findall((text or "").lower()) if w not in _STOP]


def _subjects(session, celex: str) -> list[tuple[str, str]]:
    """(node_id, display_ref) for every article/annex in *celex*."""
    rows = session.run(
        "MATCH (a:Provision {celex:$celex}) WHERE a.kind IN ['article','annex'] "
        "RETURN a.id AS id, coalesce(a.display_ref, a.name) AS ref "
        "ORDER BY a.id",
        celex=celex,
    )
    return [(r["id"], r["ref"]) for r in rows if r["ref"]]


def _subtree_text(session, node_id: str) -> str:
    """Concatenated own-text of every HAS_PART descendant (incl. the root)."""
    row = session.run(
        "MATCH (a:Provision {id:$id})-[:HAS_PART*0..]->(d) "
        "RETURN collect(d.text) AS texts",
        id=node_id,
    ).single()
    return " ".join(t for t in (row["texts"] if row else []) if t)


def _loaded_celexes(session, only: list[str] | None) -> list[str]:
    rows = session.run(
        "MATCH (a:Provision) WHERE a.kind IN ['article','annex'] "
        "RETURN DISTINCT a.celex AS celex ORDER BY celex"
    )
    celexes = [r["celex"] for r in rows if r["celex"]]
    return [c for c in celexes if not only or c in only]


def main(min_absent: int, only: list[str] | None) -> int:
    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    print("Loading retriever (embeddings + Neo4j)…")
    retriever = GraphRetriever()
    driver = GraphDatabase.driver(uri, auth=auth)

    total = 0
    empties: list[str] = []          # render returned None despite a subtree
    lossy: list[tuple[int, str, str, str]] = []   # (n_absent, celex, ref, span)
    try:
        with driver.session() as session:
            celexes = _loaded_celexes(session, only)
            for celex in celexes:
                subs = _subjects(session, celex)
                print(f"  {celex}: checking {len(subs)} article/annex render(s)…")
                for node_id, ref in subs:
                    total += 1
                    truth = _subtree_text(session, node_id)
                    if not _ctoks(truth):
                        continue  # structural-only node with no content to conserve
                    rendered = render_provision_display(retriever, ref, celex)
                    if not rendered:
                        empties.append(f"{celex} {ref} ({node_id})")
                        continue
                    absent = Counter(_ctoks(truth)) - Counter(_ctoks(rendered))
                    n_absent = sum(absent.values())
                    if n_absent >= min_absent:
                        span = " ".join(list(absent.elements())[:14])
                        lossy.append((n_absent, celex, ref, span))
    finally:
        driver.close()

    print(f"\nChecked {total} article/annex render(s) across {len(celexes)} regulation(s).")

    if empties:
        print(f"\n✗ {len(empties)} provision(s) render EMPTY despite having a subtree "
              "(would fall through to generation):")
        for e in empties[:40]:
            print(f"    {e}")

    lossy.sort(reverse=True)
    if lossy:
        print(f"\n✗ {len(lossy)} render(s) DROP graph content (>= {min_absent} tokens absent):")
        for n_absent, celex, ref, span in lossy[:40]:
            print(f"    {celex} {ref}: {n_absent} token(s) absent — e.g. {span[:110]!r}")
    if not empties and not lossy:
        print("\n✓ Every 'show me X' render reproduces its full graph subtree — "
              "no dropped or empty provisions across the corpus.")
        return 0
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-absent", type=int, default=4,
                    help="Min absent meaningful-token occurrences to flag a render (default 4).")
    ap.add_argument("--docs", nargs="*", default=None,
                    help="Restrict to these CELEX ids (default: every loaded regulation).")
    raise SystemExit(main(ap.parse_args().min_absent, ap.parse_args().docs))
