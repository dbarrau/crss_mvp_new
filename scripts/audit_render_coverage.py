#!/usr/bin/env python
"""Audit which articles/annexes would render EMPTY on a "show me X" request.

The verbatim display renders a provision's ordered HAS_PART subtree. If a top
provision (article or annex) has no children and no real body text, the display
returns nothing and the request falls through to GENERATION — which reconstructs
the provision from the model's memory (a fabrication risk on a legal tool).

This sweep flags every such provision across all loaded regulations, so empty /
mis-parsed provisions are found systematically instead of one report at a time.

    python scripts/audit_render_coverage.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_QUERY = """
MATCH (n)
WHERE (n:Article OR n:Annex) AND n.celex IS NOT NULL
OPTIONAL MATCH (n)-[:HAS_PART]->(c)
WITH n, count(c) AS kids
WHERE kids = 0 AND coalesce(size(n.text), 0) < $min_chars
RETURN n.celex AS celex,
       CASE WHEN n:Annex THEN 'Annex' ELSE 'Article' END AS kind,
       n.display_ref AS ref, n.number AS number, coalesce(size(n.text), 0) AS chars
ORDER BY celex, kind, number
"""


def main(min_chars: int = 120) -> int:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword"))
    with GraphDatabase.driver(uri, auth=auth) as drv, drv.session() as s:
        rows = s.run(_QUERY, min_chars=min_chars).data()

    if not rows:
        print("✓ No empty articles/annexes — every 'show me X' has real text to render.")
        return 0

    by_celex: dict[str, list] = {}
    for r in rows:
        by_celex.setdefault(r["celex"], []).append(r)

    print(f"⚠ {len(rows)} provision(s) would render EMPTY (fall through to generation):\n")
    for celex, items in by_celex.items():
        labels = ", ".join(f"{r['ref'] or (r['kind'] + ' ' + str(r['number']))}"
                            f" ({r['chars']}c)" for r in items)
        print(f"  {celex}:  {labels}")
    print("\nThese have only a title/heading in the graph — their body was not parsed.")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
