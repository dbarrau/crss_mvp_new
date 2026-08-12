"""
canonicalization/amendment_linker.py
=====================================
Materialise **AMENDS** edges from an amending act's provisions to the
provisions they amend in a *loaded* target regulation.

Why this stage exists
---------------------
EU amending regulations express changes as::

    Article 1
    Regulation (EU) 2024/1689 is amended as follows:
      (1) ...
      (40) in Article 113, the third paragraph is amended as follows:
             (a) point (a) is replaced by the following: '...'
             (b) point (c) is replaced by the following: '...'

The target regulation is named **once**, on the container article ("Regulation
(EU) 2024/1689 is amended as follows"); each point names only "Article N".  The
generic cross-reference extractor therefore cannot bind the points to the target
— a bare "Article 113" resolves (if at all) to the *amending* act's own Article
113, not the AI Act's.  So the amendment relationship — the one relationship that
changes what the law **is** — is invisible in the graph: only a generic
article-level ``CITES_EXTERNAL`` to "Regulation (EU) 2024/1689" survives.

This stage recovers it::

    (amending point)-[:AMENDS {amending_act, operation, target_ref}]->(target Article)

so retrieval can surface "Article 113 is amended by → [new text]" whenever the
base provision is pulled — instead of the base provision's stale pre-amendment
text standing alone (the failure the Digital Omnibus date bug root-caused to).

The amending point's own subtree carries the operative replacement text (see the
enacting-terms parser's quoted-replacement handling), so the edge plus a HAS_PART
expansion is all retrieval needs.

Scope / safety
--------------
* Container = any article whose text says "Regulation (EU) N ... is amended",
  where N resolves (via the catalog) to a **loaded** regulation other than the
  amending act itself.  Amendments to acts we do not hold are skipped.
* Source  = a descendant provision whose text opens with an amendment
  instruction naming an "Article N" / "Annex R" target AND contains an operation
  verb (replaced / inserted / deleted / amended / added).  Both guards must hold,
  so an incidental cross-reference ("in accordance with Article 6") is not linked.
* Target  = ``{amended_celex}_art_{N}`` / ``{amended_celex}_anx_{R}`` — linked
  only when that node exists.
* Idempotent: clears its own edges (``resolved_from='amendment_linker'``) first.

Runs **after** the loader (needs HAS_PART) — wired into ``python -m
canonicalization`` right after the crosslinker.  Standalone::

    python -m canonicalization.amendment_linker
    python -m canonicalization.amendment_linker --dry-run
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from domain.legislation_catalog import LEGISLATION
from canonicalization.crosslinker import CELEX_BY_NUMBER, build_target_id
from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)

_BATCH = 500

# ---------------------------------------------------------------------------
# Amendment-instruction language
# ---------------------------------------------------------------------------

# Container article: "Regulation (EU) 2024/1689 is amended as follows:".
_CONTAINER_RE = re.compile(
    r"Regulation\s*\(EU\)\s*(?P<num>\d+/\d+).{0,80}?\bis\s+amended\b",
    re.IGNORECASE | re.DOTALL,
)

# An amendment instruction opens the point and names its target: "in Article 113,
# …", "Article 6 is replaced by …", "the following Article 6a is inserted".
_TARGET_RE = re.compile(
    r"^\s*(?:in\s+|the\s+following\s+)?(?P<kind>Article|Annex)\s+"
    r"(?P<num>\d+[a-z]?|[IVXLC]+)\b",
    re.IGNORECASE,
)

# The operation verb must be present too (belt-and-suspenders against an
# incidental cross-reference that merely opens with "Article N …").
_OP_RE = re.compile(
    r"\b(replaced|inserted|deleted|amended|added|repealed)\b", re.IGNORECASE
)


def _clean(text: str | None) -> str:
    """Fold the non-breaking spaces EUR-Lex uses so the regexes see plain gaps."""
    return (text or "").replace(" ", " ")


def _amended_celex_of(container_text: str, container_celex: str) -> str | None:
    """The loaded CELEX a container article amends, or ``None`` to skip."""
    m = _CONTAINER_RE.search(_clean(container_text))
    if not m:
        return None
    target_celex = CELEX_BY_NUMBER.get(m.group("num"))
    if target_celex is None or target_celex == container_celex:
        return None  # act not held, or self-reference
    return target_celex


def _target_id_of(point_text: str, amended_celex: str) -> tuple[str, str, str] | None:
    """``(target_id, target_ref, operation)`` for one amending point, else ``None``."""
    text = _clean(point_text)
    tm = _TARGET_RE.match(text)
    if not tm:
        return None
    op = _OP_RE.search(text)
    if not op:
        return None
    kind = tm.group("kind").lower()
    num = tm.group("num")
    parts = {"annex": num} if kind == "annex" else {"article": num}
    target_id = build_target_id(amended_celex, parts)
    if not target_id:
        return None
    target_ref = f"{tm.group('kind').capitalize()} {num}"
    return target_id, target_ref, op.group(1).lower()


# ---------------------------------------------------------------------------
# Neo4j discovery + write
# ---------------------------------------------------------------------------

def _discover_edges(session) -> list[dict[str, Any]]:
    """Find every (amending point → amended article) pair to link."""
    containers = session.run(
        "MATCH (art:Provision) "
        "WHERE art.kind = 'article' AND toLower(art.text) CONTAINS 'amended as follows' "
        "RETURN art.id AS id, art.celex AS celex, art.text AS text"
    ).data()

    edges: list[dict[str, Any]] = []
    for c in containers:
        amended_celex = _amended_celex_of(c["text"], c["celex"])
        if amended_celex is None:
            continue
        amending_meta = LEGISLATION.get(c["celex"], {})
        amending_act = f"Regulation (EU) {amending_meta.get('number', c['celex'])}"

        descendants = session.run(
            "MATCH (:Provision {id: $id})-[:HAS_PART*1..6]->(pt:Provision) "
            "RETURN pt.id AS id, pt.text AS text",
            id=c["id"],
        ).data()
        for d in descendants:
            resolved = _target_id_of(d["text"] or "", amended_celex)
            if not resolved:
                continue
            target_id, target_ref, operation = resolved
            edges.append({
                "source": d["id"],
                "target": target_id,
                "amending_act": amending_act,
                "amending_celex": c["celex"],
                "operation": operation,
                "target_ref": target_ref,
            })
    return edges


def _write_edges(session, edges: list[dict]) -> int:
    """MERGE AMENDS edges between existing Provision nodes (skips missing targets)."""
    if not edges:
        return 0
    cypher = (
        "UNWIND $batch AS e "
        "MATCH (s:Provision {id: e.source}) "
        "MATCH (t:Provision {id: e.target}) "
        "MERGE (s)-[r:AMENDS {resolved_from: 'amendment_linker'}]->(t) "
        "SET r.amending_act = e.amending_act, "
        "    r.amending_celex = e.amending_celex, "
        "    r.operation = e.operation, "
        "    r.target_ref = e.target_ref "
        "RETURN count(r) AS c"
    )
    total = 0
    for i in range(0, len(edges), _BATCH):
        chunk = edges[i : i + _BATCH]
        total += session.run(cypher, batch=chunk).single()["c"]
    return total


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def link_amendments(dry_run: bool = False) -> dict[str, int]:
    """Main entry point. Returns summary counts."""
    load_dotenv()

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    written = 0
    try:
        with driver.session(database=database) as session:
            if not dry_run:
                reset = session.run(
                    "MATCH ()-[r:AMENDS {resolved_from: 'amendment_linker'}]->() "
                    "DELETE r RETURN count(r) AS c"
                ).single()["c"]
                if reset:
                    logger.info("Cleared %d stale AMENDS edges.", reset)

            edges = _discover_edges(session)
            if not dry_run and edges:
                written = _write_edges(session, edges)
    finally:
        driver.close()

    # Distinct amended targets, for the summary line.
    targets = sorted({e["target"] for e in edges})
    summary = {
        "amendment_edges_detected": len(edges),
        "edges_written": written if not dry_run else 0,
        "distinct_targets": len(targets),
    }

    print("\n=== Amendment Linker Summary ===")
    print(f"  {'AMENDS edges detected:':<34} {summary['amendment_edges_detected']:>4}")
    print(f"  {'Distinct amended provisions:':<34} {summary['distinct_targets']:>4}")
    print(f"  {'AMENDS edges written:':<34} {summary['edges_written']:>4}")
    if dry_run:
        print("  (dry run — no changes written)")
    for e in edges[:12]:
        print(f"    {e['source']}  --AMENDS ({e['operation']})-->  {e['target']}")
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise AMENDS edges (amending provision → amended provision)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    link_amendments(dry_run=args.dry_run)
