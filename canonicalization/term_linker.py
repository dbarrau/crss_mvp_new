"""
canonicalization/term_linker.py
===============================
Materialise **USES_TERM** edges between :Provision/:Guidance nodes and
:DefinedTerm nodes in Neo4j.

For every provision whose ``text`` contains a defined term (word-boundary
match, case-insensitive), a ``(provision)-[:USES_TERM]->(DefinedTerm)``
edge is created.  The defining provision itself is excluded (that link is
already captured by DEFINED_BY).

This is a post-processing step that should run **after** all documents are
loaded and DefinedTerm nodes exist.  It is idempotent: re-running it
deletes previous USES_TERM edges first.

Usage::

    python -m canonicalization.term_linker          # reads .env for Neo4j creds
    python -m canonicalization.term_linker --dry-run # preview without writing
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)

# Batch size for UNWIND queries
_BATCH = 500

# Single-word terms that are too generic for useful USES_TERM edges.
# These are real defined terms but appear in almost every provision,
# drowning out the signal from more specific multi-word terms.
_SKIP_TERMS: set[str] = {
    "risk", "system", "user", "label", "recall", "subject",
    "kit", "performance",
}


# ---------------------------------------------------------------------------
# 1. Fetch DefinedTerm index from Neo4j
# ---------------------------------------------------------------------------

def _load_terms(session) -> list[dict[str, Any]]:
    """Return all DefinedTerm nodes with their defining provision ID.

    Excludes terms in the ``_SKIP_TERMS`` stoplist (too generic).
    """
    rows = session.run(
        "MATCH (d:DefinedTerm) "
        "RETURN d.id AS id, d.term AS term, "
        "       d.term_normalized AS tn, "
        "       d.source_provision_id AS src_prov_id, "
        "       d.celex AS celex"
    ).data()
    return [r for r in rows if r["term"].lower() not in _SKIP_TERMS]


# ---------------------------------------------------------------------------
# 2. Fetch all provision text from Neo4j
# ---------------------------------------------------------------------------

def _load_provisions(session) -> list[dict[str, str]]:
    """Return id + text for every Provision and Guidance node."""
    rows = session.run(
        "MATCH (n:Provision) "
        "WHERE n.text IS NOT NULL AND n.text <> '' "
        "RETURN n.id AS id, n.text AS text "
        "UNION ALL "
        "MATCH (n:Guidance) "
        "WHERE n.text IS NOT NULL AND n.text <> '' "
        "RETURN n.id AS id, n.text AS text"
    ).data()
    return rows


# ---------------------------------------------------------------------------
# 3. Build term matcher
# ---------------------------------------------------------------------------

def _build_term_regex(terms: list[dict[str, Any]]) -> re.Pattern | None:
    """Compile a single regex alternation matching all defined terms.

    Terms are sorted longest-first so "high-risk AI system" matches before
    "AI system".  Word boundaries are enforced on both sides.
    """
    if not terms:
        return None

    # Deduplicate by lowercase term
    seen: set[str] = set()
    unique_terms: list[str] = []
    for t in sorted(terms, key=lambda x: len(x["term"]), reverse=True):
        lower = t["term"].lower()
        if lower not in seen:
            seen.add(lower)
            unique_terms.append(t["term"])

    # Build alternation with word boundaries
    escaped = [re.escape(t) for t in unique_terms]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# 4. Match terms against provisions
# ---------------------------------------------------------------------------

def _find_uses(
    provisions: list[dict[str, str]],
    terms: list[dict[str, Any]],
    term_regex: re.Pattern,
) -> list[dict[str, str]]:
    """Scan every provision's text and return (provision_id, term_id) pairs.

    Excludes the defining provision itself (the one DEFINED_BY already links).
    """
    # Build lookup: lowercase term → list of DefinedTerm IDs
    term_lookup: dict[str, list[dict[str, str]]] = {}
    for t in terms:
        lower = t["term"].lower()
        term_lookup.setdefault(lower, []).append({
            "id": t["id"],
            "src_prov_id": t["src_prov_id"],
        })

    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for prov in provisions:
        prov_id = prov["id"]
        text = prov["text"]

        matches = term_regex.findall(text)
        if not matches:
            continue

        # Deduplicate matches within this provision
        matched_terms: set[str] = set()
        for m in matches:
            matched_terms.add(m.lower())

        for term_lower in matched_terms:
            for dt in term_lookup.get(term_lower, []):
                # Skip the defining provision itself
                if prov_id == dt["src_prov_id"]:
                    continue

                pair = (prov_id, dt["id"])
                if pair not in seen:
                    seen.add(pair)
                    edges.append({
                        "prov_id": prov_id,
                        "term_id": dt["id"],
                    })

    return edges


# ---------------------------------------------------------------------------
# 5. Write USES_TERM edges
# ---------------------------------------------------------------------------

def _write_edges(session, edges: list[dict[str, str]]) -> int:
    """MERGE USES_TERM edges between Provision/Guidance nodes and DefinedTerm nodes."""
    if not edges:
        return 0

    total = 0
    cypher = (
        "UNWIND $batch AS e "
        "OPTIONAL MATCH (p1:Provision {id: e.prov_id}) "
        "OPTIONAL MATCH (p2:Guidance  {id: e.prov_id}) "
        "WITH e, coalesce(p1, p2) AS prov "
        "WHERE prov IS NOT NULL "
        "MATCH (d:DefinedTerm {id: e.term_id}) "
        "MERGE (prov)-[r:USES_TERM]->(d) "
        "RETURN count(r) AS c"
    )
    for i in range(0, len(edges), _BATCH):
        chunk = edges[i : i + _BATCH]
        result = session.run(cypher, batch=chunk)
        total += result.single()["c"]
    return total


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def link_terms(dry_run: bool = False) -> dict[str, int]:
    """Main entry point. Returns summary counts."""
    load_dotenv()

    uri = _normalize_neo4j_uri(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    user = os.environ.get(
        "NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")
    )
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            # 1. Load data
            terms = _load_terms(session)
            provisions = _load_provisions(session)
            logger.info(
                "Loaded %d DefinedTerms and %d provisions.",
                len(terms), len(provisions),
            )

            if not terms:
                logger.warning("No DefinedTerms found — nothing to link.")
                return {"terms": 0, "provisions": 0, "edges": 0}

            # 2. Build matcher
            term_regex = _build_term_regex(terms)
            if term_regex is None:
                return {"terms": 0, "provisions": 0, "edges": 0}

            # 3. Find matches
            edges = _find_uses(provisions, terms, term_regex)
            logger.info("Found %d USES_TERM candidate edges.", len(edges))

            # 4. Count provisions that use at least one term
            prov_ids_with_terms = {e["prov_id"] for e in edges}

            if dry_run:
                print(f"\n=== Term Linker (dry run) ===")
                print(f"  DefinedTerms:             {len(terms)}")
                print(f"  Provisions scanned:       {len(provisions)}")
                print(f"  Provisions with terms:    {len(prov_ids_with_terms)}")
                print(f"  USES_TERM edges (would write): {len(edges)}")

                # Show top terms by usage count (group by term name)
                from collections import Counter
                # Map term_id → term name, then count by name
                id_to_name = {t["id"]: t["term"] for t in terms}
                name_counts: Counter = Counter()
                for e in edges:
                    name_counts[id_to_name.get(e["term_id"], e["term_id"])] += 1
                print(f"\n  Top 15 most-referenced terms:")
                for term_name, count in name_counts.most_common(15):
                    print(f"    {term_name:<40s} {count:>5} provisions")
                print("  (dry run — no changes written)\n")
                return {
                    "terms": len(terms),
                    "provisions": len(provisions),
                    "provisions_with_terms": len(prov_ids_with_terms),
                    "edges": 0,
                }

            # 5. Clear previous USES_TERM edges (idempotent re-run)
            reset = session.run(
                "MATCH ()-[r:USES_TERM]->() DELETE r RETURN count(r) AS c"
            )
            reset_count = reset.single()["c"]
            if reset_count:
                logger.info("Cleared %d stale USES_TERM edges.", reset_count)

            # 6. Write new edges
            written = _write_edges(session, edges)
            logger.info("Wrote %d USES_TERM edges.", written)

    finally:
        driver.close()

    summary = {
        "terms": len(terms),
        "provisions": len(provisions),
        "provisions_with_terms": len(prov_ids_with_terms),
        "edges": written,
    }

    print(f"\n=== Term Linker Summary ===")
    print(f"  {'DefinedTerms:':<40} {len(terms):>5}")
    print(f"  {'Provisions scanned:':<40} {len(provisions):>5}")
    print(f"  {'Provisions with at least one term:':<40} {len(prov_ids_with_terms):>5}")
    print(f"  {'USES_TERM edges written:':<40} {written:>5}")
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise USES_TERM edges in Neo4j."
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    link_terms(dry_run=args.dry_run)
