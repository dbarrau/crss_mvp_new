"""
canonicalization/delegation_linker.py
=====================================
Materialise **DELEGATES_TO** edges between enacting provisions (articles,
paragraphs, points) and Annex provisions.

EU regulations have a recurring pattern: an enacting article delegates
technical detail to an Annex ("shall follow the procedure set out in
Annex IX", "the requirements of Annex III shall apply").  These delegation
relationships are already captured as CITES edges, but DELEGATES_TO makes
them semantically distinct, letting the retriever prioritise the delegation
chain over informational cross-references.

Scope:
  - Source: Provision with kind NOT starting with 'annex'
  - Target: Provision with kind STARTING with 'annex'
  - Language: source provision text must contain a delegation phrase

This is a post-processing step that should run **after** all documents are
loaded and CITES edges exist.  It is idempotent: re-running deletes
previous DELEGATES_TO edges first.

Usage::

    python -m canonicalization.delegation_linker
    python -m canonicalization.delegation_linker --dry-run
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)

_BATCH = 500

# ---------------------------------------------------------------------------
# Delegation language patterns
# ---------------------------------------------------------------------------
# These phrases signal that the source provision delegates an obligation,
# procedure, or set of requirements to the target annex.  The regex is
# matched against the full source provision text (case-insensitive).

_DELEGATION_PHRASES: list[str] = [
    r"in accordance with",
    r"set out in",
    r"laid down in",
    r"specified in",
    r"requirements?\s+of",
    r"pursuant to",
    r"shall be subject to",
    r"provided for in",
    r"referred to in",
    r"as described in",
    r"shall\s+(?:comply|conform|meet|fulfil|follow|apply)",
    r"the\s+(?:procedure|conditions|criteria|rules|obligations)"
    r"\s+(?:set out|laid down|specified|referred to|described|provided for)\s+in",
    r"covered by",
    r"listed in",
    r"contained in",
]

_DELEGATION_RE = re.compile("|".join(_DELEGATION_PHRASES), re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. Fetch candidate CITES edges (non-annex → annex)
# ---------------------------------------------------------------------------

def _load_candidates(session) -> list[dict]:
    """Return all CITES edges from non-annex provisions to annex provisions."""
    return session.run(
        "MATCH (src:Provision)-[c:CITES]->(tgt:Provision) "
        "WHERE tgt.kind STARTS WITH 'annex' "
        "  AND NOT src.kind STARTS WITH 'annex' "
        "RETURN src.id AS source, tgt.id AS target, "
        "       src.text AS src_text, c.ref_text AS ref_text"
    ).data()


# ---------------------------------------------------------------------------
# 2. Classify candidates
# ---------------------------------------------------------------------------

def _classify(candidates: list[dict]) -> list[dict]:
    """Return the subset of candidates whose source text contains delegation language."""
    delegations: list[dict] = []
    for c in candidates:
        text = c.get("src_text") or ""
        m = _DELEGATION_RE.search(text)
        if m:
            delegations.append({
                "source": c["source"],
                "target": c["target"],
                "ref_text": c.get("ref_text") or "",
                "delegation_phrase": m.group(),
            })
    return delegations


# ---------------------------------------------------------------------------
# 3. Write DELEGATES_TO edges
# ---------------------------------------------------------------------------

def _write_edges(session, edges: list[dict]) -> int:
    """MERGE DELEGATES_TO edges between existing Provision nodes."""
    if not edges:
        return 0

    total = 0
    cypher = (
        "UNWIND $batch AS e "
        "MATCH (s:Provision {id: e.source}) "
        "MATCH (t:Provision {id: e.target}) "
        "MERGE (s)-[r:DELEGATES_TO]->(t) "
        "ON CREATE SET r.resolved_from = 'delegation_classifier', "
        "              r.ref_text = e.ref_text, "
        "              r.delegation_phrase = e.delegation_phrase, "
        "              r.authority = 'binding' "
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

def link_delegations(dry_run: bool = False) -> dict[str, int]:
    """Main entry point. Returns summary counts."""
    load_dotenv()

    uri = _normalize_neo4j_uri(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database=database) as session:
            # Reset previous run (idempotent)
            if not dry_run:
                reset = session.run(
                    "MATCH ()-[r:DELEGATES_TO {resolved_from: 'delegation_classifier'}]->() "
                    "DELETE r RETURN count(r) AS c"
                )
                reset_count = reset.single()["c"]
                if reset_count:
                    logger.info(
                        "Cleared %d stale DELEGATES_TO edges.", reset_count
                    )

            # Load and classify
            candidates = _load_candidates(session)
            delegations = _classify(candidates)

            # Write
            written = 0
            if not dry_run and delegations:
                written = _write_edges(session, delegations)
    finally:
        driver.close()

    summary = {
        "candidates": len(candidates),
        "delegations_detected": len(delegations),
        "edges_written": written if not dry_run else 0,
        "informational_only": len(candidates) - len(delegations),
    }

    print("\n=== Delegation Linker Summary ===")
    print(f"  {'Non-annex → Annex CITES (candidates):':<45} {summary['candidates']:>4}")
    print(f"  {'With delegation language:':<45} {summary['delegations_detected']:>4}")
    print(f"  {'Informational only (no DELEGATES_TO):':<45} {summary['informational_only']:>4}")
    print(f"  {'DELEGATES_TO edges written:':<45} {summary['edges_written']:>4}")
    if dry_run:
        print("  (dry run — no changes written)")
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise DELEGATES_TO edges (enacting provision → annex)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    link_delegations(dry_run=args.dry_run)
