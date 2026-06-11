#!/usr/bin/env python3
"""Load legal reasoning edges from domain/ontology/legal_reasoning_chains.py into Neo4j.

For each LegalReasoningEdge, this script:
1. Resolves the source provision node by display_ref + celex
2. Resolves each target provision node by display_ref + (target celex)
3. Creates the relationship edge of the given type

Relationship types created:
    TRIGGERS_OBLIGATION_CLUSTER
    IS_PREREQUISITE_FOR
    REQUIRES_PRIOR_CHECK
    DEROGATES_FROM

Each edge gets:
    rationale   (str)   — human-readable explanation
    source_file (str)   — "legal_reasoning_chains.py" for audit traceability

Usage:
    python scripts/load_legal_reasoning_chains.py
    python scripts/load_legal_reasoning_chains.py --dry-run
    python scripts/load_legal_reasoning_chains.py --clear   # remove all edges first

Environment variables (from .env):
    NEO4J_URI, NEO4J_USERNAME / NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add workspace root to sys.path so imports work when run as script
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=False)

from domain.ontology.legal_reasoning_chains import (  # noqa: E402
    _ALL_EDGES,
    _ALL_OBLIGATION_PATCHES,
    CuratedObligationEdge,
    LegalReasoningEdge,
)
from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

_RESOLVE_REF_CYPHER = """\
MATCH (p:Provision {celex: $celex})
WHERE toLower(p.display_ref) = toLower($ref)
  AND p.kind IN ['article', 'annex_section', 'annex_part', 'annex', 'recital',
                 'section', 'chapter', 'title']
RETURN p.id AS node_id, p.display_ref AS display_ref, p.kind AS kind
ORDER BY p.hierarchy_depth ASC
LIMIT 1
"""

# Also try Guidance nodes so edges can be created for guidance documents.
_RESOLVE_REF_ANY_CYPHER = """\
OPTIONAL MATCH (p1:Provision {celex: $celex})
  WHERE toLower(p1.display_ref) = toLower($ref)
OPTIONAL MATCH (p2:Guidance {celex: $celex})
  WHERE toLower(p2.display_ref) = toLower($ref)
WITH coalesce(p1, p2) AS node
WHERE node IS NOT NULL
RETURN node.id AS node_id, node.display_ref AS display_ref, node.kind AS kind
ORDER BY node.hierarchy_depth ASC
LIMIT 1
"""

_CREATE_EDGE_CYPHER = """\
MATCH (src) WHERE src.id = $src_id
MATCH (tgt) WHERE tgt.id = $tgt_id
MERGE (src)-[r:`{rel_type}` {source_file: 'legal_reasoning_chains.py'}]->(tgt)
ON CREATE SET r.rationale = $rationale
ON MATCH  SET r.rationale = $rationale
RETURN type(r) AS rel_type
"""

_CLEAR_EDGES_CYPHER = """\
MATCH ()-[r]-()
WHERE type(r) IN [
  'TRIGGERS_OBLIGATION_CLUSTER',
  'IS_PREREQUISITE_FOR',
  'REQUIRES_PRIOR_CHECK',
  'DEROGATES_FROM'
]
  AND r.source_file = 'legal_reasoning_chains.py'
DELETE r
RETURN count(r) AS deleted
"""

# Cypher to create a curated OBLIGATION_OF edge.
# Finds the ActorRole node by matching celex + term_normalized, then merges
# the edge so repeated runs are idempotent.
_CREATE_OBLIGATION_OF_CYPHER = """\
MATCH (p) WHERE p.id = $provision_id
MATCH (r:ActorRole {celex: $celex, term_normalized: $role_term})
MERGE (p)-[e:OBLIGATION_OF {source_file: 'legal_reasoning_chains.py'}]->(r)
ON CREATE SET e.rationale = $rationale, e.curated = true
ON MATCH  SET e.rationale = $rationale, e.curated = true
RETURN p.display_ref AS ref, r.term_normalized AS role
"""

_CLEAR_OBLIGATION_PATCHES_CYPHER = """\
MATCH ()-[r:OBLIGATION_OF {source_file: 'legal_reasoning_chains.py'}]-()
DELETE r
RETURN count(r) AS deleted
"""


# ---------------------------------------------------------------------------
# Resolution cache
# ---------------------------------------------------------------------------

def _build_ref_cache(session, edges: list[LegalReasoningEdge]) -> dict[tuple[str, str], str | None]:
    """Pre-resolve all (ref, celex) pairs to their node IDs."""
    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        pairs.add((edge.source_ref, edge.celex))
        target_celex = edge.cross_celex or edge.celex
        for ref in edge.target_refs:
            pairs.add((ref, target_celex))

    cache: dict[tuple[str, str], str | None] = {}
    for ref, celex in pairs:
        rows = session.run(_RESOLVE_REF_ANY_CYPHER, ref=ref, celex=celex).data()
        cache[(ref, celex)] = rows[0]["node_id"] if rows else None
        if not rows:
            logger.warning("Could not resolve provision: %r (celex=%s)", ref, celex)

    return cache


# ---------------------------------------------------------------------------
# Main loading logic
# ---------------------------------------------------------------------------

def load_obligation_patches(
    driver, db: str, dry_run: bool = False
) -> dict[str, int]:
    """Load curated OBLIGATION_OF patches for graph coverage gaps."""
    stats = {"resolved": 0, "unresolved": 0, "created": 0}
    with driver.session(database=db) as session:
        for patch in _ALL_OBLIGATION_PATCHES:
            # Resolve provision by display_ref + celex
            rows = session.run(
                _RESOLVE_REF_ANY_CYPHER, ref=patch.provision_ref, celex=patch.celex
            ).data()
            if not rows:
                logger.warning(
                    "OBLIGATION_OF patch: could not resolve provision %r (celex=%s)",
                    patch.provision_ref, patch.celex,
                )
                stats["unresolved"] += 1
                continue
            provision_id = rows[0]["node_id"]
            stats["resolved"] += 1
            if dry_run:
                logger.info(
                    "[DRY-RUN] OBLIGATION_OF: %s (%s) -[:OBLIGATION_OF]-> %s",
                    patch.provision_ref, patch.celex, patch.role_term,
                )
                continue
            result = session.run(
                _CREATE_OBLIGATION_OF_CYPHER,
                provision_id=provision_id,
                celex=patch.celex,
                role_term=patch.role_term,
                rationale=patch.rationale[:500] if patch.rationale else "",
            ).data()
            if result:
                stats["created"] += 1
                logger.info(
                    "  OBLIGATION_OF: %s → role:%s",
                    patch.provision_ref, patch.role_term,
                )
            else:
                logger.warning(
                    "  OBLIGATION_OF: ActorRole not found for %s / %s",
                    patch.celex, patch.role_term,
                )
                stats["unresolved"] += 1
    return stats


def load_edges(driver, db: str, dry_run: bool = False) -> dict[str, int]:
    """Load all legal reasoning edges and return a count summary."""
    with driver.session(database=db) as session:
        logger.info("Pre-resolving provision references (%d edges)…", len(_ALL_EDGES))
        cache = _build_ref_cache(session, _ALL_EDGES)

        stats = {"resolved": 0, "unresolved": 0, "created": 0, "skipped": 0}

        for edge in _ALL_EDGES:
            src_id = cache.get((edge.source_ref, edge.celex))
            if not src_id:
                stats["unresolved"] += 1
                continue

            target_celex = edge.cross_celex or edge.celex
            for target_ref in edge.target_refs:
                tgt_id = cache.get((target_ref, target_celex))
                if not tgt_id:
                    logger.debug(
                        "  skip unresolved target: %s → %s (%s)",
                        edge.source_ref, target_ref, target_celex,
                    )
                    stats["unresolved"] += 1
                    continue

                stats["resolved"] += 1
                if dry_run:
                    logger.info(
                        "[DRY-RUN] %s -[%s]-> %s (%s)",
                        edge.source_ref, edge.type, target_ref, target_celex,
                    )
                    continue

                cypher = _CREATE_EDGE_CYPHER.replace("{rel_type}", edge.type)
                session.run(
                    cypher,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    rationale=edge.rationale[:500] if edge.rationale else "",
                )
                stats["created"] += 1
                logger.debug(
                    "  %s -[%s]-> %s", edge.source_ref, edge.type, target_ref
                )

    return stats


def clear_edges(driver, db: str) -> int:
    """Clear all edges written by legal_reasoning_chains.py (reasoning + OBLIGATION_OF patches)."""
    total = 0
    with driver.session(database=db) as session:
        ob_rows = session.run(_CLEAR_OBLIGATION_PATCHES_CYPHER).data()
        ob_deleted = ob_rows[0]["deleted"] if ob_rows else 0
        if ob_deleted:
            logger.info("Cleared %d curated OBLIGATION_OF edges.", ob_deleted)
        total += ob_deleted

        rows = session.run(_CLEAR_EDGES_CYPHER).data()
        deleted = rows[0]["deleted"] if rows else 0
        logger.info("Cleared %d legal reasoning edges.", deleted)
        total += deleted
    return total


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load legal reasoning edges into Neo4j"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned edges without writing to Neo4j",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Remove all existing legal_reasoning_chains.py edges before loading",
    )
    args = parser.parse_args()

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    auth = (
        os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
        os.environ.get("NEO4J_PASSWORD", "password"),
    )
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (db=%s)", uri, db)
    except Exception as exc:
        logger.error("Cannot connect to Neo4j: %s", exc)
        sys.exit(1)

    try:
        if args.clear and not args.dry_run:
            clear_edges(driver, db)

        stats = load_edges(driver, db, dry_run=args.dry_run)
        mode = "DRY-RUN" if args.dry_run else "LOADED"
        logger.info(
            "[%s] reasoning edges: resolved=%d unresolved=%d created=%d",
            mode,
            stats["resolved"],
            stats["unresolved"],
            stats["created"],
        )

        patch_stats = load_obligation_patches(driver, db, dry_run=args.dry_run)
        logger.info(
            "[%s] OBLIGATION_OF patches: resolved=%d unresolved=%d created=%d",
            mode,
            patch_stats["resolved"],
            patch_stats["unresolved"],
            patch_stats["created"],
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
