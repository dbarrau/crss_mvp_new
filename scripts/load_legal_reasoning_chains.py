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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

# Resolve a curated ref against :Provision and :Guidance nodes of the same
# regulation. A ref matches either by exact node id (an author-time stable pin)
# or, failing that, by display_ref. ALL candidates are returned (no LIMIT 1) so
# the caller can detect when a display_ref is non-unique and would otherwise be
# silently bound to whichever node happened to sort first — the failure mode
# documented in the display-ref-ambiguity note.
_RESOLVE_REF_DETAIL_CYPHER = """\
MATCH (n)
WHERE (n:Provision OR n:Guidance)
  AND n.celex = $celex
  AND (n.id = $ref OR toLower(n.display_ref) = toLower($ref))
RETURN n.id AS node_id, n.display_ref AS display_ref, n.kind AS kind,
       coalesce(n.hierarchy_depth, 9999) AS depth
ORDER BY depth ASC, n.id ASC
"""


@dataclass
class _RefResolution:
    """Outcome of resolving one curated (ref, celex) pair to a graph node."""
    ref: str
    celex: str
    node_id: str | None
    status: str  # 'unique' | 'ambiguous' | 'unresolved'
    candidates: list[dict] = field(default_factory=list)


def _resolve_ref(session, ref: str, celex: str) -> _RefResolution:
    """Resolve a curated ref, surfacing ambiguity instead of hiding it.

    - An exact node-id match is authoritative and unambiguous (this is the
      author-time escape hatch: write a stable id in legal_reasoning_chains.py
      when a display_ref is non-unique).
    - Otherwise a single display_ref match is ``unique``; multiple matches are
      ``ambiguous`` (a node is still chosen deterministically, but it is
      reported so it can be pinned); zero matches are ``unresolved``.
    """
    rows = session.run(_RESOLVE_REF_DETAIL_CYPHER, ref=ref, celex=celex).data()
    if not rows:
        return _RefResolution(ref, celex, None, "unresolved")
    id_match = next((r for r in rows if r["node_id"] == ref), None)
    if id_match is not None:
        return _RefResolution(ref, celex, id_match["node_id"], "unique")
    status = "unique" if len(rows) == 1 else "ambiguous"
    candidates = (
        [{"node_id": r["node_id"], "kind": r["kind"]} for r in rows]
        if status == "ambiguous" else []
    )
    return _RefResolution(ref, celex, rows[0]["node_id"], status, candidates)

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

def _build_ref_cache(
    session, edges: list[LegalReasoningEdge]
) -> tuple[dict[tuple[str, str], str | None], list[_RefResolution]]:
    """Pre-resolve all (ref, celex) pairs.

    Returns ``(id_cache, resolutions)`` where ``id_cache`` maps each pair to a
    chosen node id (or ``None``) and ``resolutions`` carries the full status of
    each pair for the coverage report.
    """
    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        pairs.add((edge.source_ref, edge.celex))
        target_celex = edge.cross_celex or edge.celex
        for ref in edge.target_refs:
            pairs.add((ref, target_celex))

    cache: dict[tuple[str, str], str | None] = {}
    resolutions: list[_RefResolution] = []
    for ref, celex in sorted(pairs):
        res = _resolve_ref(session, ref, celex)
        cache[(ref, celex)] = res.node_id
        resolutions.append(res)
        if res.status == "unresolved":
            logger.warning("Could not resolve reasoning ref: %r (celex=%s)", ref, celex)
        elif res.status == "ambiguous":
            logger.warning(
                "Ambiguous reasoning ref %r (celex=%s) matched %d nodes; chose %s. "
                "Pin a stable node id in legal_reasoning_chains.py to disambiguate.",
                ref, celex, len(res.candidates), res.node_id,
            )

    return cache, resolutions


# ---------------------------------------------------------------------------
# Main loading logic
# ---------------------------------------------------------------------------

def load_obligation_patches(
    driver, db: str, dry_run: bool = False
) -> dict[str, Any]:
    """Load curated OBLIGATION_OF patches for graph coverage gaps."""
    stats: dict[str, Any] = {"resolved": 0, "unresolved": 0, "created": 0}
    resolutions: list[_RefResolution] = []
    with driver.session(database=db) as session:
        for patch in _ALL_OBLIGATION_PATCHES:
            # Resolve provision by stable id or display_ref (+ celex).
            res = _resolve_ref(session, patch.provision_ref, patch.celex)
            resolutions.append(res)
            if res.node_id is None:
                logger.warning(
                    "OBLIGATION_OF patch: could not resolve provision %r (celex=%s)",
                    patch.provision_ref, patch.celex,
                )
                stats["unresolved"] += 1
                continue
            if res.status == "ambiguous":
                logger.warning(
                    "OBLIGATION_OF patch: ambiguous provision %r (celex=%s) matched "
                    "%d nodes; chose %s. Pin a stable node id to disambiguate.",
                    patch.provision_ref, patch.celex, len(res.candidates), res.node_id,
                )
            provision_id = res.node_id
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
    stats["ambiguous"] = sum(1 for r in resolutions if r.status == "ambiguous")
    stats["_resolutions"] = resolutions
    return stats


def load_edges(driver, db: str, dry_run: bool = False) -> dict[str, Any]:
    """Load all legal reasoning edges and return a count summary."""
    with driver.session(database=db) as session:
        logger.info("Pre-resolving provision references (%d edges)…", len(_ALL_EDGES))
        cache, resolutions = _build_ref_cache(session, _ALL_EDGES)

        stats: dict[str, Any] = {"resolved": 0, "unresolved": 0, "created": 0, "skipped": 0}

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

    stats["ambiguous"] = sum(1 for r in resolutions if r.status == "ambiguous")
    stats["ref_total"] = len(resolutions)
    stats["_resolutions"] = resolutions
    return stats


def _print_coverage_report(resolutions: list[_RefResolution]) -> dict[str, int]:
    """Print a visible coverage report and return its summary counts.

    Replaces the prior behaviour where an unresolved or ambiguous ref produced
    only a buried ``logger.warning`` — making the health of the ~curated edge
    set inspectable at a glance after every load.
    """
    unique = [r for r in resolutions if r.status == "unique"]
    ambiguous = [r for r in resolutions if r.status == "ambiguous"]
    unresolved = [r for r in resolutions if r.status == "unresolved"]

    print("\n=== Legal-reasoning reference coverage ===")
    print(f"  distinct refs : {len(resolutions):>4}")
    print(f"  unique        : {len(unique):>4}")
    print(f"  ambiguous     : {len(ambiguous):>4}")
    print(f"  unresolved    : {len(unresolved):>4}")

    if ambiguous:
        print("\n  AMBIGUOUS (display_ref matched >1 node; chose first by hierarchy_depth —")
        print("  pin a stable node id in legal_reasoning_chains.py to disambiguate):")
        for r in ambiguous:
            cands = ", ".join(f"{c['node_id']}({c['kind']})" for c in r.candidates)
            print(f"    [{r.celex}] {r.ref!r} -> {r.node_id}   candidates: {cands}")
    if unresolved:
        print("\n  UNRESOLVED (no node matched; edge/patch skipped):")
        for r in unresolved:
            print(f"    [{r.celex}] {r.ref!r}")
    print()
    return {
        "distinct": len(resolutions),
        "unique": len(unique),
        "ambiguous": len(ambiguous),
        "unresolved": len(unresolved),
    }


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

        all_resolutions = stats.get("_resolutions", []) + patch_stats.get("_resolutions", [])
        _print_coverage_report(all_resolutions)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
