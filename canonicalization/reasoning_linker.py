"""Materialise curated legal-reasoning edges in Neo4j.

Loads the hand-curated statutory-dependency edges and ``OBLIGATION_OF`` patches
from ``domain/ontology/legal_reasoning_chains.py`` as a semantic-enrichment
stage of the canonicalization pipeline:

    ... → role_linker → provision_role_classifier → reasoning_linker → community_linker

Edge types created (consumed directly by ``retrieval/graph_retriever.py``):

    TRIGGERS_OBLIGATION_CLUSTER
    IS_PREREQUISITE_FOR
    REQUIRES_PRIOR_CHECK
    DEROGATES_FROM
    OBLIGATION_OF   (curated patches for coverage gaps the role_linker heuristic missed)

**Why this stage exists.** These edges are the system's richest legal-reasoning
asset (including the cross-regulation GDPR↔MDR↔AI-Act chains). Without this
stage a freshly rebuilt graph has *none* of them, so the retriever's
reasoning-chain traversals silently return nothing — the multi-hop reasoning
that distinguishes CRSS from plain vector RAG quietly disappears with no error.

The loading logic lives in ``scripts/load_legal_reasoning_chains.py`` (which
remains usable as a standalone CLI). This module wraps it so it runs as part of
``python -m canonicalization`` — mirroring how :mod:`community_linker` wraps
``scripts.build_communities``.

Ordering: must run **after** ``role_linker`` (the ``OBLIGATION_OF`` patches
``MATCH`` existing ``:ActorRole`` nodes). It is placed before
``community_linker`` purely for logical grouping with the other linkers;
community detection projects only ``HAS_PART``/``CITES`` edges, so these
reasoning edges do not affect community output regardless of order.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)


def link_reasoning_chains(*, dry_run: bool = False) -> dict[str, int]:
    """Load curated legal-reasoning edges and OBLIGATION_OF patches.

    Returns a summary dict with created/unresolved counts. Safe to re-run:
    existing edges written by this loader are cleared first (idempotent), and
    all writes use ``MERGE``.
    """
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    try:
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.load_legal_reasoning_chains import (
            _print_coverage_report,
            clear_edges,
            load_edges,
            load_obligation_patches,
        )
    except ImportError as exc:
        logger.warning(
            "reasoning_linker: could not import loader (%s). Skipping stage.", exc,
        )
        return {
            "reasoning_edges": 0,
            "reasoning_unresolved": 0,
            "obligation_patches": 0,
            "patch_unresolved": 0,
        }

    from neo4j import GraphDatabase

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    auth = (
        os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
        os.environ.get("NEO4J_PASSWORD", "password"),
    )
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        if not dry_run:
            # Idempotent: only removes edges this loader previously wrote
            # (filtered on source_file = 'legal_reasoning_chains.py').
            clear_edges(driver, db)
        edge_stats = load_edges(driver, db, dry_run=dry_run)
        patch_stats = load_obligation_patches(driver, db, dry_run=dry_run)
    finally:
        driver.close()

    coverage = _print_coverage_report(
        edge_stats.get("_resolutions", []) + patch_stats.get("_resolutions", [])
    )

    summary = {
        "reasoning_edges": edge_stats.get("created", 0),
        "reasoning_unresolved": edge_stats.get("unresolved", 0),
        "reasoning_ambiguous": coverage["ambiguous"],
        "obligation_patches": patch_stats.get("created", 0),
        "patch_unresolved": patch_stats.get("unresolved", 0),
    }
    logger.info(
        "reasoning_linker: %d reasoning edges, %d OBLIGATION_OF patches "
        "(unresolved: %d edge-targets, %d patches; ambiguous refs: %d).",
        summary["reasoning_edges"], summary["obligation_patches"],
        summary["reasoning_unresolved"], summary["patch_unresolved"],
        summary["reasoning_ambiguous"],
    )
    return summary
