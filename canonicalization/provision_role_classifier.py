"""
canonicalization/provision_role_classifier.py
==============================================
Assign a primary legal role (``provision_role``) to every :Provision node in
Neo4j, using the deterministic high-precision rules in
``domain.ontology.provision_roles``.

This is a **semantic-enrichment** stage in the canonicalization pipeline.
It populates five properties on each Provision node, all carrying
provenance so any downstream legal-reasoning use is auditable:

- ``provision_role``:             primary role (closed taxonomy)
- ``provision_role_source``:      ``rule`` | ``llm`` | ``human``
- ``provision_role_rule_id``:     machine identifier of the rule that matched
- ``provision_role_confidence``:  0.0 - 1.0 (1.0 for deterministic rules)
- ``provision_role_assigned_at``: ISO timestamp of assignment

The stage is **idempotent**: previous ``rule``/``llm`` values are cleared
before re-assignment. Curated ``human`` overrides are preserved.

Usage::

    python -m canonicalization.provision_role_classifier
    python -m canonicalization.provision_role_classifier --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from canonicalization.text_enrichment import strip_context_prefix
from domain.ontology.provision_roles import (
    PROVISION_ROLE_SOURCE_RULE,
    PROVISION_ROLE_TAXONOMY,
    classify_provision,
)
from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)

_BATCH = 500


# ---------------------------------------------------------------------------
# Neo4j I/O
# ---------------------------------------------------------------------------

def _load_provisions(session) -> list[dict[str, Any]]:
    """Return every :Provision node with the fields needed for classification.

    Prefers ``text_for_analysis_full`` — the *uncapped* flattened body, present
    only on the nodes whose ``text_for_analysis`` was truncated by the embedding
    cap — then the capped ``text_for_analysis``, then the raw ``text``.
    Classification must see the *whole* normative body: for article-container
    nodes ``p.text`` is heading-only (the body lives in child paragraphs), so
    classifying on ``p.text`` leaves them ``UNCLASSIFIED``; and a late
    EXEMPTS/PENALTY cue past the embedding cap in a long article would be lost if
    it read the capped field. The ancestry prefix carried by both flattened
    fields is removed downstream via ``strip_context_prefix``.
    """
    return session.run(
        "MATCH (p:Provision) "
        "RETURN p.id AS id, p.celex AS celex, p.kind AS kind, "
        "       coalesce(p.title, '') AS title, "
        "       coalesce(p.text_for_analysis_full, p.text_for_analysis, p.text, '') AS text"
    ).data()


def _clear_previous(session) -> int:
    """Remove existing rule/llm assignments. Human overrides are preserved."""
    result = session.run(
        "MATCH (p:Provision) "
        "WHERE p.provision_role_source IN ['rule', 'llm'] "
        "REMOVE p.provision_role, p.provision_role_source, "
        "       p.provision_role_rule_id, p.provision_role_confidence, "
        "       p.provision_role_assigned_at "
        "RETURN count(p) AS c"
    )
    return result.single()["c"]


def _write_assignments(session, assignments: list[dict[str, Any]]) -> int:
    """Batch-write provision_role properties via UNWIND.

    Never overwrites a provision whose existing ``provision_role_source`` is
    ``human``.
    """
    if not assignments:
        return 0
    cypher = (
        "UNWIND $batch AS a "
        "MATCH (p:Provision {id: a.id}) "
        "WHERE p.provision_role_source IS NULL OR p.provision_role_source <> 'human' "
        "SET p.provision_role = a.role, "
        "    p.provision_role_source = a.source, "
        "    p.provision_role_rule_id = a.rule_id, "
        "    p.provision_role_confidence = a.confidence, "
        "    p.provision_role_assigned_at = a.assigned_at "
        "RETURN count(p) AS c"
    )
    total = 0
    for i in range(0, len(assignments), _BATCH):
        chunk = assignments[i : i + _BATCH]
        total += session.run(cypher, batch=chunk).single()["c"]
    return total


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def classify_provision_roles(dry_run: bool = False) -> dict[str, Any]:
    """Main entry point. Returns summary counts including role distribution."""
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
            provisions = _load_provisions(session)
            logger.info("Loaded %d provisions for classification.", len(provisions))

            assigned_at = datetime.now(timezone.utc).isoformat()
            assignments: list[dict[str, Any]] = []
            role_counts: Counter = Counter()
            rule_counts: Counter = Counter()
            per_celex: dict[str, Counter] = {}

            for row in provisions:
                assignment = classify_provision(
                    text=strip_context_prefix(row["text"]),
                    kind=row["kind"],
                    title=row["title"] or None,
                    provision_id=row["id"],
                    celex=row["celex"],
                )
                role_counts[assignment.role] += 1
                rule_counts[assignment.rule_id] += 1
                per_celex.setdefault(row["celex"], Counter())[assignment.role] += 1
                assignments.append({
                    "id": row["id"],
                    "role": assignment.role,
                    "source": PROVISION_ROLE_SOURCE_RULE,
                    "rule_id": assignment.rule_id,
                    "confidence": assignment.confidence,
                    "assigned_at": assigned_at,
                })

            if dry_run:
                print("\n=== Provision Role Classifier (dry run) ===")
                _print_summary(
                    len(provisions), role_counts, rule_counts, per_celex, written=0
                )
                return {
                    "provisions": len(provisions),
                    "written": 0,
                    "role_counts": dict(role_counts),
                    "rule_counts": dict(rule_counts),
                }

            cleared = _clear_previous(session)
            if cleared:
                logger.info("Cleared %d previous rule/llm assignments.", cleared)
            written = _write_assignments(session, assignments)
            logger.info("Wrote provision_role on %d provisions.", written)

    finally:
        driver.close()

    print("\n=== Provision Role Classifier Summary ===")
    _print_summary(len(provisions), role_counts, rule_counts, per_celex, written=written)

    return {
        "provisions": len(provisions),
        "written": written,
        "role_counts": dict(role_counts),
        "rule_counts": dict(rule_counts),
    }


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def _print_summary(
    total_provisions: int,
    role_counts: Counter,
    rule_counts: Counter,
    per_celex: dict[str, Counter],
    *,
    written: int,
) -> None:
    print(f"  {'Provisions scanned:':<40} {total_provisions:>6}")
    print(f"  {'Provisions written:':<40} {written:>6}")
    print()
    print("  Distribution by role:")
    for role in PROVISION_ROLE_TAXONOMY:
        count = role_counts.get(role, 0)
        if count:
            pct = 100.0 * count / total_provisions if total_provisions else 0.0
            print(f"    {role:<18} {count:>6} ({pct:5.1f}%)")
    print()
    print("  Distribution by CELEX (top 5 roles each):")
    for celex in sorted(per_celex):
        sub = per_celex[celex]
        top = ", ".join(f"{r}={n}" for r, n in sub.most_common(5))
        print(f"    {celex}: {top}")
    print()
    print("  Top 10 rules fired:")
    for rule_id, count in rule_counts.most_common(10):
        print(f"    {rule_id:<42s} {count:>6}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Classify each :Provision node by its legal role."
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()
    classify_provision_roles(dry_run=args.dry_run)
