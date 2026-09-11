"""Build-gate: every hardcoded retrieval anchor must resolve in the graph.

CRSS force-loads specific provisions when a topic appears in a question
(``_CONTEXT_ANCHOR_REFS``, the qualification backbone, curated OBLIGATION_OF /
reasoning edges, definition anchors). Each names a provision by a frozen
display_ref. A re-ingest, renumbering, or amendment can move or delete that
target — and the failure is silent: the anchor resolves to nothing, or falls
back to the parent article and the decisive sub-provision never reaches context.

This test turns that silent rot into a red build. It requires a live Neo4j with
the graph loaded and skips otherwise (like the other integration tests). The
resolution logic itself lives in ``scripts/audit_anchor_resolution.py`` and is
reused here so the CLI audit and the test agree by construction.
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from scripts.audit_anchor_resolution import (  # noqa: E402
    collect_checks,
    resolve_ref,
    resolve_role,
    resolve_term,
)


@pytest.fixture(scope="module")
def neo4j_session():
    """Yield a Neo4j session, or skip the module if the DB is unreachable."""
    try:
        from neo4j import GraphDatabase
    except ImportError:  # pragma: no cover
        pytest.skip("neo4j driver not installed")

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "testpassword"))
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        with driver.session() as probe:
            n = probe.run("MATCH (p:Provision) RETURN count(p) AS n").single()["n"]
        if not n:
            driver.close()
            pytest.skip("Neo4j reachable but no provisions loaded")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Neo4j unavailable: {exc}")

    with driver.session() as session:
        yield session
    driver.close()


def test_all_anchors_resolve(neo4j_session):
    """No curated anchor may be MISSING or silently DEGRADED (parent-only)."""
    broken: list[str] = []
    for c in collect_checks():
        if c.kind == "term":
            status = resolve_term(neo4j_session, c.ref, c.celex)
        elif c.kind == "role":
            status = resolve_role(neo4j_session, c.ref, c.celex)
        else:
            status = resolve_ref(neo4j_session, c.ref, c.celex)
        if status != "OK":
            broken.append(f"{status:8} [{c.group}] {c.celex} {c.ref} ({c.kind})")

    assert not broken, (
        "Curated anchors no longer resolve in the graph — these silently "
        "degrade answers (DEGRADED = only the parent article resolves, the "
        "targeted sub-provision is gone):\n  " + "\n  ".join(broken)
    )
