"""Standing Omnibus↔AI-Act coherency guard.

Every provision that exists in the BASE act but is GONE from the consolidated
(current) act is a stale-citation risk. This test asserts that each such vanished
node is *explained* — either (a) recorded as deleted in the superseded record, (b)
a descendant of a recorded deletion (removed with its parent), or (c) restructured
under a citable ancestor that SURVIVES (a replace, not a deletion). An unexplained
vanished provision means the Omnibus changed something the superseded record does
not cover — the coherency gap this test exists to catch on the next re-consolidation.

Reads the base parsed.json + runs the consolidation report-only (no Neo4j / LLM);
skips if the base data is not present in this checkout.
"""
from __future__ import annotations

import json

import pytest

from consolidation.build import build_consolidated, _parsed_path, _DATA_ROOT
from consolidation.superseded import compute_superseded
from domain.legislation_catalog import consolidation_plan


def _plan():
    return consolidation_plan()


@pytest.mark.parametrize("base_celex,amender_celex", _plan())
def test_every_vanished_provision_is_explained(base_celex, amender_celex):
    base_path = _parsed_path(base_celex, "EN", _DATA_ROOT)
    if not base_path.exists():
        pytest.skip(f"base data for {base_celex} not present in this checkout")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    consolidated, _report = build_consolidated(base_celex, amender_celex)
    base_by_id = {n["id"]: n for n in base["provisions"]}
    cons_ids = {n["id"] for n in consolidated["provisions"]}
    parent = {c: n["id"] for n in base["provisions"] for c in n.get("children", []) or []}

    recorded = {r.deleted_id for r in compute_superseded(base_celex, amender_celex)}
    vanished = set(base_by_id) - cons_ids

    unexplained = []
    for vid in vanished:
        node = vid
        explained = False
        # walk self → ancestors
        while node:
            if node in recorded:            # (a)/(b) covered by a deletion record
                explained = True
                break
            if node != vid and node in cons_ids:   # (c) a surviving citable ancestor → restructuring
                explained = True
                break
            node = parent.get(node)
        if not explained:
            unexplained.append(vid)

    assert not unexplained, (
        f"{base_celex}: vanished provisions not covered by the superseded record "
        f"nor by a surviving ancestor — the record is stale, re-run "
        f"scripts/build_superseded_index.py: {sorted(unexplained)}"
    )
