"""Unit tests for ambiguity-aware reasoning-edge ref resolution (#4).

These exercise the pure resolution logic with a fake Neo4j session, so no live
database is required.
"""
from scripts.load_legal_reasoning_chains import _resolve_ref


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    """Returns a preset row list regardless of query, capturing kwargs."""

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def run(self, cypher, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult(self._rows)


def test_resolve_ref_unique_single_display_ref_match():
    session = _FakeSession([
        {"node_id": "C_art_6", "display_ref": "Article 6", "kind": "article", "depth": 4},
    ])
    res = _resolve_ref(session, "Article 6", "C")
    assert res.status == "unique"
    assert res.node_id == "C_art_6"
    assert res.candidates == []


def test_resolve_ref_ambiguous_picks_first_and_reports_candidates():
    # Two nodes share the display_ref (the annex sub-node failure mode).
    session = _FakeSession([
        {"node_id": "C_annex_8_a", "display_ref": "Annex VIII", "kind": "annex_section", "depth": 3},
        {"node_id": "C_annex_8_b", "display_ref": "Annex VIII", "kind": "annex_part", "depth": 5},
    ])
    res = _resolve_ref(session, "Annex VIII", "C")
    assert res.status == "ambiguous"
    assert res.node_id == "C_annex_8_a"  # deterministic: first by the Cypher ORDER BY
    assert {c["node_id"] for c in res.candidates} == {"C_annex_8_a", "C_annex_8_b"}


def test_resolve_ref_exact_id_pin_is_authoritative_even_when_others_match():
    # Author pinned a stable node id; it must win and be reported unambiguous,
    # even though a same-display_ref sibling is also returned.
    session = _FakeSession([
        {"node_id": "C_annex_8_b", "display_ref": "Annex VIII", "kind": "annex_part", "depth": 5},
        {"node_id": "C_annex_8_a", "display_ref": "Annex VIII", "kind": "annex_section", "depth": 3},
    ])
    res = _resolve_ref(session, "C_annex_8_b", "C")
    assert res.status == "unique"
    assert res.node_id == "C_annex_8_b"
    assert res.candidates == []


def test_resolve_ref_unresolved_when_no_match():
    session = _FakeSession([])
    res = _resolve_ref(session, "Article 999", "C")
    assert res.status == "unresolved"
    assert res.node_id is None
