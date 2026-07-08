"""Unit tests for CITES-edge promotion to first-class citable provisions.

Root cause of the fabricated-node-id failure (see
docs/grounded_generation_contract.md): the retriever resolves each provision's
CITES edges into ``cited_provisions`` snippets that render *without* an ``id:``
line, so the model cannot cite them under the grounded contract and may invent a
node id.  ``_collect_cites_targets`` returns those targets' real, unique node
ids so the pointer-expansion stage can promote them.  These tests pin its
selection, ordering, de-duplication and scope-filtering logic.  No Neo4j / LLM.
"""
from __future__ import annotations

from application._context import _collect_cites_targets

AI = "32024R1689"
GDPR = "32016R0679"
MDR = "32017R0745"


def _prov(article_id: str, *, cited: list[dict] | None = None,
          cross_reg: list[dict] | None = None) -> dict:
    return {
        "article_id": article_id,
        "cited_provisions": cited or [],
        "cross_reg_cited": cross_reg or [],
    }


def test_collects_cited_target_ids_not_already_in_bag():
    prov = _prov(
        f"{AI}_art_23",
        cited=[{"id": f"{AI}_art_43"}, {"id": f"{AI}_art_11"}],
    )
    targets = _collect_cites_targets([prov], already_ids={f"{AI}_art_23"})
    assert targets == [f"{AI}_art_43", f"{AI}_art_11"]


def test_excludes_targets_already_retrieved():
    prov = _prov(
        f"{AI}_art_23",
        cited=[{"id": f"{AI}_art_43"}, {"id": f"{AI}_art_11"}],
    )
    # art_43 is already in the main bag → must not be re-promoted.
    targets = _collect_cites_targets(
        [prov], already_ids={f"{AI}_art_23", f"{AI}_art_43"},
    )
    assert targets == [f"{AI}_art_11"]


def test_cross_regulation_targets_ordered_first():
    prov = _prov(
        f"{AI}_art_23",
        cited=[{"id": f"{AI}_art_11"}, {"id": f"{GDPR}_art_35"}],
        cross_reg=[{"id": f"{GDPR}_art_35"}],
    )
    targets = _collect_cites_targets([prov], already_ids=set())
    # The cross-regulation link is the more decisive bridge → comes first.
    assert targets == [f"{GDPR}_art_35", f"{AI}_art_11"]


def test_deduplicates_across_provisions_preserving_order():
    p1 = _prov(f"{AI}_art_23", cited=[{"id": f"{AI}_art_43"}])
    p2 = _prov(f"{AI}_art_16", cited=[{"id": f"{AI}_art_43"}, {"id": f"{AI}_art_11"}])
    targets = _collect_cites_targets([p1, p2], already_ids=set())
    assert targets == [f"{AI}_art_43", f"{AI}_art_11"]


def test_scope_filter_drops_out_of_regulation_targets():
    prov = _prov(
        f"{AI}_art_23",
        cited=[{"id": f"{AI}_art_11"}, {"id": f"{GDPR}_art_35"}, {"id": f"{MDR}_016.002"}],
        cross_reg=[{"id": f"{GDPR}_art_35"}, {"id": f"{MDR}_016.002"}],
    )
    # Scoped to the AI Act: GDPR/MDR cross-refs must not starve the in-scope
    # cross-reference out of the bounded budget.
    targets = _collect_cites_targets(
        [prov], already_ids=set(), celex_filter={AI},
    )
    assert targets == [f"{AI}_art_11"]


def test_ignores_snippets_without_an_id():
    prov = _prov(
        f"{AI}_art_23",
        cited=[{"ref": "Article 43"}, {"id": None}, {"id": f"{AI}_art_11"}],
    )
    targets = _collect_cites_targets([prov], already_ids=set())
    assert targets == [f"{AI}_art_11"]


def test_empty_when_no_cites():
    prov = _prov(f"{AI}_art_23")
    assert _collect_cites_targets([prov], already_ids=set()) == []
