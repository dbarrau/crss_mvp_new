"""Unit tests for provision context budgeting.

``ask_stream`` reserves the space already consumed by the non-provision parts
(definitions, community overview, applicability notes) and passes the *remaining*
budget to ``_trim_provisions_to_budget``, so the total prompt — not just the
provision block — respects ``CRSS_CONTEXT_CHAR_BUDGET``.  These tests pin the
primitives that mechanism depends on: the trimmer honours an explicit budget
unconditionally (the old ≤25-count fast path let a 25-provision bag render a
222 KB context), is monotonic, and every non-direct provision block is bounded
by ``_PROVISION_BLOCK_CAP`` (giant definition articles at the head of a
backbone starved the role channel out of the budget entirely — v6 eval,
HQ_003).  No Neo4j / LLM.
"""
from __future__ import annotations

from application._context import (
    _PROVISION_BLOCK_CAP,
    _format_one_provision,
    _trim_provisions_to_budget,
)
from domain.legislation_catalog import AI_ACT_CELEX


def _make_provision(idx: int, body_chars: int = 2000, **extra) -> dict:
    p = {
        "article_id": f"32024R1689_article_{idx}",
        "celex": AI_ACT_CELEX,
        "regulation": "AI Act",
        "article_ref": f"Article {idx}",
        "article_path": "",
        "article_text": "x" * body_chars,
        "children": [],
        "cited_provisions": [],
        "cross_reg_cited": [],
        "provision_role": "OBLIGATION",
        "score": 1.0,
        "matched_leaf_id": None,
    }
    p.update(extra)
    return p


def _giant(idx: int, **extra) -> dict:
    """A provision whose per-piece budgets sum far past the block cap."""
    children = [
        {"id": f"32024R1689_article_{idx}_p{i}", "ref": f"Paragraph {i}",
         "kind": "paragraph", "raw_text": "y" * 690}
        for i in range(1, 13)
    ]
    cites = [
        {"id": f"c{i}", "ref": f"Article {90 + i}", "text": "z" * 690}
        for i in range(6)
    ]
    return _make_provision(idx, body_chars=3900, children=children,
                           cited_provisions=cites, **extra)


def _bag(n: int) -> list[dict]:
    return [_make_provision(i) for i in range(1, n + 1)]


# ── block cap ───────────────────────────────────────────────────────────────

def test_block_cap_bounds_giant_provisions():
    block = _format_one_provision(1, _giant(3), "OBLIGATION")
    assert len(block) <= _PROVISION_BLOCK_CAP + 100  # + truncation marker
    assert "truncated to context budget" in block


def test_direct_ref_subject_exempt_from_block_cap():
    """'What does Annex III list?' must render its children complete — the
    structural-lookup lesson: never truncate the direct subject."""
    block = _format_one_provision(1, _giant(3, _direct_ref_match=True), "OBLIGATION")
    assert "truncated to context budget" not in block
    assert len(block) > _PROVISION_BLOCK_CAP


def test_system_anchor_not_exempt_from_block_cap():
    """A force-loaded backbone anchor (e.g. Annex I for a qualification
    question) is not the user's subject — the full-render exemption must not
    apply, or one anchor eats the budget and starves the role channel
    (v6 eval, HQ_003: trim kept 9 of 49 provisions)."""
    block = _format_one_provision(
        1, _giant(3, _direct_ref_match=True, _system_anchor=True), "OBLIGATION"
    )
    assert len(block) <= _PROVISION_BLOCK_CAP + 100
    assert "truncated to context budget" in block


def test_normal_provision_untouched_by_block_cap():
    block = _format_one_provision(1, _make_provision(5), "OBLIGATION")
    assert "truncated" not in block


def test_named_provision_capped_on_analytical_route():
    """A direct-ref provision renders its full subtree only on the display route.
    On an analytical route (allow_subject_render=False) it is context, not the
    user's subject, so it stays bounded by the block cap — closing the bloat where
    AI Act Article 5 (20 KB) / MDR Article 10 (15 KB) rendered full and crowded the
    decisive backbone out of the budget."""
    giant = _giant(5, _direct_ref_match=True)
    subject = _format_one_provision(1, giant, "OBLIGATION", allow_subject_render=True)
    analytical = _format_one_provision(1, giant, "OBLIGATION", allow_subject_render=False)
    assert len(subject) > _PROVISION_BLOCK_CAP          # display route: full render
    assert len(analytical) <= _PROVISION_BLOCK_CAP + 100  # analytical: capped
    assert "truncated to context budget" in analytical


# ── budget trim ─────────────────────────────────────────────────────────────

def test_small_bag_over_budget_is_now_trimmed():
    """The old ≤25-count fast path skipped trimming entirely; a small bag of
    giants must now be trimmed to the budget like any other."""
    bag = [_giant(i) for i in range(1, 11)]
    one = len(_format_one_provision(1, bag[0], "OBLIGATION"))
    kept = _trim_provisions_to_budget(bag, budget=one * 3)
    assert 1 <= len(kept) < len(bag)


def test_bag_returned_unchanged_when_it_fits():
    bag = _bag(20)
    assert _trim_provisions_to_budget(bag, budget=10_000_000) == bag


def test_large_bag_trimmed_to_fit_reduced_budget():
    bag = _bag(40)
    one = len(_format_one_provision(1, bag[0], "OBLIGATION"))
    kept = _trim_provisions_to_budget(bag, budget=one * 3)
    assert 1 <= len(kept) <= 4


def test_at_least_one_provision_always_kept():
    bag = _bag(30)
    assert len(_trim_provisions_to_budget(bag, budget=0)) == 1


def test_smaller_budget_keeps_no_more_provisions_monotonic():
    bag = _bag(55)
    one = len(_format_one_provision(1, bag[0], "OBLIGATION"))
    counts = [len(_trim_provisions_to_budget(bag, budget=one * m)) for m in (2, 5, 10, 20)]
    assert counts == sorted(counts)  # non-decreasing as budget grows


# ── backbone priority (direct-ref never evicted by unpinned) ─────────────────

def test_backbone_kept_over_unpinned_when_budget_is_tight():
    """A force-loaded backbone provision (``_direct_ref_match``) must survive over a
    lower-priority unpinned provision even when the unpinned one arrives FIRST and
    the budget fits only one block. Regression: the arrival-order trim kept a
    score-0.0 Omnibus node and dropped the force-loaded safety-component definition."""
    unpinned = _make_provision(1)                       # arrives first, not backbone
    backbone = _make_provision(2, _direct_ref_match=True)  # arrives second, decisive
    one = len(_format_one_provision(1, unpinned, "OBLIGATION"))
    kept = _trim_provisions_to_budget([unpinned, backbone], budget=one + 10)
    refs = [p["article_ref"] for p in kept]
    assert refs == ["Article 2"]                        # backbone survived, unpinned dropped


def test_output_preserves_original_order_across_tiers():
    a = _make_provision(1)                               # unpinned
    b = _make_provision(2, _direct_ref_match=True)       # backbone
    c = _make_provision(3)                               # unpinned
    kept = _trim_provisions_to_budget([a, b, c], budget=10_000_000)  # all fit
    assert [p["article_ref"] for p in kept] == ["Article 1", "Article 2", "Article 3"]


def test_backbone_tier_still_bounded_by_budget():
    """Prioritising the backbone must not defeat the hard cap: an all-backbone bag
    that overflows is still trimmed (time-to-first-token stays bounded)."""
    bag = [_giant(i, _direct_ref_match=True) for i in range(1, 11)]
    one = len(_format_one_provision(1, bag[0], "OBLIGATION"))
    kept = _trim_provisions_to_budget(bag, budget=one * 3)
    assert 1 <= len(kept) < len(bag)
