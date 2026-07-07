"""Unit tests for provision context budgeting.

``ask_stream`` reserves the space already consumed by the non-provision parts
(definitions, community overview, applicability notes) and passes the *remaining*
budget to ``_trim_provisions_to_budget``, so the total prompt — not just the
provision block — respects ``CRSS_CONTEXT_CHAR_BUDGET``.  These tests pin the
primitive that mechanism depends on: the trimmer honours an explicit budget and
is monotonic (a smaller budget keeps no more provisions).  No Neo4j / LLM.
"""
from __future__ import annotations

from application._context import (
    _TRIM_PROVISION_THRESHOLD,
    _format_one_provision,
    _trim_provisions_to_budget,
)
from domain.legislation_catalog import AI_ACT_CELEX


def _make_provision(idx: int, body_chars: int = 2000) -> dict:
    return {
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


def _bag(n: int) -> list[dict]:
    return [_make_provision(i) for i in range(1, n + 1)]


def test_small_bag_never_trimmed_regardless_of_budget():
    # At/below the threshold, full coverage is preserved even under a tiny budget
    # (deliberate: narrow overview queries must not silently drop articles).
    bag = _bag(_TRIM_PROVISION_THRESHOLD)
    assert _trim_provisions_to_budget(bag, budget=1) == bag


def test_large_bag_trimmed_to_fit_reduced_budget():
    bag = _bag(_TRIM_PROVISION_THRESHOLD + 15)  # comfortably over the threshold
    one = len(_format_one_provision(1, bag[0], "OBLIGATION"))
    kept = _trim_provisions_to_budget(bag, budget=one * 3)
    assert 1 <= len(kept) < len(bag)
    # Rendered size stays within a small multiple of the budget (kept is a prefix).
    assert len(kept) <= 4


def test_large_bag_fits_when_budget_is_ample():
    bag = _bag(_TRIM_PROVISION_THRESHOLD + 15)
    huge = 10_000_000
    assert _trim_provisions_to_budget(bag, budget=huge) == bag


def test_at_least_one_provision_always_kept():
    bag = _bag(_TRIM_PROVISION_THRESHOLD + 5)
    assert len(_trim_provisions_to_budget(bag, budget=0)) == 1


def test_smaller_budget_keeps_no_more_provisions_monotonic():
    # This is exactly what reserving space for definitions does: shrink the
    # provision budget -> keep fewer (never more) provisions.
    bag = _bag(_TRIM_PROVISION_THRESHOLD + 30)
    one = len(_format_one_provision(1, bag[0], "OBLIGATION"))
    counts = [len(_trim_provisions_to_budget(bag, budget=one * m)) for m in (2, 5, 10, 20)]
    assert counts == sorted(counts)  # non-decreasing as budget grows