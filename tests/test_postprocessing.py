"""Tests for the reframed confidence banner (#3 — honesty-machinery presentation).

The banner used to lead with a bare "Confidence: LOW (Score: 61%)" + a generic
"independently verify" disclaimer that the quality judge anchored on as a blanket
reliability signal. The reframe surfaces only *actionable* caveats under a calm
"Scope & limitations" heading, and emits nothing when there is nothing to act on.
The composite score still flows to the UI as a structured ``confidence`` event.
"""
from application._postprocessing import _build_confidence_banner


def _conf(level, *, coverage=1.0, legal_force=1.0, non_binding=0, total=5):
    return {
        "confidence_level": level,
        "confidence_score": 0.61,
        "breakdown": {
            "retrieval_coverage": coverage,
            "legal_force_alignment": legal_force,
            "faithfulness": 1.0,
            "context_completeness": 1.0,
        },
        "legal_force_distribution": {
            "binding": total - non_binding,
            "non_binding": non_binding,
            "unknown": 0,
        },
    }


def test_high_confidence_emits_nothing():
    assert _build_confidence_banner(_conf("HIGH", coverage=0.1)) == ""


def test_no_bare_score_or_generic_disclaimer():
    # Even at LOW with a triggered caveat, the bare score and the generic
    # "independently verify" boilerplate must be gone.
    banner = _build_confidence_banner(_conf("LOW", coverage=0.3))
    assert "Scope & limitations" in banner
    assert "61%" not in banner
    assert "Score:" not in banner
    assert "Confidence:" not in banner
    assert "independently verified" not in banner


def test_low_coverage_surfaces_actionable_note():
    banner = _build_confidence_banner(_conf("LOW", coverage=0.3))
    assert "coverage" in banner.lower()


def test_non_binding_majority_reports_counts():
    banner = _build_confidence_banner(_conf("MEDIUM", legal_force=0.2, non_binding=4, total=5))
    assert "4 of 5" in banner
    assert "non-binding" in banner


def test_sub_high_with_no_actionable_caveat_is_silent():
    # MEDIUM/LOW but coverage + legal force are both fine → nothing to say.
    assert _build_confidence_banner(_conf("MEDIUM", coverage=0.9, legal_force=0.9)) == ""

# ---------------------------------------------------------------------------
# Internal context-index labels ("[14] Article 10(2)") must not leak to readers.
# ---------------------------------------------------------------------------

from application._postprocessing import _CONTEXT_INDEX_PATTERN  # noqa: E402


def test_context_index_labels_are_stripped_keeping_the_real_ref():
    text = "Risk management under [14] Article 10(2) MDR and [5] Article 43(4) AI Act."
    out = _CONTEXT_INDEX_PATTERN.sub("", text)
    assert "[14]" not in out and "[5]" not in out
    assert "Article 10(2) MDR" in out
    assert "Article 43(4) AI Act" in out
