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


# ---------------------------------------------------------------------------
# EPHEMERAL: superseded AI Act application-date flag (Reg (EU) 2026/1744 bridge).
# Flag-only, never rewrites; fires only when the stale date appears WITHOUT its
# corrected replacement, so correct "was 2027, now 2028" lineage answers are safe.
# ---------------------------------------------------------------------------

from application._postprocessing import _flag_superseded_ai_act_dates  # noqa: E402


def test_superseded_date_flag_fires_on_stale_answer():
    ans = "Under the AI Act, high-risk Article 6(1) systems apply from 2 August 2027."
    flags = _flag_superseded_ai_act_dates(ans)
    assert len(flags) == 1
    assert "2 August 2028" in flags[0]


def test_superseded_date_flag_silent_when_corrected_date_present():
    # Presence of the amended date is the correctness signal.
    ans = "High-risk AI under Article 6(1) applies from 2 August 2028 (Reg 2026/1744)."
    assert _flag_superseded_ai_act_dates(ans) == []


def test_superseded_date_flag_silent_on_lineage_answer():
    # A CORRECT answer legitimately names the old date in the amendment lineage —
    # it must NOT be flagged (this is precisely why the guard is not a strip).
    ans = (
        "Article 113 originally set 2 August 2027 for high-risk Annex I systems, "
        "but this was amended to 2 August 2028 by Regulation (EU) 2026/1744."
    )
    assert _flag_superseded_ai_act_dates(ans) == []


def test_superseded_date_flag_requires_ai_act_context():
    # The bare date outside any AI Act context is not ours to flag.
    assert _flag_superseded_ai_act_dates("The lease renews on 2 August 2027.") == []


def test_superseded_date_flag_handles_nonbreaking_space():
    # EUR-Lex dates use no-break spaces; the regex's \\s+ must still match.
    ans = "AI Act high-risk under Article 6 applies from 2 August 2027."
    assert len(_flag_superseded_ai_act_dates(ans)) == 1
