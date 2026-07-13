"""Composite confidence scoring for CRSS answers.

Computes a five-component confidence score that is attached to every answer
and stored in the audit trace.  No LLM calls — all signals are derived from
retrieval metadata and the deterministic faithfulness check.

Components
----------
retrieval_coverage   (35 %) — fraction of route-specific sufficiency checks passed
retrieval_relevance  (20 %) — mean cosine similarity of top-3 retrieved provisions
faithfulness         (20 %) — fraction of verbatim quotes verified against source text
context_completeness (10 %) — penalises corrective passes, missing role provisions, etc.
legal_force_alignment(15 %) — binding/non-binding mix vs. what the question requires

Output
------
{
    "confidence_score":  float,          # 0.0 – 1.0
    "confidence_level":  str,            # HIGH / MEDIUM / LOW / CRITICAL
    "breakdown": {
        "retrieval_coverage":    float,
        "retrieval_relevance":   float,
        "faithfulness":          float,
        "context_completeness":  float,
        "legal_force_alignment": float,
    },
    "legal_force_distribution": {
        "binding":     int,
        "non_binding": int,
        "unknown":     int,
    },
}
"""
from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_W_COVERAGE    = 0.35
_W_RELEVANCE   = 0.20
_W_FAITHFULNESS = 0.20
_W_COMPLETENESS = 0.10
_W_LEGAL_FORCE  = 0.15

# ---------------------------------------------------------------------------
# Thresholds for discrete levels
# ---------------------------------------------------------------------------

_LEVEL_HIGH     = 0.85
_LEVEL_MEDIUM   = 0.65
_LEVEL_LOW      = 0.40
# Below _LEVEL_LOW → CRITICAL

# ---------------------------------------------------------------------------
# MDCG guidance detection phrases
# ---------------------------------------------------------------------------

_MDCG_PHRASES: tuple[str, ...] = (
    "mdcg",
    "guidance",
    "mdcg 2025",
    "mdcg 2024",
    "mdcg 2023",
    "mdcg 2022",
    "mdcg 2021",
    "mdcg 2020",
    "mdcg 2019",
)

# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def _retrieval_coverage_score(sufficiency: dict) -> float:
    """Fraction of route-specific sufficiency checks that passed.

    Returns 0.5 (neutral) when no checks are present so the score does not
    unfairly penalise routes that have no structured check (e.g. general_compliance).
    """
    checks = sufficiency.get("checks") or []
    if not checks:
        return 0.5
    passed = sum(1 for c in checks if c.get("passed"))
    return passed / len(checks)


def _retrieval_relevance_score(provisions: list[dict]) -> float:
    """Mean cosine similarity of the top-3 retrieved provisions.

    Provisions with score == 1.0 (direct-ref matches) are excluded from the
    mean because they are deterministic lookups, not similarity-ranked results.
    An empty list or all-1.0 list returns 0.75 (assumed adequate relevance for
    direct lookups).
    """
    sim_scores = [
        p["score"]
        for p in provisions
        if p.get("score") is not None and p["score"] < 1.0
    ]
    if not sim_scores:
        return 0.75  # direct-ref or role-retrieval — assume adequate relevance
    top = sorted(sim_scores, reverse=True)[:3]
    return sum(top) / len(top)


def _faithfulness_score(faith_report: Any) -> float:
    """Fraction of verbatim quotes verified against the retrieved source text.

    Returns 1.0 when no quotes are present (nothing to verify).
    Uses duck-typing so it works whether faith_report is a dataclass,
    namedtuple, or plain object with total_quotes / unverified_count attrs.
    """
    total = getattr(faith_report, "total_quotes", 0)
    if total == 0:
        return 1.0
    unverified = getattr(faith_report, "unverified_count", 0)
    return max(0.0, (total - unverified) / total)


def _context_completeness_score(
    sufficiency: dict,
    *,
    had_corrective_pass: bool,
    had_pointer_expansion: bool,
    had_role_provisions: bool,
    role_specs: list,
) -> float:
    """Penalise signals that indicate the initial retrieval was incomplete.

    Starts at 1.0 and subtracts weighted penalties:
    - Role specs present but no role provisions retrieved  → −0.30
    - Sufficiency check overall failed                    → −0.20
    - Corrective retrieval pass was needed                → −0.15
    - Pointer expansion added provisions                  → −0.10
    """
    score = 1.0
    if role_specs and not had_role_provisions:
        score -= 0.30
    if not sufficiency.get("ok", True):
        score -= 0.20
    if had_corrective_pass:
        score -= 0.15
    if had_pointer_expansion:
        score -= 0.10
    return max(score, 0.0)


def _legal_force_alignment_score(
    provisions: list[dict],
    question: str,
    mentioned_regs: set[str],
) -> float:
    """Measure whether the binding-force mix matches what the question requires.

    Three cases:
    1. Question explicitly targets MDCG guidance → non-binding provisions are
       appropriate; score rewards non-binding majority.
    2. Question targets one or more specific regulations → binding provisions
       expected; score penalises if binding ratio < 0.4.
    3. No specific regulation mentioned → neutral score of 0.70.
    """
    if not provisions:
        return 0.0

    q_lower = question.lower()
    asks_about_mdcg = any(phrase in q_lower for phrase in _MDCG_PHRASES)

    binding     = sum(1 for p in provisions if p.get("binding_force") == "binding")
    non_binding = sum(1 for p in provisions if p.get("binding_force") == "non_binding")
    total = len(provisions)

    if asks_about_mdcg:
        # Non-binding guidance is the expected source — reward it.
        return min(1.0, (non_binding + 0.3 * binding) / total)

    if mentioned_regs:
        # Binding regulation text is expected.
        binding_ratio = binding / total
        if binding_ratio >= 0.70:
            return 1.0
        elif binding_ratio >= 0.40:
            return 0.65
        else:
            return 0.30

    # No explicit regulation scope — mild neutral score.
    return 0.70


# ---------------------------------------------------------------------------
# Legal-force distribution helper (used in audit trace)
# ---------------------------------------------------------------------------


def _legal_force_distribution(provisions: list[dict]) -> dict[str, int]:
    """Count provisions by binding_force value."""
    counts = Counter(p.get("binding_force") or "unknown" for p in provisions)
    return {
        "binding":      counts.get("binding", 0),
        "non_binding":  counts.get("non_binding", 0),
        "interpretive": counts.get("interpretive", 0),
        "unknown":      counts.get("unknown", 0),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_confidence(
    *,
    sufficiency: dict,
    provisions: list[dict],
    faith_report: Any,
    had_corrective_pass: bool,
    had_pointer_expansion: bool,
    had_role_provisions: bool,
    role_specs: list,
    question: str,
    mentioned_regs: set[str],
) -> dict[str, Any]:
    """Compute the composite confidence score for a CRSS answer.

    Parameters
    ----------
    sufficiency:
        The dict returned by ``_evaluate_route_sufficiency()``.
    provisions:
        The final list of provision dicts passed to the LLM context.
    faith_report:
        The object returned by ``_check_faithfulness()``.
    had_corrective_pass:
        Whether a corrective retrieval pass was triggered.
    had_pointer_expansion:
        Whether pointer expansion added new provisions.
    had_role_provisions:
        Whether role-targeted retrieval returned results.
    role_specs:
        The ``[(term_normalized, celex), ...]`` list from the agent.
    question:
        The (possibly rewritten) user question.
    mentioned_regs:
        The set of regulation names detected in the question.

    Returns
    -------
    dict with keys: ``confidence_score``, ``confidence_level``,
    ``breakdown``, ``legal_force_distribution``.
    """
    coverage    = _retrieval_coverage_score(sufficiency)
    relevance   = _retrieval_relevance_score(provisions)
    faithfulness = _faithfulness_score(faith_report)
    completeness = _context_completeness_score(
        sufficiency,
        had_corrective_pass=had_corrective_pass,
        had_pointer_expansion=had_pointer_expansion,
        had_role_provisions=had_role_provisions,
        role_specs=role_specs,
    )
    legal_force = _legal_force_alignment_score(provisions, question, mentioned_regs)

    composite = (
        _W_COVERAGE     * coverage
        + _W_RELEVANCE   * relevance
        + _W_FAITHFULNESS * faithfulness
        + _W_COMPLETENESS * completeness
        + _W_LEGAL_FORCE  * legal_force
    )
    composite = round(min(max(composite, 0.0), 1.0), 3)

    if composite >= _LEVEL_HIGH:
        level = "HIGH"
    elif composite >= _LEVEL_MEDIUM:
        level = "MEDIUM"
    elif composite >= _LEVEL_LOW:
        level = "LOW"
    else:
        level = "CRITICAL"

    return {
        "confidence_score": composite,
        "confidence_level": level,
        "breakdown": {
            "retrieval_coverage":    round(coverage, 3),
            "retrieval_relevance":   round(relevance, 3),
            "faithfulness":          round(faithfulness, 3),
            "context_completeness":  round(completeness, 3),
            "legal_force_alignment": round(legal_force, 3),
        },
        "legal_force_distribution": _legal_force_distribution(provisions),
    }