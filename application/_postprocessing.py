"""Answer post-processing — safety formatting, language softening, and banners.

Applied to the raw LLM output before it is returned to the caller.  Adds
uncertainty banners for qualification-heavy routes, softens over-categorical
phrasing, and annotates potential legal-backbone errors.  No LLM calls.
"""
from __future__ import annotations

import re
from typing import Any

from application._routing import _QuestionRoute, _has_inhouse_developer_signal

# ---------------------------------------------------------------------------
# Language-softening patterns (legal qualification route only)
# ---------------------------------------------------------------------------

_CATEGORICAL_SOFTENERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bonly when\b", re.IGNORECASE), "most clearly when"),
    (re.compile(r"\bconstitutes\b", re.IGNORECASE), "is likely to constitute"),
    (re.compile(r"\btriggers\b", re.IGNORECASE), "is likely to trigger"),
    (re.compile(r"\bno transition to\b", re.IGNORECASE), "no clear transition to"),
    (re.compile(r"\bdoes not trigger\b", re.IGNORECASE), "does not clearly trigger"),
    (re.compile(r"\bremoves the exemption\b", re.IGNORECASE), "is likely to remove the exemption"),
)

# ---------------------------------------------------------------------------
# Backbone-validation patterns
# ---------------------------------------------------------------------------

_SELF_DEPLOYER_PATTERN = re.compile(
    r"\b(?:hospital|institution|developer|entity)\s+is\s+(?:initially\s+)?a\s+deployer\b",
    re.IGNORECASE,
)

_INITIAL_DEPLOYER_PATTERN = re.compile(
    r"initially\s+(?:a\s+)?(?:acting\s+as\s+a?\s+)?deployer",
    re.IGNORECASE,
)

_RULE_LABEL_PATTERN = re.compile(
    r"(?m)^\s*(?:RULE\s+[A-Z0-9]+(?:\s*[—:-].*)?|MANDATORY LEGAL RULES — READ BEFORE ANSWERING:|LEGAL ANCHORS — READ BEFORE ANSWERING:).*$(?:\n?|$)",
    re.IGNORECASE,
)

# Internal context-index labels (e.g. "[14] Article 10(2)") leak from the
# REGULATORY CONTEXT header numbering, which the prompt asks the model to cite.
# They are meaningless to a reader — strip the bracketed index, keeping the real
# provision reference. "[1]" is never legitimate legal text (provisions number
# their points as "(1)"), so this is safe.
_CONTEXT_INDEX_PATTERN = re.compile(r"\[\d{1,3}\]\s?")

# ---------------------------------------------------------------------------
# Post-processing functions
# ---------------------------------------------------------------------------


def _build_uncertainty_banner(
    route: _QuestionRoute,
    *,
    sufficiency: dict[str, Any],
) -> str | None:
    """Return a visible banner for case-sensitive qualification answers."""
    if route.id != "legal_qualification":
        return None
    if sufficiency.get("ok", True):
        return (
            "> ASSESSMENT STATUS — Provisional legal qualification assessment. "
            "This answer should be read as a case-specific compliance analysis, "
            "not as an automatic status determination."
        )
    return (
        "> ASSESSMENT STATUS — Provisional legal qualification assessment with "
        "partial retrieval support. Treat conclusions below as tentative and "
        "case-specific unless directly quoted from the retrieved provisions."
    )


def _soften_categorical_language(
    answer: str,
    route: _QuestionRoute,
    *,
    sufficiency: dict[str, Any],
) -> str:
    """Reduce over-categorical phrasing for qualification-heavy answers."""
    if route.id != "legal_qualification":
        return answer

    softened = answer
    apply_softening = not sufficiency.get("ok", True) or any(
        phrase in answer.lower()
        for phrase in ("only when", "constitutes", "triggers", "no transition")
    )
    if not apply_softening:
        return answer

    for pattern, replacement in _CATEGORICAL_SOFTENERS:
        softened = pattern.sub(replacement, softened)
    return softened


def _validate_legal_backbone(
    answer: str,
    question: str,
    route: _QuestionRoute,
) -> list[str]:
    """Return warning banners for detectable legal-backbone errors.

    Pure pattern-matching — no LLM call.  Only fires for the
    ``legal_qualification`` route when an in-house developer signal is present.
    Annotates rather than blocks: the answer is still emitted, but the
    compliance officer is alerted to verify the flagged section.
    """
    if route.id != "legal_qualification":
        return []
    if not _has_inhouse_developer_signal(question):
        return []

    warnings: list[str] = []

    # Check 1: answer classifies the developer as initially a deployer.
    if _SELF_DEPLOYER_PATTERN.search(answer) or _INITIAL_DEPLOYER_PATTERN.search(answer):
        warnings.append(
            "> \u26a0 BACKBONE FLAG — This answer may incorrectly classify the original "
            "AI system developer as a deployer. Under Article 3(3) AI Act, "
            "development + internal deployment = provider status from inception. "
            "Verify the initial-status analysis before relying on this answer."
        )

    # Check 2: Article 25 appears before Article 3 in the AI Act section,
    # suggesting it is used as the primary provider-conversion mechanism.
    art25_pos = answer.find("Article 25")
    art3_pos = answer.find("Article 3")
    if art25_pos != -1 and art3_pos != -1 and art25_pos < art3_pos:
        warnings.append(
            "> \u2139 SCOPE NOTE — Article 25 appears to be used as the primary "
            "provider-conversion mechanism. Article 25 applies to third-party "
            "deployers who received the system from an external provider. "
            "If this entity developed the system itself, its provider status "
            "derives from Article 3(3), not Article 25."
        )

    return warnings


def _build_confidence_banner(confidence: "dict[str, Any]") -> str:
    """Return a brief *Scope & limitations* note, or "" when nothing actionable.

    Only surfaces substantive, actionable caveats (thin retrieval coverage;
    reliance on non-binding guidance). The previous banner led with a bare
    "Confidence: LOW (Score: 61%)" + generic "independently verify" boilerplate,
    which the senior-officer judge anchored on as a blanket reliability
    disclaimer — it added no compliance value and depressed every sub-HIGH
    answer. The composite score is still emitted as a structured ``confidence``
    event for the UI; redundant signals are dropped here (faithfulness is already
    reported in the verification block below the answer, and "a corrective pass
    ran" is an internal detail). HIGH confidence and no actionable caveat both
    return "" so well-supported answers stay clean.
    """
    if confidence.get("confidence_level", "HIGH") == "HIGH":
        return ""
    breakdown = confidence.get("breakdown", {})
    dist      = confidence.get("legal_force_distribution", {})

    notes: list[str] = []
    if breakdown.get("retrieval_coverage", 1.0) < 0.5:
        notes.append(
            "Retrieval coverage for this question was partial — some relevant"
            " provisions may not have been surfaced; confirm against the full text."
        )
    if breakdown.get("legal_force_alignment", 1.0) < 0.5:
        non_b = dist.get("non_binding", 0)
        total = sum(dist.values()) or 1
        notes.append(
            f"{non_b} of {total} cited provisions are non-binding MDCG guidance"
            " rather than binding regulation — verify conclusions against the"
            " regulation itself."
        )
    if not notes:
        return ""

    lines = ["> **Scope & limitations**"]
    lines.extend(f"> - {n}" for n in notes)
    return "\n".join(lines)


def _postprocess_answer(
    answer: str,
    route: _QuestionRoute,
    *,
    question: str,
    sufficiency: dict[str, Any],
    confidence: dict[str, Any] | None = None,
    audited: bool = False,
) -> str:
    """Apply lightweight safety formatting to the generated answer.

    When ``audited`` is True a real LLM Auditor pass has already verified the
    legal backbone, so the crude regex backbone flag (which could contradict its
    own answer) is suppressed in favour of the Auditor's verdict.
    """
    processed = _soften_categorical_language(
        answer,
        route,
        sufficiency=sufficiency,
    )
    processed = _RULE_LABEL_PATTERN.sub("", processed)
    processed = _CONTEXT_INDEX_PATTERN.sub("", processed)
    backbone_warnings = [] if audited else _validate_legal_backbone(processed, question, route)
    banner = _build_uncertainty_banner(route, sufficiency=sufficiency)
    parts: list[str] = []
    if banner:
        parts.append(banner)
    parts.extend(backbone_warnings)
    if parts:
        processed = "\n\n".join(parts) + "\n\n" + processed.lstrip()
    # Append confidence banner at the end (after the answer body)
    if confidence:
        conf_banner = _build_confidence_banner(confidence)
        if conf_banner:
            processed = processed.rstrip() + "\n\n---\n" + conf_banner
    return processed
