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


def _postprocess_answer(
    answer: str,
    route: _QuestionRoute,
    *,
    question: str,
    sufficiency: dict[str, Any],
) -> str:
    """Apply lightweight safety formatting to the generated answer."""
    processed = _soften_categorical_language(
        answer,
        route,
        sufficiency=sufficiency,
    )
    backbone_warnings = _validate_legal_backbone(processed, question, route)
    banner = _build_uncertainty_banner(route, sufficiency=sufficiency)
    parts: list[str] = []
    if banner:
        parts.append(banner)
    parts.extend(backbone_warnings)
    if parts:
        return "\n\n".join(parts) + "\n\n" + processed.lstrip()
    return processed
