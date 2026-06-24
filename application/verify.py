"""Post-generation verification — the single self-check stage over a draft answer.

Runs the deterministic guards that used to live as three separately bolted-on,
env-flagged blocks inside ``ask_stream`` (see PATCH_LEDGER C1/C2/C4/C5):

- **Citation-scope note (C4)** — when the question is scoped to a single
  regulation, flag cited Article/Annex/Recital refs that are absent from the
  retrieved context.
- **Faithfulness + attribution (C1/C2)** — verify every verbatim quote against
  the retrieved corpus; redact fabricated / misattributed / concatenated quotes
  and prepend a warning block.
- **Confidence (C5)** — the five-component composite, read from retrieval
  metadata + the faithfulness report.

Folding them into one stage gives the read path a single post-generation
verification surface (the substrate for proof-carrying answers, C2's direction)
and one place that owns the env flags. ``verify_answer`` is a pure function over
the draft answer + retrieved evidence; ``ask_stream`` yields the confidence
event from its result and is otherwise uninvolved.

This relocation is **behaviour-neutral**: the checks run in the same order
(scope → faithfulness → confidence), read the same env flags, redact and warn
identically, and compute confidence on the post-redaction answer exactly as
before. Known follow-up recorded in the ledger: the post-redaction faithfulness
recompute that feeds confidence is always clean (redaction removed the offending
quotes), so the confidence faithfulness component is a constant 1.0 — computing
the report once and reusing it would make that component meaningful, but that
changes confidence numbers and is left as a separate, user-approved step.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from application._faithfulness import (
    build_warning_block,
    check_faithfulness,
    faithfulness_mode,
    out_of_scope_citation_refs,
    remove_unverified_quotes,
)
from application._confidence import compute_confidence

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of the post-generation verification stage.

    ``answer`` is the (possibly redacted, warning-prefixed, scope-noted) text to
    show the user; ``confidence`` is the composite the agent yields as a
    ``confidence`` event.
    """

    answer: str
    confidence: dict[str, Any]


def _apply_citation_scope_note(
    answer: str,
    provisions: list[dict],
    *,
    target_celexes: set[str] | None,
    mentioned_regs: set[str],
) -> str:
    """C4 — append a note when cited refs fall outside a single-reg context."""
    if not (
        target_celexes
        and len(target_celexes) == 1
        and mentioned_regs
        and len(mentioned_regs) == 1
        and os.environ.get("CRSS_CITATION_SCOPE_CHECK", "1") != "0"
        and answer
    ):
        return answer
    scope_reg_name = next(iter(mentioned_regs))
    try:
        out_of_scope_refs = out_of_scope_citation_refs(answer, provisions)
        if out_of_scope_refs:
            scope_text = ", ".join(out_of_scope_refs[:30])
            answer += (
                "\n\n---\n> **⚠ Citation scope note:** "
                "The following citations are not present in the retrieved "
                f"context for this question (scoped to {scope_reg_name}): "
                f"{scope_text}. Please verify against the source provisions."
            )
            logger.info("Citation scope deterministic check flagged: %s", scope_text)
        else:
            logger.debug("Citation scope deterministic check: CLEAN")
    except Exception as exc:  # noqa: BLE001 — self-check is best-effort
        logger.warning("Citation scope self-check skipped: %s", exc)
    return answer


def _apply_faithfulness(
    answer: str,
    provisions: list[dict],
    definitions: list[dict],
    *,
    faith_mode: int,
) -> str:
    """C1/C2 — redact ungrounded/displaced quotes and prepend a warning block."""
    if not (faith_mode >= 1 and answer):
        return answer
    try:
        report = check_faithfulness(answer, provisions, definitions)
        if not report.ok or report.near_verbatim:
            # Redact only genuinely divergent quotes (near-verbatim ones are
            # grounded and stay); surface both tiers in the block.
            answer = remove_unverified_quotes(answer, report)
            block = build_warning_block(report)
            if block:
                answer = f"{block}\n\n{answer}"
            logger.info(
                "Faithfulness check: %d fabricated, %d misattributed, "
                "%d near-verbatim, of %d quote(s)",
                report.unverified_count,
                report.misattributed_count,
                report.near_verbatim_count,
                report.total_quotes,
            )
        else:
            logger.debug(
                "Faithfulness check: all %d quote(s) verified", report.total_quotes
            )
        if faith_mode == 2:
            logger.warning(
                "CRSS_FAITHFULNESS_CHECK=2 (strict) is not yet implemented; "
                "behaving as flag mode."
            )
    except Exception as exc:  # noqa: BLE001 — self-check is best-effort
        logger.warning("Faithfulness check skipped: %s", exc)
    return answer


def verify_answer(
    answer: str,
    *,
    provisions: list[dict],
    definitions: list[dict],
    role_provisions: list[dict],
    sufficiency: dict[str, Any],
    target_celexes: set[str] | None,
    mentioned_regs: set[str],
    role_specs: list[tuple[str, str]],
    corrective_actions: list[str],
    question: str,
) -> VerificationResult:
    """Run the post-generation verification stage over a draft answer.

    Order is fixed and behaviour-preserving: citation-scope note (C4) →
    faithfulness/attribution redaction (C1/C2) → composite confidence (C5). The
    confidence faithfulness input is recomputed on the post-redaction answer,
    exactly as the pre-fold pipeline did.
    """
    answer = _apply_citation_scope_note(
        answer,
        provisions,
        target_celexes=target_celexes,
        mentioned_regs=mentioned_regs,
    )

    faith_mode = faithfulness_mode(os.environ.get("CRSS_FAITHFULNESS_CHECK", "1"))
    answer = _apply_faithfulness(answer, provisions, definitions, faith_mode=faith_mode)

    # Confidence reads a faithfulness report computed on the (now redacted)
    # answer; see the module docstring for the known constant-component caveat.
    faith_report_for_conf = None
    if faith_mode >= 1 and answer:
        try:
            faith_report_for_conf = check_faithfulness(answer, provisions, definitions)
        except Exception:  # noqa: BLE001 — confidence degrades gracefully
            pass

    had_pointer_expansion = any(p.get("_pointer_expansion") for p in provisions)
    confidence = compute_confidence(
        sufficiency=sufficiency,
        provisions=provisions,
        faith_report=faith_report_for_conf,
        had_corrective_pass=bool(corrective_actions),
        had_pointer_expansion=had_pointer_expansion,
        had_role_provisions=bool(role_provisions),
        role_specs=role_specs,
        question=question,
        mentioned_regs=mentioned_regs,
    )
    logger.info(
        "Confidence: %s (%.1f%%)",
        confidence["confidence_level"],
        confidence["confidence_score"] * 100,
    )
    return VerificationResult(answer=answer, confidence=confidence)