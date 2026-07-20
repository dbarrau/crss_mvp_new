"""LLM-assisted residual quote repair — the strict tier (``CRSS_FAITHFULNESS_CHECK=2``).

The deterministic repair in ``_faithfulness.repair_and_redact`` fixes the
offenders whose true source verification already located: a paraphrase becomes
the source's exact sentence run, a misattributed citation is re-pointed. What
falls through is exactly the residue the v8-panel eval isolated (HQ_006/009/
016/036): the model's *own analysis* wrapped in quotation marks (not legal text
at all, so it grounds nowhere), quotations typed from training memory with no
retrieved source behind them, and multi-provision concatenation dumps. Those
need a judgment call, not better string matching — so each residual offender
gets ONE narrowly-scoped LLM decision:

- ``replace`` — copy the correct wording from a supplied source excerpt. Applied
  **only** if the returned text verifies as an *exact* substring of the retrieved
  corpus (same normalisation as the checker), so a hallucinated "replacement" is
  rejected outright.
- ``demote``  — strip the quotation marks: the span is the answer's own prose
  wrongly presented as quotation. The inner text is kept byte-for-byte; no new
  words enter the answer.
- ``delete``  — replace the span with the redaction marker: it presents itself
  as legal wording that no retrieved source supports.

Safety invariant: the model never authors free text into the answer. ``replace``
is gated on exact-substring verification, ``demote``/``delete`` are pure string
surgery on existing text, and any parse failure, API error or rejected
replacement simply leaves the quote in place — the caller's final deterministic
pass (``verify.py``) redacts it exactly as flag mode (mode 1) would have. The
worst case is therefore byte-identical to today's behaviour.

The repair model is ``CRSS_REPAIR_MODEL`` (default ``mistral-medium-latest``);
one call per offending quote, capped at ``_MAX_LLM_REPAIRS`` per answer.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from application._faithfulness import (
    FaithfulnessReport,
    Quote,
    _article_ref_parent_chain,
    _best_sentence_run,
    _build_corpus,
    _build_raw_sources,
    _build_sources,
    _citation_ref_span,
    _MIN_QUOTE_LEN,
    _nearest_citation_ref,
    _REDACTION_MARKER,
    _unique_grounding_ref,
    _WHITESPACE_RE,
    grounding_verdict,
)

logger = logging.getLogger(__name__)

_MAX_LLM_REPAIRS = 4          # offending quotes adjudicated per answer
_CONTEXT_CHARS = 260          # answer text shown on each side of the quote
_SOURCE_WINDOW_CHARS = 1600   # source excerpt size (centred on best match)
_MAX_CANDIDATE_SOURCES = 3

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _repair_model() -> str:
    return os.environ.get("CRSS_REPAIR_MODEL", "mistral-medium-latest")


def _default_client():
    from mistralai.client import Mistral
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def _resolve_source_key(ref: str | None, raw_sources: dict[str, tuple[str, str]]) -> str | None:
    """Resolve *ref* into a raw_sources key, walking the parent chain like
    the deterministic repair does (Article 43(4) → Article 43)."""
    if not ref:
        return None
    if ref in raw_sources:
        return ref
    for parent in _article_ref_parent_chain(ref):
        if parent in raw_sources:
            return parent
    return None


def _source_window(raw_text: str, quote_text: str) -> str:
    """Excerpt of *raw_text* centred on its best match for *quote_text*."""
    if len(raw_text) <= _SOURCE_WINDOW_CHARS:
        return raw_text
    _ratio, run = _best_sentence_run(quote_text, raw_text)
    if run:
        probe = run[:80]
        # The run is whitespace-collapsed; search on a collapsed copy so the
        # probe still locates its origin, then map back via proportion.
        collapsed = _WHITESPACE_RE.sub(" ", raw_text)
        idx = collapsed.find(probe)
        if idx >= 0:
            approx = int(idx / max(len(collapsed), 1) * len(raw_text))
            lo = max(0, approx - _SOURCE_WINDOW_CHARS // 4)
            return raw_text[lo : lo + _SOURCE_WINDOW_CHARS]
    return raw_text[:_SOURCE_WINDOW_CHARS]


def _candidate_sources(
    quote: Quote,
    answer: str,
    raw_sources: dict[str, tuple[str, str]],
    source_map: dict[str, str],
) -> list[tuple[str, str]]:
    """Rank candidate sources for a flagged quote: true grounding provision
    first (misattributed case), then the provision the answer cites next to
    the quote, then the best fuzzy matches across the whole retrieved bag."""
    ordered: list[str] = []

    true_ref = _resolve_source_key(_unique_grounding_ref(quote.text, source_map), raw_sources)
    cited_ref = _resolve_source_key(
        _nearest_citation_ref(answer, quote.start, quote.end), raw_sources
    )
    for key in (true_ref, cited_ref):
        if key and key not in ordered:
            ordered.append(key)

    if len(ordered) < _MAX_CANDIDATE_SOURCES:
        scored = sorted(
            (
                (_best_sentence_run(quote.text, raw)[0], key)
                for key, (_pretty, raw) in raw_sources.items()
                if key not in ordered
            ),
            reverse=True,
        )
        for ratio, key in scored:
            if ratio <= 0.25 or len(ordered) >= _MAX_CANDIDATE_SOURCES:
                break
            ordered.append(key)

    return [
        (raw_sources[k][0], _source_window(raw_sources[k][1], quote.text))
        for k in ordered
    ]


def _build_prompt(quote: Quote, answer: str, candidates: list[tuple[str, str]]) -> str:
    lo = max(0, quote.start - _CONTEXT_CHARS)
    hi = min(len(answer), quote.end + _CONTEXT_CHARS)
    context = (
        answer[lo : quote.start]
        + " >>>FLAGGED>>> "
        + answer[quote.start : quote.end]
        + " <<<FLAGGED<<< "
        + answer[quote.end : hi]
    )
    src_lines = [
        f"[{i}] {pretty}:\n{window}" for i, (pretty, window) in enumerate(candidates, 1)
    ] or ["(no retrieved source resembles this quotation)"]
    return (
        "You repair ONE quotation in a draft EU-compliance answer. The quotation "
        "FAILED deterministic verification against the retrieved legal sources.\n\n"
        f"FLAGGED QUOTATION (inner text):\n{quote.text}\n\n"
        f"ANSWER CONTEXT (flagged span marked >>>FLAGGED>>> ... <<<FLAGGED<<<):\n"
        f"{context}\n\n"
        "RETRIEVED SOURCE EXCERPTS:\n" + "\n\n".join(src_lines) + "\n\n"
        "Choose exactly ONE action:\n"
        '- "replace": ONLY if the quotation clearly intends to quote one of the '
        "excerpts above AND you can copy the correct contiguous wording EXACTLY, "
        "character for character, from one excerpt (at least 40 characters). "
        "Never compose, merge or adjust wording.\n"
        '- "demote": the flagged span is the answer\'s own analysis or summary '
        "that was wrongly wrapped in quotation marks; keep the text, drop the "
        "quote marks.\n"
        '- "delete": the span presents itself as legal wording but no excerpt '
        "supports it.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"action": "replace"|"demote"|"delete", "replacement": "<exact copy, '
        'or empty>", "source": <excerpt number or null>}'
    )


def _parse_decision(raw: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        decision = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(decision, dict):
        return None
    if decision.get("action") not in {"replace", "demote", "delete"}:
        return None
    return decision


def llm_repair_residuals(
    answer: str,
    report: FaithfulnessReport,
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> tuple[str, list[str]]:
    """Adjudicate the residual offenders in *report*; return ``(answer, notes)``.

    *report* must be freshly computed on *answer* (offsets are trusted). Quotes
    with no applicable/valid decision are left untouched for the caller's final
    redaction pass — this function never worsens the mode-1 outcome.
    """
    offenders: list[Quote] = list(report.unverified) + list(report.misattributed)
    if not offenders:
        return answer, []
    offenders = sorted(offenders, key=lambda q: q.start)[:_MAX_LLM_REPAIRS]

    if client is None:
        client = _default_client()
    model = model or _repair_model()

    corpus = _build_corpus(provisions, definitions)
    raw_sources = _build_raw_sources(provisions, definitions)
    _sources, source_map, _unkeyed = _build_sources(provisions, definitions)

    misattributed_spans = {(q.start, q.end) for q in report.misattributed}
    edits: list[tuple[int, int, str]] = []
    notes: list[str] = []
    for q in offenders:
        try:
            candidates = _candidate_sources(q, answer, raw_sources, source_map)
            resp = client.chat.complete(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": _build_prompt(q, answer, candidates)}],
            )
            decision = _parse_decision(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 — one quote must not sink the pass
            logger.warning("LLM quote repair failed for one quote: %s", exc)
            continue
        if decision is None:
            continue

        action = decision["action"]
        if action == "replace":
            replacement = _WHITESPACE_RE.sub(
                " ", str(decision.get("replacement") or "")
            ).replace('"', "'").strip()
            # Hard gate: the replacement must be an EXACT corpus substring of
            # quotable length, or it does not enter the answer.
            if (
                len(replacement) >= _MIN_QUOTE_LEN
                and grounding_verdict(replacement, corpus) == "exact"
            ):
                edits.append((q.start, q.end, "“" + replacement + "”"))
                src_i = decision.get("source")
                src_pretty = (
                    candidates[src_i - 1][0]
                    if isinstance(src_i, int) and 1 <= src_i <= len(candidates)
                    else None
                )
                # A misattributed offender carries a WRONG citation label next
                # to it: replacing only the quote text would re-flag on the
                # final pass (real text, still displaced). Re-point the label
                # to the chosen source — same mechanics as the deterministic
                # repoint in repair_and_redact.
                if src_pretty and (q.start, q.end) in misattributed_spans:
                    ref_span = _citation_ref_span(answer, q.start, q.end)
                    if ref_span:
                        edits.append((ref_span[0], ref_span[1], src_pretty))
                        notes.append(
                            f"citation corrected: the quoted text is from {src_pretty}"
                        )
                notes.append(
                    f"quote corrected to the exact text of {src_pretty or 'its source'}"
                )
            else:
                logger.info(
                    "LLM repair: replacement rejected (not an exact corpus "
                    "substring); quote left for redaction."
                )
        elif action == "demote":
            # Keep the inner text byte-for-byte; only the quote marks go.
            edits.append((q.start, q.end, q.text))
            notes.append(
                "quotation marks removed from a span that was the answer's own "
                "wording, not a source quote"
            )
        else:  # delete
            edits.append((q.start, q.end, _REDACTION_MARKER))
            notes.append("an unverifiable quotation was removed")

    if not edits:
        return answer, []
    repaired = answer
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        if 0 <= start < end <= len(repaired):
            repaired = repaired[:start] + replacement + repaired[end:]
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)
    return repaired.strip(), notes