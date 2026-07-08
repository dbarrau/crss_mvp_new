"""Structured grounded answer — the hard-enforcement form of the grounded
generation contract (see ``docs/grounded_generation_contract.md``).

Where ``_grounded_citation`` resolves inline ``[cite:]``/``[quote:]`` pointers in
a free-text completion, this module drives Mistral **structured outputs**
(``chat.parse`` with a pydantic schema): the model returns a ``GroundedAnswer``
whose quotations and citations live in a dedicated ``citations`` channel keyed by
node id — there is no field in which the model can author verbatim quote *text*.
The body references each citation by an opaque ``[[marker]]`` token; this renderer
substitutes the verbatim source text (for ``quote``) or the human reference (for
``cite``) from the retrieved bag.

Validated on the MDCG 2019-11 question: structured mode drove model-authored
``>`` blockquotes to zero (inline pointers left 3–9), with all node ids
resolving.  A residual literal quote can still leak into the free-text ``body``;
that is caught by the faithfulness net downstream, exactly as before.

The renderer is deterministic (no LLM, no I/O) and reuses the id→text index and
the quote/cite rendering from ``_grounded_citation``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from application._grounded_citation import (
    _DROP,
    _Entry,
    _cite_ref,
    _clean_husks,
    _ref_already_in_prose,
    _render_quote,
    _render_ref,
    _resolve_id,
    build_pointer_index,  # re-exported for callers assembling the index
    quote_char_cap,
)

__all__ = [
    "Citation",
    "GroundedAnswer",
    "RenderResult",
    "render_grounded_answer",
    "build_pointer_index",
]

# Body references a citation as [[marker]]; markers are short opaque tokens.
_MARKER_RE = re.compile(r"\[\[\s*([A-Za-z0-9_.\-]+)\s*\]\]")


class Citation(BaseModel):
    """One entry in the structured citation channel."""

    marker: str = Field(
        description="short unique token like q1 or c2; appears in body as [[q1]]"
    )
    node_id: str = Field(
        description="EXACT id copied from the context header (e.g. 32017R0745_art_10 "
        "or a paragraph's id) — never a display reference like 'Article 10'"
    )
    mode: Literal["quote", "cite"] = Field(
        description="'quote' renders the node's verbatim text; 'cite' renders its reference"
    )


class GroundedAnswer(BaseModel):
    """The structured answer the model returns under ``chat.parse``."""

    body: str = Field(
        description=(
            "The full markdown compliance answer. Every quotation or legal citation "
            "appears ONLY as a [[marker]] token — NEVER as literal quoted text, a "
            "'>' block, or a URL. Paraphrase freely; delegate all verbatim text and "
            "all provision references to markers."
        )
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="one entry per distinct [[marker]] used in body",
    )


@dataclass
class RenderResult:
    """Outcome of :func:`render_grounded_answer`."""

    text: str
    quoted_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    unresolved_markers: list[str] = field(default_factory=list)
    unresolved_ids: list[str] = field(default_factory=list)
    deduped_ids: list[str] = field(default_factory=list)
    suppressed_ref_dups: list[str] = field(default_factory=list)
    global_ref_ids: list[str] = field(default_factory=list)


def render_grounded_answer(
    answer: GroundedAnswer,
    index: dict[str, _Entry],
    fallback_refs: dict[str, tuple[str, str]] | None = None,
) -> RenderResult:
    """Substitute ``[[marker]]`` tokens in *answer.body* using its citations.

    A citation whose ``node_id`` is not in the retrieved *index* falls back to
    ``fallback_refs`` (the retriever's global ``{id: (ref, regulation)}`` map) so
    a real but un-retrieved provision still renders its human-readable reference.
    A marker with no citation entry, or a ``node_id`` that exists **nowhere**, is
    dropped and its empty scaffolding cleaned away — the reader never sees an
    empty ``****`` husk or a raw internal node id.
    """
    fallback_refs = fallback_refs or {}
    cite_map = {c.marker: c for c in answer.citations}
    quoted: list[str] = []
    cited: list[str] = []
    unresolved_markers: list[str] = []
    unresolved_ids: list[str] = []
    deduped: list[str] = []
    suppressed: list[str] = []
    global_resolved: list[str] = []
    seen_quotes: set[str] = set()
    char_cap = quote_char_cap()

    def _sub(m: re.Match[str]) -> str:
        marker = m.group(1)
        citation = cite_map.get(marker)
        if citation is None:
            unresolved_markers.append(marker)
            return _DROP
        node_id, entry, ref_reg = _resolve_id(
            citation.node_id, index, fallback_refs
        )
        if entry is None:
            # Not in the retrieved bag — a real provision known to the global map
            # (possibly via canonicalising a fabricated id) renders its reference;
            # an id that exists nowhere is dropped and husk-cleaned.
            if ref_reg is None:
                unresolved_ids.append(citation.node_id)
                return _DROP
            ref, reg = ref_reg
            disp = _cite_ref(node_id, ref)
            cited.append(node_id)
            global_resolved.append(node_id)
            if _ref_already_in_prose(m.string[: m.start()], disp):
                suppressed.append(node_id)
                return ""
            return _render_ref(disp, reg)
        # Dedupe: a repeat quote of an already-quoted node renders as a cite,
        # so a section quoted for several classes is not dumped verbatim each time.
        if citation.mode == "quote" and node_id not in seen_quotes:
            seen_quotes.add(node_id)
            quoted.append(node_id)
            return _render_quote(entry, char_cap)
        if citation.mode == "quote":
            deduped.append(node_id)
        cited.append(node_id)
        # Article-anchored reference ("Article 23(1)"), not the child's bare
        # display_ref; drop it when the prose already named the provision.
        disp = _cite_ref(node_id, entry.ref)
        if _ref_already_in_prose(m.string[: m.start()], disp):
            suppressed.append(node_id)
            return ""
        return _render_ref(disp, entry.regulation)

    rendered = _clean_husks(_MARKER_RE.sub(_sub, answer.body))
    return RenderResult(
        text=rendered.strip(),
        quoted_ids=quoted,
        cited_ids=cited,
        unresolved_markers=unresolved_markers,
        unresolved_ids=unresolved_ids,
        deduped_ids=deduped,
        suppressed_ref_dups=suppressed,
        global_ref_ids=global_resolved,
    )
