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
    _Entry,
    _render_cite,
    _render_quote,
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


def render_grounded_answer(
    answer: GroundedAnswer, index: dict[str, _Entry]
) -> RenderResult:
    """Substitute ``[[marker]]`` tokens in *answer.body* using its citations.

    A marker with no citation entry, or a citation whose ``node_id`` is not in the
    retrieved *index*, is **dropped** (never rendered as an unsupported quote) and
    reported.  ``quote`` → verbatim block from the node; ``cite`` → human ref.
    """
    cite_map = {c.marker: c for c in answer.citations}
    quoted: list[str] = []
    cited: list[str] = []
    unresolved_markers: list[str] = []
    unresolved_ids: list[str] = []
    deduped: list[str] = []
    seen_quotes: set[str] = set()
    char_cap = quote_char_cap()

    def _sub(m: re.Match[str]) -> str:
        marker = m.group(1)
        citation = cite_map.get(marker)
        if citation is None:
            unresolved_markers.append(marker)
            return ""
        entry = index.get(citation.node_id)
        if entry is None:
            unresolved_ids.append(citation.node_id)
            return ""
        # Dedupe: a repeat quote of an already-quoted node renders as a cite,
        # so a section quoted for several classes is not dumped verbatim each time.
        if citation.mode == "quote" and citation.node_id not in seen_quotes:
            seen_quotes.add(citation.node_id)
            quoted.append(citation.node_id)
            return _render_quote(entry, char_cap)
        if citation.mode == "quote":
            deduped.append(citation.node_id)
        cited.append(citation.node_id)
        return _render_cite(entry)

    rendered = _MARKER_RE.sub(_sub, answer.body)
    # Collapse whitespace/markup artefacts left by dropped markers.
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    rendered = re.sub(r" +\n", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return RenderResult(
        text=rendered.strip(),
        quoted_ids=quoted,
        cited_ids=cited,
        unresolved_markers=unresolved_markers,
        unresolved_ids=unresolved_ids,
        deduped_ids=deduped,
    )
