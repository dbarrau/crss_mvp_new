"""Grounded-citation resolver — the deterministic core of the grounded
generation contract (see ``docs/grounded_generation_contract.md``).

The model is asked to emit **pointers**, never verbatim source text:

    [cite: <node_id>]   -> attach a claim to a provision (renders the ref)
    [quote: <node_id>]  -> render that node's exact stored text (the model
                           never types the quoted words)

This module resolves those pointers against the retrieved bag.  It is the
inverse of ``_faithfulness``: instead of searching the corpus for text the model
authored, it fetches text *keyed by* the id the model pointed at — so fabricated
quotation is impossible by construction.

The pointer key is the stable **node id** (``article_id`` for a provision,
``id`` for a child), never ``display_ref`` (which is ``None``/non-unique on many
nodes — the citation-ambiguity trap).  Fully deterministic; no LLM, no I/O.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# node ids look like "32017R0745_art_10", "32017R0745_010.014", "MDCG_2019_11_s3".
# Permit word chars plus the '.' and '-' that appear in leaf ids.
_POINTER_RE = re.compile(r"\[(cite|quote):\s*([A-Za-z0-9_.\-]+)\s*\]")

# --- Quote economy ----------------------------------------------------------
# Structured output moved verbatim expansion out of the model into the renderer,
# which hid the cost of quoting from the model — so it over-quotes (whole sections,
# duplicated).  Economy of quoting now belongs wherever expansion belongs (here),
# not to the model.  Two deterministic levers, applied by both render paths:
#   * dedupe   — a repeated quote of an already-quoted node renders as a cite;
#   * char cap — a soft backstop truncating an over-long single quote at a
#                sentence boundary.  0 disables.  See docs/grounded_generation_contract.md.
_DEFAULT_QUOTE_CHAR_CAP = 600


def quote_char_cap() -> int:
    """Per-quote character cap (soft economy backstop); 0 disables. Env-tunable."""
    try:
        return max(0, int(os.environ.get("CRSS_QUOTE_CHAR_CAP", _DEFAULT_QUOTE_CHAR_CAP)))
    except (TypeError, ValueError):
        return _DEFAULT_QUOTE_CHAR_CAP


def _cap_quote_text(text: str, char_cap: int) -> str:
    """Truncate an over-long quote to a leading excerpt at a sentence boundary.

    Backstop only — deduping and pointing at leaf paragraphs are the primary
    economy levers; this catches the residual whole-section quote.
    """
    text = text.strip()
    if char_cap <= 0 or len(text) <= char_cap:
        return text
    cut = text[:char_cap]
    boundary = cut.rfind(". ")
    if boundary > char_cap // 2:
        cut = cut[: boundary + 1]
    return cut.rstrip() + " […]"


@dataclass(frozen=True)
class _Entry:
    text: str
    ref: str
    regulation: str
    binding_force: str | None


@dataclass
class ResolveResult:
    """Outcome of :func:`resolve_pointers`."""

    text: str
    quoted_ids: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    unresolved_ids: list[str] = field(default_factory=list)
    deduped_ids: list[str] = field(default_factory=list)


def build_pointer_index(
    provisions: list[dict[str, Any]],
    definitions: list[dict[str, Any]] | None = None,
) -> dict[str, _Entry]:
    """Map every retrieved node id to the verbatim text the renderer may emit.

    Keys are the exact ids the context renderer exposes to the model:
    ``article_id`` for each provision and ``id`` for each child leaf.  A child id
    that repeats across provisions (shared ancestors like a Chapter) keeps its
    first binding; leaf paragraphs are unique so this does not lose quotable text.
    """
    index: dict[str, _Entry] = {}
    for p in provisions or []:
        if not p:
            continue
        regulation = p.get("regulation", "") or ""
        force = p.get("binding_force")
        aid = p.get("article_id")
        if aid and aid not in index:
            index[aid] = _Entry(
                text=(p.get("article_text") or "").strip(),
                ref=p.get("article_ref", "") or "",
                regulation=regulation,
                binding_force=force,
            )
        for c in p.get("children") or []:
            cid = c.get("id")
            if not cid or cid in index:
                continue
            text = (c.get("raw_text") or c.get("text") or "").strip()
            index[cid] = _Entry(
                text=text,
                ref=c.get("ref", "") or "",
                regulation=regulation,
                binding_force=c.get("binding_force") or force,
            )
    # Definitions carry no node id today; they are addressed by term/ref and are
    # already verified by the faithfulness net.  Left out of the pointer vocabulary
    # for v1 rather than inventing an unstable key.
    return index


def _render_cite(entry: _Entry) -> str:
    """Human-readable inline reference for a ``[cite:]`` pointer."""
    parts = [entry.ref] if entry.ref else []
    if entry.regulation:
        parts.append(entry.regulation)
    return " ".join(parts) if parts else ""


def _render_quote(entry: _Entry, char_cap: int = 0) -> str:
    """Verbatim block for a ``[quote:]`` pointer, capped to *char_cap* chars.

    Empty-text nodes (e.g. a structural ancestor pointed at by mistake) fall back
    to their reference so the answer never shows an empty blockquote.
    """
    if not entry.text:
        return _render_cite(entry)
    text = _cap_quote_text(entry.text, char_cap)
    return "> " + text.replace("\n", "\n> ")


def resolve_pointers(
    answer: str, index: dict[str, _Entry]
) -> ResolveResult:
    """Rewrite ``[cite:]``/``[quote:]`` pointers in *answer* using *index*.

    Resolved ``[quote:]`` -> verbatim block; ``[cite:]`` -> human ref.  A pointer
    whose id is not in the retrieved bag is **dropped** (never rendered as an
    unsupported quote) and reported in ``unresolved_ids``.
    """
    quoted: list[str] = []
    cited: list[str] = []
    unresolved: list[str] = []
    deduped: list[str] = []
    seen_quotes: set[str] = set()
    char_cap = quote_char_cap()

    def _sub(m: re.Match[str]) -> str:
        kind, node_id = m.group(1), m.group(2)
        entry = index.get(node_id)
        if entry is None:
            unresolved.append(node_id)
            return ""
        # Dedupe: a repeat quote of an already-quoted node renders as a cite,
        # so the reader never sees the same verbatim block twice.
        if kind == "quote" and node_id not in seen_quotes:
            seen_quotes.add(node_id)
            quoted.append(node_id)
            return _render_quote(entry, char_cap)
        if kind == "quote":
            deduped.append(node_id)
        cited.append(node_id)
        return _render_cite(entry)

    rendered = _POINTER_RE.sub(_sub, answer)
    # Collapse whitespace artefacts left by dropped pointers.
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    rendered = re.sub(r" +\n", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return ResolveResult(
        text=rendered.strip(),
        quoted_ids=quoted,
        cited_ids=cited,
        unresolved_ids=unresolved,
        deduped_ids=deduped,
    )
