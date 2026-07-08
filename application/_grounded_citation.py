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
#
# The model emits pointers in two shapes (its output format is not stable): the
# bare contract form ``[cite: <id>]`` and — despite the "no hyperlinks" rule — a
# markdown-link form ``[Article 43](cite:<id>)`` that would otherwise leak the
# internal id as a URL.  One regex catches both; the ``label`` group is the
# human-readable link text of the second form (empty for the first).
_POINTER_RE = re.compile(
    r"\[(?P<label>[^\]\n]*)\]\((?P<lkind>cite|quote):\s*(?P<lid>[A-Za-z0-9_.\-]+)\s*\)"
    r"|"
    r"\[(?P<kind>cite|quote):\s*(?P<id>[A-Za-z0-9_.\-]+)\s*\]"
)

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
    suppressed_ref_dups: list[str] = field(default_factory=list)
    global_ref_ids: list[str] = field(default_factory=list)


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


def _render_ref(ref: str, regulation: str) -> str:
    """Human-readable inline reference from a ``(display_ref, regulation)`` pair.

    This is the ONLY citation form a reader ever sees — an internal node id is
    never rendered.  Empty when there is no display reference to show.
    """
    parts = [ref] if ref else []
    if regulation:
        parts.append(regulation)
    return " ".join(parts) if parts else ""


def _render_cite(entry: _Entry) -> str:
    """Human-readable inline reference for a ``[cite:]`` pointer."""
    return _render_ref(entry.ref, entry.regulation)


# --- Fabricated-id canonicalisation -----------------------------------------
# The ingestion normalizer stores an article as `<celex>_art_23` but re-encodes
# its *paragraphs* under a different convention: `<celex>_023.001` (zero-padded
# article, dot, zero-padded paragraph, no `art_` prefix — see the `art_10 →
# "010.001"` transform in ingestion/parse/normalizer.py).  A model that only sees
# `art_23` on the article header naturally invents `art_23_1` for paragraph 1,
# which matches nothing.  This applies the pipeline's own transform so the guess
# maps back to the real id; the caller still verifies the result exists, so a
# genuinely invented paragraph maps to nothing and stays dropped (no laundering).
_FABRICATED_PARA_RE = re.compile(r"^(.+)_art_(\d+)([a-z]?)_(\d+)$")


def _canonicalize_id(node_id: str) -> str | None:
    """Map a fabricated ``…_art_N_M`` paragraph id to the real ``…_0NN.00M``."""
    m = _FABRICATED_PARA_RE.match(node_id)
    if not m:
        return None
    celex, art, alpha, para = m.groups()
    return f"{celex}_{int(art):03d}{alpha}.{int(para):03d}"


def _resolve_id(
    node_id: str,
    index: dict[str, _Entry],
    fallback_refs: dict[str, tuple[str, str]],
) -> tuple[str, "_Entry | None", "tuple[str, str] | None"]:
    """Resolve an id to ``(resolved_id, entry, ref_reg)``.

    Tries, in order: the retrieved *index*; the global *fallback_refs* map; then
    a canonicalised variant of a fabricated ``_art_N_M`` id against both.  Exactly
    one of ``entry`` / ``ref_reg`` is set on success; both are ``None`` when the
    id exists nowhere (→ the caller drops it).
    """
    entry = index.get(node_id)
    if entry is not None:
        return node_id, entry, None
    if node_id in fallback_refs:
        return node_id, None, fallback_refs[node_id]
    canon = _canonicalize_id(node_id)
    if canon:
        canon_entry = index.get(canon)
        if canon_entry is not None:
            return canon, canon_entry, None
        if canon in fallback_refs:
            return canon, None, fallback_refs[canon]
    return node_id, None, None


# --- Husk cleanup ------------------------------------------------------------
# A pointer whose id is unknown even to the global reference map (a paragraph or
# point the model invented, which exists nowhere) is dropped — but naively
# returning "" leaves the model's scaffolding behind as "as required by ****",
# an empty "> " blockquote, or an empty table cell.  Dropped pointers instead
# emit this private-use sentinel; ``_clean_husks`` then removes the sentinel
# together with the emphasis/parenthetical wrapper and any dangling connector
# phrase ("as required by", "set out in", …) that introduced it, so the prose
# closes up cleanly rather than exposing a broken citation.
_DROP = "DROP"  # private-use sentinel, never present in real content

_CONNECTOR = (
    r"(?:as\s+)?(?:required\s+by|set\s+out\s+in|provided\s+for\s+in|"
    r"referred\s+to\s+in|governed\s+by|implied\s+by|derived\s+from|"
    r"pursuant\s+to|in\s+accordance\s+with|under|in|see)"
)


def _clean_husks(text: str) -> str:
    """Remove drop sentinels and the empty markup/connector husks around them."""
    d = re.escape(_DROP)
    # 0. Defensive backstop: strip any citation-URL syntax that reached the
    #    output through a path the pointer resolver did not rewrite (e.g. the
    #    structured body, or an unusual format) so an internal node id can NEVER
    #    reach the reader.  Keep the human-readable link label; drop the id.
    text = re.sub(r"\[([^\]\n]*)\]\((?:cite|quote):[^)\n]*\)", r"\1", text)
    text = re.sub(r"\((?:cite|quote):[^)\n]*\)", "", text)
    # 1. sentinel wrapped in bold/italic or parentheses/brackets → strip wrapper
    text = re.sub(r"\*{1,3}\s*" + d + r"\s*\*{1,3}", _DROP, text)
    text = re.sub(r"[(\[]\s*" + d + r"\s*[)\]]", "", text)
    # 2. a leading connector phrase (and optional preceding comma / "and") that
    #    only introduced the now-dropped reference → remove it with the sentinel;
    #    repeated so "… in X and Y" with both dropped collapses fully.
    for _ in range(3):
        text, n = re.subn(
            r"(?:[,;:]\s*)?(?:\band\b\s*)?" + _CONNECTOR + r"\s*" + d,
            "", text, flags=re.I,
        )
        if not n:
            break
    # 3. any bare sentinel remnant
    text = text.replace(_DROP, "")
    # 4. tidy the punctuation / empty wrappers a drop can leave behind
    text = re.sub(r"\(\s*\)|\[\s*\]|\*\*\s*\*\*|\*\s*\*", "", text)
    text = re.sub(r"[ \t]+([.,;:)])", r"\1", text)      # space before punctuation
    text = re.sub(r"([(\[])[ \t]+", r"\1", text)        # space after open bracket
    text = re.sub(r",\s*([.;:])", r"\1", text)          # orphan comma before stop
    # 5. drop a blockquote line that lost its only content
    text = re.sub(r"(?m)^>[ \t]*$\n?", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# --- Adjacent-reference de-duplication (deterministic backstop) --------------
# The prompt tells the model the citation owns the provision reference, so it
# should not also name the Article/Annex in prose.  That holds most of the time;
# the residual case is a model that writes "…as required by Article 43 [cite:…]"
# anyway, whose rendered cite then repeats "Article 43".  When the same reference
# already sits in the prose immediately before the pointer, the render paths drop
# the pointer's visible copy so the reference shows once (the pointer is still
# recorded in ``cited_ids`` for the audit trail).  Conservative by design: only a
# near, whole-token match suppresses; anything else renders normally.
_REF_WINDOW = 48  # chars of preceding prose scanned for a duplicate reference


def _norm_ref(text: str) -> str:
    """Lowercase, drop markdown emphasis, fold the dash family, collapse spaces."""
    text = text.replace("*", "").replace("_", " ")
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        text = text.replace(dash, "-")
    return " ".join(text.lower().split())


def _ref_already_in_prose(preceding: str, ref: str) -> bool:
    """True when *ref* is already named at the tail of the *preceding* prose.

    Whole-token match (a trailing negative lookahead) so "Article 4" does not
    match inside "Article 43".  Scans only the last ``_REF_WINDOW`` chars, so a
    far-away mention of the same article does not trigger suppression.
    """
    if not ref:
        return False
    tail = _norm_ref(preceding[-_REF_WINDOW:])
    pattern = re.escape(_norm_ref(ref)) + r"(?![0-9a-z])"
    return re.search(pattern, tail) is not None


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
    answer: str,
    index: dict[str, _Entry],
    fallback_refs: dict[str, tuple[str, str]] | None = None,
) -> ResolveResult:
    """Rewrite ``[cite:]``/``[quote:]`` pointers in *answer* using *index*.

    Resolved ``[quote:]`` -> verbatim block; ``[cite:]`` -> human ref.  When a
    pointer's id is not in the retrieved bag but *is* a real provision known to
    ``fallback_refs`` (the retriever's global ``{id: (ref, regulation)}`` map),
    its human-readable reference is still rendered — so a real article the model
    cited but retrieval did not surface reads as "Article 25 EU AI Act", not an
    empty husk.  Only an id that exists **nowhere** is dropped; the sentinel it
    leaves is cleaned away (with its "as required by …" scaffolding) so the
    reader never sees ``****`` or a raw node id.
    """
    fallback_refs = fallback_refs or {}
    quoted: list[str] = []
    cited: list[str] = []
    unresolved: list[str] = []
    deduped: list[str] = []
    suppressed: list[str] = []
    global_resolved: list[str] = []
    seen_quotes: set[str] = set()
    char_cap = quote_char_cap()

    def _sub(m: re.Match[str]) -> str:
        # Either alternative of _POINTER_RE matched; pick the populated groups.
        if m.group("lid") is not None:
            kind, raw_id, label = m.group("lkind"), m.group("lid"), m.group("label")
        else:
            kind, raw_id, label = m.group("kind"), m.group("id"), None
        node_id, entry, ref_reg = _resolve_id(raw_id, index, fallback_refs)
        if entry is None:
            # Not in the retrieved bag — a real provision known to the global map
            # (possibly via canonicalising a fabricated id) renders its reference;
            # an id that exists nowhere is dropped and husk-cleaned.  For the
            # markdown-link form we keep its human-readable label rather than a
            # husk, since the reader loses nothing and no internal id leaks.
            if ref_reg is None:
                unresolved.append(raw_id)
                return label if label else _DROP
            ref, reg = ref_reg
            cited.append(node_id)
            global_resolved.append(node_id)
            if _ref_already_in_prose(m.string[: m.start()], ref):
                suppressed.append(node_id)
                return ""
            return _render_ref(ref, reg)
        # Dedupe: a repeat quote of an already-quoted node renders as a cite,
        # so the reader never sees the same verbatim block twice.
        if kind == "quote" and node_id not in seen_quotes:
            seen_quotes.add(node_id)
            quoted.append(node_id)
            return _render_quote(entry, char_cap)
        if kind == "quote":
            deduped.append(node_id)
        cited.append(node_id)
        # Drop the visible reference when the model already named this provision
        # in the prose right before the pointer (still recorded above).
        if _ref_already_in_prose(m.string[: m.start()], entry.ref):
            suppressed.append(node_id)
            return ""
        return _render_cite(entry)

    rendered = _clean_husks(_POINTER_RE.sub(_sub, answer))
    return ResolveResult(
        text=rendered.strip(),
        quoted_ids=quoted,
        cited_ids=cited,
        unresolved_ids=unresolved,
        deduped_ids=deduped,
        suppressed_ref_dups=suppressed,
        global_ref_ids=global_resolved,
    )
