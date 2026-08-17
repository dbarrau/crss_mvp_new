"""Verbatim provision display — an authoritative, LLM-free render of a directly
requested provision.

Why this exists: a "show me Article X" request must return the *actual* text from
the corpus, not an LLM reconstruction. When such a lookup was routed through the
generative path, the model paraphrased and *fabricated* base paragraphs (e.g. it
invented AI Act Article 6(3)–(7)) while presenting them as controlling law. This
path bypasses generation entirely: it renders the ordered ``HAS_PART`` subtree
verbatim (``retrieve_by_refs`` already attaches it). Deterministic; no LLM;
nothing to fabricate.

The graph is *consolidated* — amending acts are applied to the base nodes
(``consolidation.applier``), so the subtree already holds current law (inserted
paragraphs, replacements, deletions in place). The render therefore shows the
subtree as-is; a footer names the amending act(s) from the ``amended_by`` tag the
consolidation left on the affected nodes. There is no render-time splice.

Two public entry points, both re-exported from ``application.agent``:

* :func:`wants_verbatim_display` — conservative intent gate on the *original*
  question (so a display request is honoured even mid-conversation, unaffected by
  the history rewrite that otherwise reframes it).
* :func:`render_provision_display` — the render itself, or ``None`` when the
  provision can't be resolved (the caller then falls back to generation).
"""
from __future__ import annotations

import re

from application._config import (
    _REG_NAME_TO_CELEX,
    _detect_mentioned_regulations,
    _extract_provision_refs,
)
from domain.legislation_catalog import LEGISLATION

# ── intent detection ────────────────────────────────────────────────────────
# Fire ONLY for a pure display request naming exactly one provision in exactly
# one regulation. Anything analytical (how does it apply, obligations, is it
# high-risk, a scenario) must go to the generative path.
_DISPLAY_VERB = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:show(?:\s+me)?|display|give\s+me|quote|print|reproduce|read(?:\s+me)?|"
    r"pull\s+up|render|let'?s\s+see|what\s+does|what\s+is|what'?s)\b",
    re.I,
)
_ANALYTICAL = re.compile(
    r"\b(?:how|why|appl(?:y|ies|ication)|oblig\w*|comply|complian\w*|classif\w*|"
    r"high[-\s]?risk|liab\w*|responsib\w*|assess\w*|scenario|integrat\w*|require\w*|"
    r"if\s+a\b|for\s+a\b|my\b|does\s+it\b|would\s+it\b)",
    re.I,
)
_MAX_DISPLAY_WORDS = 16


def wants_verbatim_display(question: str) -> tuple[str, str] | None:
    """Return ``(ref, celex)`` to render verbatim, or ``None``.

    Runs on the ORIGINAL user question (not the history-rewritten one), so a
    literal "show me Article X" is honoured even when asked after other turns.
    Deliberately narrow: a display verb, no analytical language, a short
    question, and exactly one provision reference in exactly one regulation.
    """
    if not question:
        return None
    q = question.strip()
    if not _DISPLAY_VERB.search(q) or _ANALYTICAL.search(q):
        return None
    if len(q.split()) > _MAX_DISPLAY_WORDS:
        return None
    refs = _extract_provision_refs(q)
    if len(refs) != 1:
        return None
    regs = _detect_mentioned_regulations(q)
    if len(regs) != 1:
        return None
    # `_detect_mentioned_regulations` yields the human name ("EU AI Act"); the
    # render needs the CELEX the graph is keyed on.
    name = next(iter(regs))
    celex = _REG_NAME_TO_CELEX.get(name, name if name in LEGISLATION else None)
    if not celex:
        return None
    return refs[0], celex


# ── rendering ────────────────────────────────────────────────────────────────

# Paragraph-level kinds render flush and reset the indent origin; everything else
# (points, roman sub-items, dash indents) nests beneath its paragraph.
_PARA_LEVEL = frozenset({"article", "paragraph", "subparagraph"})
_INDENT = "    "    # one nesting level: four NON-BREAKING spaces (U+00A0); HTML keeps them, Markdown ignores them


def _render_nodes(nodes: list[dict]) -> str:
    """Blockquote render: one markdown *paragraph* per unit, with a blank '>' line
    between each so every paragraph/point becomes its own ``<p>``.

    Enumerators are **bolded** ("**1.**", "**(a)**"). That is load-bearing, not
    cosmetic: it (i) stops ``marked`` parsing a leading "1." as an ordered-list
    item, and (ii) stops the front-end legal-list normaliser splitting a point at
    an inline cross-reference (e.g. the "point (a)" inside point (b)) — a bolded
    leading marker is preceded by '*', so it is no longer counted as an
    enumerator, and the lone inline reference then trips the normaliser's
    "needs >=2 markers" guard. No leading-space indentation (4 spaces would become
    a code block); the bold enumerators carry the hierarchy instead."""
    lines: list[str] = []
    first = True
    baseline = 0                                       # depth of the nearest paragraph-level ancestor
    for nd in nodes:
        depth = nd.get("depth") or 0
        kind = nd.get("kind")
        if kind in _PARA_LEVEL:
            baseline = depth                           # paragraphs/subparagraphs reset the indent origin
        if depth == 0:
            continue                                   # heading is rendered above the quote
        text = re.sub(r"\s+", " ", (nd.get("text") or "").replace("\xa0", " ")).strip()
        if not text:
            continue
        number = (nd.get("number") or "").strip()
        if kind in _PARA_LEVEL:
            body = re.sub(r"^(\d+[a-z]?\.)\s*", r"**\1** ", text)   # bold the leading N. / Na.
        elif kind == "indent":
            body = f"— {text}"
        elif number:
            body = f"**({number})** {text}"
        else:
            body = text
        # Points/sub-points nest under their paragraph via NON-ASCII (nbsp) indent
        # — HTML preserves it, but Markdown does not read it as a code block / list,
        # so the hierarchy shows (e.g. Art 7(2)(k)(i)/(ii) under (k)) without the
        # mangling ASCII indentation caused. Depth is measured from the paragraph,
        # so a point under a subparagraph is not double-indented.
        visual = 0 if kind in _PARA_LEVEL else max(depth - baseline, 1)
        indent = _INDENT * visual
        if not first:
            lines.append(">")                          # blank quote line between every unit
        lines.append(f"> {indent}{body}")
        first = False
    return "\n".join(lines)


def _pick_subject(provisions: list[dict], ref: str, celex: str) -> dict | None:
    want = ref.strip().lower()
    for p in provisions:
        pref = (p.get("article_ref") or p.get("display_ref") or "").strip().lower()
        if pref == want and p.get("regulation_id") == celex and p.get("subtree"):
            return p
    return next((p for p in provisions if p.get("subtree")), None)


def _celex_display(celex: str) -> str:
    """"32026R1744" → "Regulation (EU) 2026/1744 (Digital Omnibus on AI)"."""
    meta = LEGISLATION.get(celex) or {}
    number, name = meta.get("number"), meta.get("name")
    if number and name:
        return f"Regulation (EU) {number} ({name})"
    return f"Regulation (EU) {number}" if number else celex


def _amending_acts(subtree: list[dict]) -> list[str]:
    """The amending acts (friendly names) that touched this subtree, in the order
    first seen — from the ``amended_by`` tag consolidation left on the nodes."""
    seen: list[str] = []
    for nd in subtree:
        celex = nd.get("amended_by")
        if celex:
            label = _celex_display(celex)
            if label not in seen:
                seen.append(label)
    return seen


def render_provision_display(retriever, ref: str, celex: str) -> str | None:
    """Verbatim render of *ref* in *celex* from the consolidated graph. Returns
    markdown, or ``None`` when the provision can't be resolved (the caller falls
    back to the generative path)."""
    try:
        provisions = retriever.retrieve_by_refs([ref], {celex})
    except Exception:
        return None
    subject = _pick_subject(provisions, ref, celex)
    if not subject or not subject.get("subtree"):
        return None
    subtree = subject["subtree"]
    base_ref = (subtree[0].get("ref") or ref).strip()
    heading = re.sub(r"\s+", " ", (subtree[0].get("text") or "")).strip()

    body = _render_nodes(subtree)
    if not body:
        return None

    reg_name = (LEGISLATION.get(celex, {}) or {}).get("name", celex)
    acts = _amending_acts(subtree)
    title = f"### {base_ref} — {heading}" if heading else f"### {base_ref}"
    if acts:
        subtitle = (f"*{reg_name}, shown as currently in force — consolidating amendments by "
                    f"{', '.join(acts)}. Verbatim source text; not legal advice.*")
        footer = ("\n\n---\n**As amended.** The text above reflects amendments by "
                  f"{', '.join(f'**{a}**' for a in acts)}, applied in place.")
    else:
        subtitle = f"*{reg_name} — verbatim source text; not legal advice.*"
        footer = ""

    return f"{title}\n{subtitle}\n\n{body}{footer}"
