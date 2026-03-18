"""Text enrichment for multi-granularity embeddings.

Populates ``text_for_analysis`` on every provision by:

1. **Flattening** — parent nodes whose ``text`` is empty or title-only
   receive the concatenated text of all their descendants (bottom-up).
2. **Context prefixing** — each node's ``text_for_analysis`` is prefixed
   with its structural ancestry (chapter title → section title → article
   title) so that embeddings capture positional semantics.

The resulting ``text_for_analysis`` field is the recommended input for
embedding models (e.g. multilingual-e5-large).

Public API
----------
- :func:`enrich_text_for_analysis` — in-place enrichment of a provisions list.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Kinds that carry structural headings (used for context prefix).
_HEADING_KINDS = frozenset({
    "chapter", "section", "article",
    "annex", "annex_chapter", "annex_part", "annex_section",
})

# Kinds whose text is typically just a heading / title, not normative body.
_TITLE_ONLY_KINDS = frozenset({
    "document", "preamble", "enacting_terms", "final_provisions", "annexes",
    "chapter", "section",
    "annex", "annex_chapter", "annex_part",
})

# Kinds where we do NOT produce a text_for_analysis
# (structural containers with no independent semantic value).
_SKIP_KINDS = frozenset({
    "document", "preamble", "enacting_terms", "final_provisions", "annexes",
})

# Separator between context prefix and body text.
_CTX_SEP = " | "
# Separator between ancestor levels in the prefix.
_PATH_SEP = " > "
# Separator when concatenating children's text.
_CHILD_SEP = " "


def enrich_text_for_analysis(provisions: List[Dict[str, Any]]) -> int:
    """Populate ``text_for_analysis`` on every provision (in-place).

    Parameters
    ----------
    provisions:
        The full provisions list from ``parsed.json``.

    Returns
    -------
    int
        Number of provisions enriched (received a non-empty
        ``text_for_analysis``).
    """
    by_id: Dict[str, Dict[str, Any]] = {p["id"]: p for p in provisions}

    # ------------------------------------------------------------------
    # Phase 1 — Flatten text bottom-up.
    #
    # For nodes whose ``text`` is empty or matches their ``title``
    # (= heading-only), we collect descendant leaf text instead.
    # ------------------------------------------------------------------
    flattened: Dict[str, str] = {}
    _flatten_all(provisions, by_id, flattened)

    # ------------------------------------------------------------------
    # Phase 2 — Build context prefix from ancestry and assemble
    # ``text_for_analysis`` = prefix + body.
    # ------------------------------------------------------------------
    enriched = 0
    for prov in provisions:
        kind = prov.get("kind", "")
        if kind in _SKIP_KINDS:
            prov["text_for_analysis"] = None
            continue

        body = flattened.get(prov["id"], "")
        if not body.strip():
            prov["text_for_analysis"] = None
            continue

        prefix = _build_context_prefix(prov, by_id)
        if prefix:
            prov["text_for_analysis"] = prefix + _CTX_SEP + body
        else:
            prov["text_for_analysis"] = body

        enriched += 1

    logger.info(
        "text_for_analysis: enriched %d / %d provisions.", enriched, len(provisions),
    )
    return enriched


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _flatten_all(
    provisions: List[Dict[str, Any]],
    by_id: Dict[str, Dict[str, Any]],
    out: Dict[str, str],
) -> None:
    """Compute flattened body text for every provision via post-order DFS."""
    # We process provisions from leaves upward.  The ``children`` list
    # gives the tree structure.

    # Track visited to avoid re-computation.
    visited: set = set()

    def _flatten(pid: str) -> str:
        if pid in visited:
            return out.get(pid, "")
        visited.add(pid)

        prov = by_id.get(pid)
        if prov is None:
            return ""

        children_ids: List[str] = prov.get("children", [])
        own_text = (prov.get("text") or "").strip()
        kind = prov.get("kind", "")
        title = (prov.get("title") or "").strip()

        # Determine if own_text is meaningful body vs. heading-only.
        has_body = bool(own_text) and own_text != title

        if not children_ids:
            # Leaf node — use own text directly.
            out[pid] = own_text
            return own_text

        # Non-leaf — recurse into children.
        child_texts = [_flatten(cid) for cid in children_ids]
        joined_children = _CHILD_SEP.join(t for t in child_texts if t)

        if has_body:
            # Node has both its own body text and children.
            # Prepend own text (e.g. introductory paragraph stem).
            flattened = own_text + _CHILD_SEP + joined_children if joined_children else own_text
        else:
            # Node text is empty or heading-only; body comes from children.
            flattened = joined_children

        out[pid] = flattened
        return flattened

    for prov in provisions:
        _flatten(prov["id"])


def _build_context_prefix(
    prov: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
) -> str:
    """Build a human-readable ancestry prefix from the provision's path.

    Example output::

        Chapter III — Requirements for High-Risk AI Systems >
        Section 2 — Requirements for high-risk AI systems >
        Article 11 — Technical documentation

    Only ancestors with heading-like kinds contribute segments.
    The provision itself is included if it is a heading kind.
    """
    path_ids: List[str] = prov.get("path", []) or []
    segments: List[str] = []

    for anc_id in path_ids:
        anc = by_id.get(anc_id)
        if anc is None:
            continue
        seg = _heading_segment(anc)
        if seg:
            segments.append(seg)

    # Include self if it's a heading kind (e.g. article, annex_section).
    own_seg = _heading_segment(prov)
    if own_seg:
        segments.append(own_seg)

    return _PATH_SEP.join(segments)


def _heading_segment(prov: Dict[str, Any]) -> Optional[str]:
    """Return a concise heading string for a provision, or None."""
    kind = prov.get("kind", "")
    if kind not in _HEADING_KINDS:
        return None

    number = prov.get("number", "")
    title = (prov.get("title") or "").strip()

    label = _LABEL_MAP.get(kind, kind.replace("_", " ").title())

    if number and title:
        return f"{label} {number} \u2014 {title}"
    if number:
        return f"{label} {number}"
    if title:
        return f"{label} \u2014 {title}"
    return None


_LABEL_MAP: Dict[str, str] = {
    "chapter": "Chapter",
    "section": "Section",
    "article": "Article",
    "annex": "Annex",
    "annex_chapter": "Annex Chapter",
    "annex_part": "Annex Part",
    "annex_section": "Annex Section",
}
