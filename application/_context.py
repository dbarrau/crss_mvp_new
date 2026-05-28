"""Context assembly and formatting for the LLM prompt.

Converts retrieved provisions and definitions into structured text blocks,
extracts inline cross-references found inside provision bodies, and formats
definition lookup results.  No LLM calls; no retriever I/O.
"""
from __future__ import annotations

from application._config import _BODY_LIMIT, _INLINE_REF_RE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_POINTER_REFS = 10

# CELEX prefixes that identify MDCG guidance documents.
_GUIDANCE_CELEX_PREFIXES = ("MDCG_",)

# ---------------------------------------------------------------------------
# Inline reference extraction
# ---------------------------------------------------------------------------


def _normalize_ref(raw: str) -> str:
    """Normalise a raw inline reference match to canonical form."""
    ref = raw.strip().replace("\xa0", " ")
    if ref.lower().startswith("annexes"):
        ref = "Annex" + ref[7:]
    return ref


def _extract_inline_refs(provisions: list[dict]) -> list[str]:
    """Scan retrieved provisions for inline references to other provisions.

    Returns normalised references (e.g. ``['Annex VII', 'Article 17']``)
    that are mentioned in the body or children of the already-retrieved
    provisions but are not themselves among those provisions.

    Capped at ``_MAX_POINTER_REFS`` to prevent context flooding.
    """
    already = {p.get("article_ref", "") for p in provisions}
    found: dict[str, None] = {}  # preserves insertion order, deduplicates

    for p in provisions:
        # Scan the parent body text
        body = (p.get("article_text", "") or "").replace("\xa0", " ")
        for m in _INLINE_REF_RE.finditer(body):
            ref = _normalize_ref(m.group(0))
            if ref not in already and ref not in found:
                found[ref] = None
                if len(found) >= _MAX_POINTER_REFS:
                    return list(found)

        # Scan children text (paragraphs / points)
        for c in p.get("children") or []:
            text = (c.get("raw_text") or c.get("text") or "").replace("\xa0", " ")
            for m in _INLINE_REF_RE.finditer(text):
                ref = _normalize_ref(m.group(0))
                if ref not in already and ref not in found:
                    found[ref] = None
                    if len(found) >= _MAX_POINTER_REFS:
                        return list(found)

    return list(found)


# ---------------------------------------------------------------------------
# Definition formatting
# ---------------------------------------------------------------------------


def _format_definitions(definitions: list[dict]) -> str:
    """Format definition lookup results as a context block for the LLM."""
    parts: list[str] = []
    for d in definitions:
        reg = d.get("regulation", "")
        ref = d.get("article_ref", "")
        term = d.get("term", "")
        text = d.get("definition_text", "")
        dtype = d.get("definition_type", "formal")
        if dtype == "contextual":
            label = f"Scoped definition of \u2018{term}\u2019"
        else:
            label = f"Formal definition of \u2018{term}\u2019"
        if ref:
            label += f" \u2014 {ref}"
        if reg:
            label += f" ({reg})"
        if dtype == "contextual":
            label += " [scoped to this article]"
        parts.append(f"{label}:\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Provision context formatting
# ---------------------------------------------------------------------------


def _format_context(provisions: list[dict]) -> str:
    """Turn retriever results into a structured text block for the LLM."""
    parts: list[str] = []
    for i, p in enumerate(provisions, 1):
        regulation = p.get("regulation", "")
        celex = p.get("celex", "")
        is_guidance = any(celex.startswith(pfx) for pfx in _GUIDANCE_CELEX_PREFIXES)
        layer_tag = " [GUIDANCE]" if is_guidance else " [LEGISLATION]"
        header = f"[{i}] {p.get('article_ref', 'Unknown')} ({regulation}){layer_tag}"
        path = p.get("article_path", "")
        if path:
            header += f"\n    Path: {path}"

        body = p.get("article_text", "") or ""
        if len(body) > _BODY_LIMIT:
            body = body[:_BODY_LIMIT] + " [\u2026see paragraph details below\u2026]"

        # Child provisions — use raw provision text so paragraph numbers are
        # unambiguous (e.g. "4.   'active device' means..."), without the
        # repeated ancestry prefix that obscures the numbering.
        children = p.get("children") or []
        matched_leaf = p.get("matched_leaf_id")
        child_lines: list[str] = []
        matched_lines: list[str] = []
        for c in children:
            ref = c.get("ref") or c.get("kind", "")
            is_match = bool(matched_leaf and c.get("id") == matched_leaf)
            limit = 1200 if is_match else 1000
            text = (c.get("raw_text") or c.get("text") or "")
            if len(text) > limit:
                cut = text[:limit]
                last_period = cut.rfind('.')
                if last_period > limit // 2:
                    text = cut[:last_period + 1]
                else:
                    text = cut
            if text:
                if is_match:
                    matched_lines.append(f"  [\u2605 MATCHED] {ref}: {text}")
                else:
                    child_lines.append(f"  {ref}: {text}")
        child_lines = (matched_lines + child_lines)[:40]

        # Cross-referenced provisions (separate internal vs cross-regulation)
        cited = p.get("cited_provisions") or []
        cross_reg = p.get("cross_reg_cited") or []
        cross_reg_ids = {c.get("id") for c in cross_reg}
        cite_lines: list[str] = []
        for c in cited:
            ref = c.get("ref", "")
            is_xreg = c.get("id") in cross_reg_ids
            # Give more text budget to cross-regulation citations;
            # internal citations also need room for substantive annex content.
            limit = 1000 if is_xreg else 1000
            text = (c.get("text") or "")[:limit]
            if text:
                tag = " [CROSS-REG]" if is_xreg else ""
                cite_lines.append(f"  -> {ref}{tag}: {text}")

        section = header + "\n" + body
        if p.get("_cross_reg_expansion"):
            section = f"[{section.lstrip('[')}  [via cross-regulation link]"
        if p.get("_pointer_expansion"):
            section = f"[{section.lstrip('[')}  [referenced in retrieved provisions]"
        if child_lines:
            section += "\n\nParagraphs/Points:\n" + "\n".join(child_lines)
        if cite_lines:
            section += "\n\nCross-references:\n" + "\n".join(cite_lines)
        parts.append(section)

    return "\n\n---\n\n".join(parts)
