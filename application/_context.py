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

# Provision-role bucket ordering and human-readable section labels.
# The order below is semantic, not alphabetical: definitions first (so the LLM
# anchors actor/object identity), then scope/classification (when each thing
# applies), then status modifiers (EXTENDS_STATUS, EXEMPTS), then duties
# (OBLIGATION, PROHIBITION), then procedures/penalties, then interpretive
# fillers. Any provision lacking a recognised role lands in OTHER.
#
# These labels are presentation-only; they do not introduce new taxonomy.
_ROLE_BUCKET_ORDER: tuple[tuple[str, str], ...] = (
    ("DEFINES", "DEFINITIONS"),
    ("SCOPE", "SCOPE"),
    ("CLASSIFICATION", "CLASSIFICATION"),
    ("EXTENDS_STATUS", "STATUS EXTENSIONS"),
    ("EXEMPTS", "EXEMPTIONS / MODIFIERS"),
    ("OBLIGATION", "OBLIGATIONS"),
    ("PROHIBITION", "PROHIBITIONS"),
    ("PROCEDURAL", "PROCEDURES"),
    ("PENALTY", "PENALTIES"),
    ("INTERPRETIVE", "INTERPRETIVE (recitals / explanatory)"),
)
_KNOWN_ROLES: frozenset[str] = frozenset(role for role, _ in _ROLE_BUCKET_ORDER)
_OTHER_BUCKET_LABEL = "OTHER"

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


def _community_summary_header(provisions: list[dict]) -> str:
    """Build a de-duplicated community-summary preamble for community-route context.

    Returns an empty string when no provisions carry a ``community_summary``
    field (e.g. older data or HyDE fallback), so the caller can skip it safely.
    """
    seen: dict[str, str] = {}  # community_id -> summary_text (ordered, deduped)
    for p in provisions:
        cid = p.get("community_id") or ""
        summary = p.get("community_summary") or ""
        if cid and summary and cid not in seen:
            seen[cid] = summary

    if not seen:
        return ""

    lines: list[str] = ["[Community Overview]"]
    for cid, summary in seen.items():
        short_id = cid.replace("community::", "").lstrip("0") or "0"
        lines.append(f"  Community {short_id}: {summary}")
    lines.append("─" * 60)
    return "\n".join(lines)


def _format_context(provisions: list[dict]) -> str:
    """Turn retriever results into a structured text block for the LLM.

    Provisions are grouped into ordered semantic buckets by ``provision_role``
    (e.g. DEFINITIONS, EXEMPTIONS, OBLIGATIONS). Bucket headers separate the
    sections; per-provision numbering is contiguous across buckets so that
    answer-side citations like ``[3]`` remain stable. Provisions without a
    recognised role fall into the OTHER bucket at the end.
    """
    # 1) Render each provision to its own block (header + body + paragraphs +
    #    cross-references) with a contiguous citation index. We also stamp the
    #    bucket label on each block so we can re-group while preserving order.
    blocks: list[tuple[str, str]] = []  # (bucket_label, rendered_block)
    for i, p in enumerate(provisions, 1):
        role = (p.get("provision_role") or "").strip().upper()
        bucket_label = next(
            (label for r, label in _ROLE_BUCKET_ORDER if r == role),
            _OTHER_BUCKET_LABEL,
        )
        blocks.append((bucket_label, _format_one_provision(i, p, role)))

    # 2) Group blocks by bucket while preserving the canonical bucket order
    #    and the original within-bucket ordering (which is the retriever's
    #    ranking).
    bucket_order_index = {
        label: idx for idx, (_role, label) in enumerate(_ROLE_BUCKET_ORDER)
    }
    bucket_order_index[_OTHER_BUCKET_LABEL] = len(_ROLE_BUCKET_ORDER)

    grouped: dict[str, list[str]] = {}
    for label, block in blocks:
        grouped.setdefault(label, []).append(block)

    # If every provision falls into a single bucket, skip the section headers
    # entirely — keeps the output identical to the legacy format for the
    # narrow-scope queries where bucketing adds no information.
    if len(grouped) <= 1:
        flat = [block for _label, block in blocks]
        return "\n\n---\n\n".join(flat)

    parts: list[str] = []
    for label in sorted(grouped, key=lambda lab: bucket_order_index.get(lab, 999)):
        items = grouped[label]
        section_header = f"### {label} ###"
        parts.append(section_header + "\n\n" + "\n\n---\n\n".join(items))
    return "\n\n".join(parts)


def _format_one_provision(index: int, p: dict, role: str) -> str:
    """Render a single retrieved provision block.

    Extracted from ``_format_context`` so the bucket-grouping wrapper can
    iterate without duplicating the per-provision rendering logic.
    """
    regulation = p.get("regulation", "")
    celex = p.get("celex", "")
    is_guidance = any(celex.startswith(pfx) for pfx in _GUIDANCE_CELEX_PREFIXES)
    layer_tag = " [GUIDANCE]" if is_guidance else " [LEGISLATION]"
    celex_badge = f" (CELEX: {celex})" if celex and not is_guidance else ""
    role_badge = f" [role: {role}]" if role in _KNOWN_ROLES else ""
    _force = p.get("binding_force")
    if _force == "binding":
        force_badge = " [BINDING]"
    elif _force == "non_binding":
        force_badge = " [NON-BINDING GUIDANCE]"
    else:
        force_badge = ""
    header = (
        f"[{index}] {p.get('article_ref', 'Unknown')} \u2014 {regulation}"
        f"{celex_badge}{layer_tag}{force_badge}{role_badge}"
    )
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
    return section
