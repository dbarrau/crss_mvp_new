"""Context assembly and formatting for the LLM prompt.

Converts retrieved provisions and definitions into structured text blocks,
extracts inline cross-references found inside provision bodies, and formats
definition lookup results.  No LLM calls; no retriever I/O.
"""
from __future__ import annotations

import os
import re

from application._config import _BODY_LIMIT, _INLINE_REF_RE


def _natural_key(s: str) -> list:
    """Sort key that orders node ids by document position, numeric-aware.

    Splits into digit / non-digit runs so ``…_anx_III_2`` sorts before
    ``…_anx_III_10`` and ``1_a`` before ``1_b`` — keeping points and their
    sub-items in reading order (the retrieval ``collect()`` loses that order).
    """
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s or "")]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_POINTER_REFS = 10

def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


# Per-provision rendering budgets — plain layout constants, deliberately *not*
# env-tunable.  They bound how much child-paragraph and cross-reference detail
# each provision contributes to the LLM prompt; the rolled-up body (capped at
# _BODY_LIMIT) already carries the substance, so a long tail of children /
# citations mostly inflated the prompt (a ~27-child article → ~25 KB; a full
# 40-provision context reached ~313 KB / ~78 K tokens, slowing generation).
# (D1: these were one env knob each — CRSS_CHILD_CHARS, CRSS_CITE_CHARS, … — that
# no deployment ever set; the single operational lever is the global
# CRSS_CONTEXT_CHAR_BUDGET below, which subsumes them by bounding the total.)
_MAX_CHILD_LINES = 12
_CHILD_CHARS = 700
_CHILD_MATCHED_CHARS = 1000
_MAX_CITE_LINES = 6
_CITE_CHARS = 700
# Interpretive links (guidance↔legislation INTERPRETS edges).  Kept tight so
# surfacing MDCG interpretation alongside binding text does not re-inflate the
# prompt (the bloat lever the context budget guards).
_MAX_INTERP_LINES = 3
_INTERP_CHARS = 600

# Global cap on the rendered provision context (chars).  The LLM must prefill
# every input token before it emits the first output token, so an oversized
# context directly inflates (and destabilises) time-to-first-token on the large
# generation model.  Broad routes (e.g. cross_regulation) can retrieve 40+
# provisions → ~220 KB / ~56 K tokens; we keep the highest-priority provisions
# (the retriever orders direct / role / reranked hits first and appends low-value
# cross-reference + pointer expansions last) and drop the tail once the budget is
# hit.  ~140 KB ≈ ~35 K tokens preserves the decisive backbone while roughly
# halving prefill latency.  This is the one context knob a deployment may need to
# tune, so it stays env-overridable via CRSS_CONTEXT_CHAR_BUDGET.
_CONTEXT_CHAR_BUDGET = _int_env("CRSS_CONTEXT_CHAR_BUDGET", 140_000)

# Only trim when the provision bag is large enough that bloat is a real risk
# (plain constant, not env-tunable).  Single-regulation queries retrieve at most
# k=20 provisions; even at max per-provision size (~17 KB) that stays around
# 340 KB — the trim was designed for the 40+ provision cross-regulation case
# (40 × 17 KB = 680 KB, compacted to 140 KB ≈ 35 K tokens).  At or below the
# threshold every provision is preserved so the LLM gets complete coverage
# (important for "what is X about" overview questions where every article matters).
_TRIM_PROVISION_THRESHOLD = 25

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


def _collect_cites_targets(
    provisions: list[dict],
    already_ids: set[str],
    celex_filter: set[str] | None = None,
) -> list[str]:
    """Collect node ids of CITES-edge targets not yet in the retrieved bag.

    The retriever resolves each provision's CITES edges into
    ``cited_provisions`` (with the cross-regulation subset also in
    ``cross_reg_cited``), each carrying the target's stable, unique node
    ``id``.  Those targets render only as short snippets *without* an ``id:``
    line, so the model cannot cite them under the grounded-citation contract —
    and, needing to name a cross-referenced article (e.g. AI Act Art 43), it
    may fabricate a node id instead.  Returning their ids lets the pointer-
    expansion stage promote them to first-class *citable* provisions.

    Cross-regulation targets are ordered first (they are the more decisive
    links, mirroring ``_format_one_provision``'s cross-ref ordering), then
    internal cites.  De-duplicated, preserving that priority order, and
    excluding anything already in ``already_ids``.

    When ``celex_filter`` is set (a scope-restricted question), targets in
    other regulations are dropped — mirroring the reverse cross-reg expansion,
    which is likewise skipped under scope — so out-of-scope regs cannot starve
    the in-scope cross-references out of the bounded promotion budget.
    """
    xreg: dict[str, None] = {}       # ordered set: cross-regulation targets
    internal: dict[str, None] = {}   # ordered set: same-regulation targets
    for p in provisions:
        cross_ids = {c.get("id") for c in (p.get("cross_reg_cited") or [])}
        for c in p.get("cited_provisions") or []:
            cid = c.get("id")
            if not cid or cid in already_ids:
                continue
            if celex_filter and not any(cid.startswith(cx) for cx in celex_filter):
                continue
            (xreg if cid in cross_ids else internal).setdefault(cid, None)
    return list(xreg) + [cid for cid in internal if cid not in xreg]


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


def _trim_provisions_to_budget(
    provisions: list[dict], budget: int = _CONTEXT_CHAR_BUDGET
) -> list[dict]:
    """Drop the lowest-priority tail so the rendered context fits ``budget`` chars.

    Provisions arrive in priority order (direct lookups, role hits and reranked
    vector hits first; cross-reference / pointer expansions appended last), so we
    keep the leading provisions and stop once adding the next would exceed the
    budget.  At least one provision is always kept.  Returns the input unchanged
    when it already fits, so narrow-scope queries are unaffected.

    Skipped entirely for small bags (``len <= _TRIM_PROVISION_THRESHOLD``): only
    broad cross-regulation routes accumulate enough provisions to need trimming,
    and trimming a narrow query would silently drop relevant articles from an
    overview answer.
    """
    if not provisions or len(provisions) <= _TRIM_PROVISION_THRESHOLD:
        return provisions
    kept: list[dict] = []
    total = 0
    for i, p in enumerate(provisions, 1):
        role = (p.get("provision_role") or "").strip().upper()
        size = len(_format_one_provision(i, p, role))
        if kept and total + size > budget:
            break
        kept.append(p)
        total += size
    return kept


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


def _cross_ref_and_interp_lines(p: dict) -> tuple[list[str], list[str]]:
    """Return the capped cross-reference and interpretive-link lines for a block.

    Shared by the compact and the full-subtree provision renderers so both
    surface the same downstream context (e.g. the Annex a point cites, the MDCG
    guidance that interprets an article) without duplicating the formatting.

    Cross-regulation citations are more decisive than internal ones, so they are
    kept first and the combined list is capped.  Interpretive links follow the
    guidance<->legislation INTERPRETS edges in both directions.
    """
    cited = p.get("cited_provisions") or []
    cross_reg_ids = {c.get("id") for c in (p.get("cross_reg_cited") or [])}
    _xreg_cites: list[str] = []
    _internal_cites: list[str] = []
    for c in cited:
        ref = c.get("ref", "")
        is_xreg = c.get("id") in cross_reg_ids
        text = (c.get("text") or "")[:_CITE_CHARS]
        if text:
            tag = " [CROSS-REG]" if is_xreg else ""
            line = f"  -> {ref}{tag}: {text}"
            (_xreg_cites if is_xreg else _internal_cites).append(line)
    cite_lines = (_xreg_cites + _internal_cites)[:_MAX_CITE_LINES]

    interp_lines: list[str] = []
    for g in (p.get("interpreting_guidance") or [])[:_MAX_INTERP_LINES]:
        ref = g.get("ref", "")
        text = (g.get("text") or "")[:_INTERP_CHARS]
        if text:
            interp_lines.append(f"  [GUIDANCE interprets this] {ref}: {text}")
    for ip in (p.get("interpreted_provisions") or [])[:_MAX_INTERP_LINES]:
        ref = ip.get("ref", "")
        text = (ip.get("text") or "")[:_INTERP_CHARS]
        if text:
            interp_lines.append(f"  [INTERPRETS legislation] {ref}: {text}")
    return cite_lines, interp_lines


def _render_subtree(subtree: list[dict]) -> str:
    """Render an ordered HAS_PART subtree as faithfully-numbered, nested text.

    ``subtree`` is the retriever's document-ordered (pre-order) list of nodes,
    each ``{ref, number, kind, text, depth}`` with the looked-up provision at
    ``depth == 0``.  Every line carries the node's OWN text (never the flattened
    ``text_for_analysis``) and its enumerator, indented by depth, so a point's
    roman sub-items nest visibly beneath their chapeau and each unit reads as the
    exact, referenceable text of the regulation:

        1.   Providers of general-purpose AI models shall:
          (b) draw up, keep up-to-date and make available … shall:
              (i) enable providers of AI systems to have a good understanding …;
              (ii) contain, at a minimum, the elements set out in Annex XII;
    """
    lines: list[str] = []
    for node in subtree:
        depth = node.get("depth") or 0
        text = (node.get("text") or "").replace("\xa0", " ").strip()
        if not text:
            continue
        indent = "  " * depth
        # The root paragraph/article already carries its own number ("1.  …");
        # descendant points/items store the bare body, so prefix the enumerator
        # the way the source cites it: "(a)" / "(i)" for lettered/roman points,
        # an em-dash for unnumbered "indent" bullets (whose number is only an
        # ordinal, not part of the text).
        number = (node.get("number") or "").strip()
        if depth == 0:
            label = ""
        elif node.get("kind") == "indent":
            label = "— "
        elif number:
            label = f"({number}) "
        else:
            label = ""
        lines.append(f"{indent}{label}{text}")
    return "\n".join(lines)


def _format_subject_provision(header: str, subtree: list[dict], p: dict) -> str:
    """Render the direct-lookup subject as its faithful, ordered subtree.

    Replaces the flattened-body + separate child-list layout for the one
    provision the question actually names.  Cross-references and interpretive
    links still follow (via :func:`_cross_ref_and_interp_lines`), but the body is
    the real document structure.  Not subject to ``_PROVISION_BLOCK_CAP`` —
    rendering the named provision completely is the whole point of the lookup,
    exactly as the previous ``full_children`` path was exempt.
    """
    section = header + "\n" + _render_subtree(subtree)
    cite_lines, interp_lines = _cross_ref_and_interp_lines(p)
    if cite_lines:
        section += "\n\nCross-references:\n" + "\n".join(cite_lines)
    if interp_lines:
        section += "\n\nInterpretive links:\n" + "\n".join(interp_lines)
    return section


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
    elif _force == "interpretive":
        # Recitals / citations / preamble: aid interpretation of the enacting
        # terms but impose no obligations — must not be cited as the legal basis.
        force_badge = " [INTERPRETIVE — NOT OPERATIVE]"
    else:
        force_badge = ""
    header = (
        f"[{index}] {p.get('article_ref', 'Unknown')} \u2014 {regulation}"
        f"{celex_badge}{layer_tag}{force_badge}{role_badge}"
    )
    # Stable node id the model points at under the grounded-citation contract
    # (see docs/grounded_generation_contract.md). This is the *only* citable key \u2014
    # never display_ref, which is None/non-unique on many nodes.
    article_id = p.get("article_id", "")
    if article_id:
        header += f"\n    id: {article_id}"
    path = p.get("article_path", "")
    if path:
        header += f"\n    Path: {path}"

    # When this provision is the DIRECT SUBJECT of a structural lookup and its
    # ordered HAS_PART subtree was attached, render that subtree exactly as
    # numbered in the source document (chapeau -> points -> roman sub-items),
    # from each node's own text. This faithful, referenceable view replaces both
    # the flattened embedding body and the separate child list — which duplicated
    # the body and leaked whole re-flattened ancestor articles/sections into it.
    if p.get("_direct_ref_match") and not p.get("_pointer_expansion"):
        subtree = p.get("subtree")
        if subtree:
            return _format_subject_provision(header, subtree, p)

    body = p.get("article_text", "") or ""
    if len(body) > _BODY_LIMIT:
        body = body[:_BODY_LIMIT] + " [\u2026see paragraph details below\u2026]"

    # Child provisions — use raw provision text so paragraph numbers are
    # unambiguous (e.g. "4.   'active device' means..."), without the
    # repeated ancestry prefix that obscures the numbering.
    # Render children in DOCUMENT ORDER: the retrieval Cypher's collect() does not
    # preserve order, so a large structured provision (e.g. Annex III's 8 points ×
    # lettered sub-items) would otherwise arrive scrambled and be sampled wrong.
    # The child id encodes ordinal + parentage (…_anx_III_1_a), so a natural sort
    # restores "point 1, 1(a), 1(b), 1(c), point 2, …".
    children = sorted(
        p.get("children") or [], key=lambda c: _natural_key(c.get("id") or "")
    )
    matched_leaf = p.get("matched_leaf_id")
    # When this provision is the DIRECT SUBJECT of a structural lookup, render
    # every child so a "what does X list?" question is answered completely.
    # Peripheral (cross-referenced / pointer-expanded) provisions stay capped so
    # the prompt does not re-inflate — the reason for _MAX_CHILD_LINES.
    full_children = bool(
        p.get("_direct_ref_match") and not p.get("_pointer_expansion")
    )
    rendered: list[str] = []       # every child line, in document order
    matched_lines: list[str] = []  # matched leaf(s), for the capped path
    for c in children:
        ref = c.get("ref") or c.get("kind", "")
        cid = c.get("id") or ""
        id_tag = f" (id: {cid})" if cid else ""
        is_match = bool(matched_leaf and c.get("id") == matched_leaf)
        limit = _CHILD_MATCHED_CHARS if is_match else _CHILD_CHARS
        text = (c.get("raw_text") or c.get("text") or "")
        if len(text) > limit:
            cut = text[:limit]
            last_period = cut.rfind('.')
            if last_period > limit // 2:
                text = cut[:last_period + 1]
            else:
                text = cut
        if not text:
            continue
        line = f"  {'[\u2605 MATCHED] ' if is_match else ''}{ref}{id_tag}: {text}"
        rendered.append(line)
        if is_match:
            matched_lines.append(line)
    if full_children:
        child_lines = rendered                       # complete, in document order
    else:
        # Keep the matched leaf plus the leading children; the rolled-up body
        # already summarises the provision, so a long tail is largely redundant
        # and was the dominant source of context bloat. See _MAX_CHILD_LINES.
        seen = set(matched_lines)
        tail = [ln for ln in rendered if ln not in seen]
        child_lines = (matched_lines + tail)[:_MAX_CHILD_LINES]

    cite_lines, interp_lines = _cross_ref_and_interp_lines(p)

    section = header + "\n" + body
    if p.get("_cross_reg_expansion"):
        section = f"[{section.lstrip('[')}  [via cross-regulation link]"
    if p.get("_pointer_expansion"):
        section = f"[{section.lstrip('[')}  [referenced in retrieved provisions]"
    if child_lines:
        section += "\n\nParagraphs/Points:\n" + "\n".join(child_lines)
    if cite_lines:
        section += "\n\nCross-references:\n" + "\n".join(cite_lines)
    if interp_lines:
        section += "\n\nInterpretive links:\n" + "\n".join(interp_lines)
    return section
