"""MDCG clean-markdown → parsed.json structurer.

Transforms the post-processed MDCG markdown (produced by
:mod:`ingestion.parse.guidance.mdcg_parser`) into the same
``parsed.json`` schema consumed by :class:`RegulationGraphLoader`.

Public API
----------
- :func:`structure_mdcg` — parse markdown into a provision tree.

The output is a dict with keys ``graph_version``, ``celex_id``,
``regulation_id``, ``provisions``, ``relations``, and
``defined_terms`` — identical in schema to the regulation output
so the Neo4j loader works without modification.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── heading regex ──────────────────────────────────────────────────────────
# Matches markdown headings: "## 4.3.1 Title text" or "# Annex" etc.
# Groups:  level (number of #), optional section_num, title text
_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})\s+"           # e.g. "###"
    r"(?:"
    r"(?P<sec_num>\d+(?:\.\d+)*)"       # e.g. "4.3.1"
    r"\.?\s+"                            # optional trailing dot + space
    r")?"
    r"(?P<title>.+)",                    # rest of line is the title
)

# Charts appear in the Annex as "## Chart A", "## Chart B", etc.
_CHART_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,3})\s+"
    r"(?:Design changes.*–\s*)?"         # long main-chart title
    r"(?:Chart\s+)?(?P<chart_id>[A-Z]|Main\s+Chart)"
    r"(?:\s*[-–:].*)?$",
    re.IGNORECASE,
)

# "## Footnotes" section — we skip it (metadata, not a provision)
_FOOTNOTES_RE = re.compile(r"^#{1,3}\s+Footnotes\s*$", re.IGNORECASE)

# "# Contents" or table-of-contents heading — skip
_CONTENTS_RE = re.compile(r"^#{1,3}\s+Contents\s*$", re.IGNORECASE)

# Preamble headings that duplicate the title / are revision notes
_SKIP_HEADING_RE = re.compile(
    r"^#\s+MDCG\s+\d{4}[–-]\d+|^#\s+Guidance\s+on\s+",
    re.IGNORECASE,
)

# Strip HTML superscript tags from heading text for cleaner titles
_SUP_RE = re.compile(r"<sup>.*?</sup>", re.IGNORECASE)

# Strip markdown link wrappers: [text](url) → text
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Detect TOC entries: heading titles ending with a standalone page number
# e.g. "Annex II - Qualification examples ... 31"  ← "31" is a page number
_TOC_PAGE_RE = re.compile(r"\s+\d{1,3}\s*$")


def _clean_heading(title: str) -> str:
    """Remove footnote markers and URLs from heading text."""
    title = _SUP_RE.sub("", title)
    title = _LINK_RE.sub(r"\1", title)
    return title.strip()


# ── public API ─────────────────────────────────────────────────────────────

def structure_mdcg(
    md_path: str | Path,
    doc_id: str,
    doc_name: str,
    lang: str = "EN",
) -> dict[str, Any]:
    """Parse MDCG clean markdown into the ``parsed.json`` schema.

    Parameters
    ----------
    md_path:
        Path to the ``*_clean.md`` file.
    doc_id:
        Document identifier used as the ``celex_id`` equivalent
        (e.g. ``"MDCG_2020_3"``).
    doc_name:
        Human-readable name (e.g. ``"MDCG 2020-3 Rev.1"``).
    lang:
        Language code.

    Returns
    -------
    dict
        A ``parsed.json``-compatible dict ready for
        :meth:`RegulationGraphLoader.load_file`.
    """
    md_text = Path(md_path).read_text(encoding="utf-8")
    provisions = _build_provision_tree(md_text, doc_id, doc_name, lang)

    # Run text enrichment (populates text_for_analysis)
    try:
        from canonicalization.text_enrichment import enrich_text_for_analysis
        n = enrich_text_for_analysis(provisions)
        logger.info("Text enrichment: %d provisions enriched.", n)
    except Exception as exc:
        logger.warning("Text enrichment skipped: %s", exc)

    # Extract regulation cross-references (CITES_EXTERNAL)
    try:
        from ingestion.parse.semantic_layer.guidance_references import (
            extract_guidance_relations,
        )
        relations = extract_guidance_relations(provisions)
        logger.info("Guidance cross-references: %d relations extracted.", len(relations))
    except Exception as exc:
        logger.warning("Guidance cross-reference extraction skipped: %s", exc)
        relations = []

    return {
        "graph_version": "0.1",
        "celex_id": doc_id,
        "regulation_id": doc_name,
        "source_name": doc_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provisions": provisions,
        "relations": relations,
        "defined_terms": [],
    }


# ── internal ───────────────────────────────────────────────────────────────

def _build_provision_tree(
    md_text: str,
    doc_id: str,
    doc_name: str,
    lang: str,
) -> list[dict[str, Any]]:
    """Split markdown into heading-delimited sections and build a tree."""

    # Phase 1: split into (heading_level, section_number, title, body) tuples
    sections = _split_sections(md_text)

    if not sections:
        logger.warning("No sections found in markdown for %s", doc_id)
        return []

    # Phase 2: Build provision dicts
    provisions: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}

    # Root document node
    root_id = f"{doc_id}_document"
    root = _make_provision(
        id=root_id,
        kind="guidance_document",
        text=doc_name,
        title=doc_name,
        number=None,
        hierarchy_depth=0,
        path=[],
        parent_id=None,
        children=[],
        lang=lang,
    )
    provisions.append(root)
    by_id[root_id] = root

    # Track the "stack" of open parents at each depth to determine
    # where to place each section.  depth_map[depth] = provision_id
    depth_map: dict[int, str] = {0: root_id}

    for heading_level, sec_num, title, body, is_chart, chart_id in sections:
        # Determine kind and ID
        if is_chart:
            kind = "guidance_chart"
            prov_id = f"{doc_id}_chart_{chart_id.replace(' ', '_')}"
            number = chart_id
            display = f"Chart {chart_id}"
        elif sec_num:
            depth_count = sec_num.count(".")
            if depth_count == 0:
                kind = "guidance_section"
            elif depth_count == 1:
                kind = "guidance_subsection"
            else:
                kind = "guidance_paragraph"
            # Normalize: "4.3.2" → "4_3_2"
            num_slug = sec_num.replace(".", "_")
            prov_id = f"{doc_id}_sec_{num_slug}"
            number = sec_num
            display = f"Section {sec_num}"
        else:
            # Unnumbered heading (e.g., "# Annex", "#### Additional considerations")
            # Map by heading level
            if heading_level <= 1:
                kind = "guidance_section"
            elif heading_level == 2:
                kind = "guidance_section"
            elif heading_level == 3:
                kind = "guidance_subsection"
            else:
                kind = "guidance_paragraph"
            # Generate a slug from the title
            slug = _slugify(title)[:50]
            prov_id = f"{doc_id}_{slug}"
            number = None
            display = title[:80]

        # Avoid duplicate IDs
        if prov_id in by_id:
            prov_id = f"{prov_id}_{len(provisions)}"

        # Determine parent: look for the nearest ancestor with a shallower
        # heading level.
        if is_chart:
            # Charts are children of the Annex section if one exists,
            # otherwise direct children of root.
            parent_depth = 1  # treat charts at depth 2
        else:
            parent_depth = heading_level - 1

        # Find parent by walking depth_map downward
        parent_id = root_id
        for d in range(parent_depth, -1, -1):
            if d in depth_map:
                parent_id = depth_map[d]
                break

        # Set the current provision in the depth map at its level
        current_depth = heading_level if not is_chart else 2
        depth_map[current_depth] = prov_id
        # Clear deeper levels (they're no longer valid parents)
        for d in list(depth_map.keys()):
            if d > current_depth:
                del depth_map[d]

        # Compute path and hierarchy_depth from parent
        parent_prov = by_id[parent_id]
        path = parent_prov["path"] + [parent_id]
        hierarchy_depth = len(path)

        # Combine title and body text
        text = body.strip() if body else ""
        if title and text:
            text = f"{title}\n\n{text}"
        elif title:
            text = title

        prov = _make_provision(
            id=prov_id,
            kind=kind,
            text=text,
            title=title,
            number=number,
            hierarchy_depth=hierarchy_depth,
            path=path,
            parent_id=parent_id,
            children=[],
            lang=lang,
            display_ref=display,
        )
        provisions.append(prov)
        by_id[prov_id] = prov

        # Register as child of parent
        parent_prov["children"].append(prov_id)

    return provisions


def _split_sections(
    md_text: str,
) -> list[tuple[int, str | None, str, str, bool, str | None]]:
    """Split markdown into sections delimited by headings.

    Returns list of:
        (heading_level, section_number, title, body_text, is_chart, chart_id)
    """
    lines = md_text.split("\n")
    sections: list[tuple[int, str | None, str, str, bool, str | None]] = []
    current: dict[str, Any] | None = None

    # Track which Chart IDs we've seen as ## headings so we can merge
    # the ### subtitle into the same section body instead of creating
    # a duplicate node.
    seen_chart_ids: set[str] = set()

    in_preamble = True  # skip lines before the first numbered section

    for line in lines:
        stripped = line.strip()

        # Skip footnotes section entirely
        if _FOOTNOTES_RE.match(stripped):
            # Flush current
            if current:
                sections.append(_flush(current))
                current = None
            break  # everything after "## Footnotes" is footnote content

        # Skip contents heading
        if _CONTENTS_RE.match(stripped):
            continue

        # Check for heading
        if stripped.startswith("#"):
            # Check for chart heading
            chart_m = _CHART_HEADING_RE.match(stripped)
            heading_m = _HEADING_RE.match(stripped)

            if chart_m:
                hashes = chart_m.group("hashes")
                level = len(hashes)
                raw_chart_id = chart_m.group("chart_id").strip()
                chart_id = raw_chart_id if raw_chart_id != "Main Chart" else "Main"
                chart_title = _clean_heading(stripped.lstrip("#").strip())

                # If this is a ### subtitle for a chart we already opened
                # at ## level (e.g., "## Chart B" then "### Chart B - Change
                # of the Design*"), merge into the current section body.
                if level >= 3 and chart_id in seen_chart_ids and current and current.get("is_chart"):
                    current["body_lines"].append(stripped)
                    continue

                in_preamble = False
                if current:
                    sections.append(_flush(current))

                seen_chart_ids.add(chart_id)
                current = {
                    "level": level,
                    "sec_num": None,
                    "title": chart_title,
                    "body_lines": [],
                    "is_chart": True,
                    "chart_id": chart_id,
                }
                continue

            if heading_m:
                hashes = heading_m.group("hashes")
                level = len(hashes)
                sec_num = heading_m.group("sec_num")
                title = _clean_heading(heading_m.group("title"))

                # Skip preamble headings (title duplicates, revision notes)
                if _SKIP_HEADING_RE.match(stripped):
                    continue

                # A numbered section means we're past the preamble —
                # BUT skip TOC entries whose titles end with a page number
                # (e.g. "Annex II - ... 31").
                if sec_num and not _TOC_PAGE_RE.search(title):
                    in_preamble = False

                # Skip TOC entries that look like "## 9. Annex II ... 31"
                # (they appear in the Contents section with page numbers)
                if in_preamble:
                    continue

                # Normalise: a # heading inside the body (e.g. "# Annex I")
                # is treated as ## so it becomes a sibling of normal sections
                # rather than an ancestor that swallows everything below it.
                if level == 1:
                    level = 2

                if current:
                    sections.append(_flush(current))

                current = {
                    "level": level,
                    "sec_num": sec_num,
                    "title": title,
                    "body_lines": [],
                    "is_chart": False,
                    "chart_id": None,
                }
                continue

        # Body line
        if not in_preamble and current is not None:
            current["body_lines"].append(line)
        elif not in_preamble and current is None and stripped:
            # Text before any heading but after preamble — unusual, skip
            pass

    # Flush last section
    if current:
        sections.append(_flush(current))

    return sections


def _flush(
    current: dict[str, Any],
) -> tuple[int, str | None, str, str, bool, str | None]:
    body = "\n".join(current["body_lines"])
    return (
        current["level"],
        current["sec_num"],
        current["title"],
        body,
        current["is_chart"],
        current["chart_id"],
    )


def _make_provision(
    *,
    id: str,
    kind: str,
    text: str,
    title: str | None,
    number: str | None,
    hierarchy_depth: int,
    path: list[str],
    parent_id: str | None,
    children: list[str],
    lang: str,
    display_ref: str | None = None,
) -> dict[str, Any]:
    """Create a provision dict matching the parsed.json schema."""
    if display_ref is None:
        display_ref = title or kind
    return {
        "id": id,
        "kind": kind,
        "text": text,
        "title": title,
        "number": number,
        "hierarchy_depth": hierarchy_depth,
        "path": list(path),
        "parent_id": parent_id,
        "children": list(children),
        "lang": lang,
        "display_ref": display_ref,
    }


def _slugify(text: str) -> str:
    """Convert text to a safe ID slug."""
    text = _SUP_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


# ── CLI / standalone usage ─────────────────────────────────────────────────

def write_parsed_json(
    md_path: str | Path,
    doc_id: str,
    doc_name: str,
    lang: str = "EN",
    output_path: str | Path | None = None,
) -> Path:
    """Structure markdown and write parsed.json.

    Parameters
    ----------
    md_path:
        Path to the clean markdown file.
    doc_id:
        Document identifier (e.g., ``"MDCG_2020_3"``).
    doc_name:
        Human-readable name (e.g., ``"MDCG 2020-3 Rev.1"``).
    lang:
        Language code.
    output_path:
        Where to write parsed.json. Defaults to the same directory
        as *md_path*.

    Returns
    -------
    Path
        Path to the written ``parsed.json`` file.
    """
    import json

    result = structure_mdcg(md_path, doc_id, doc_name, lang)

    if output_path is None:
        output_path = Path(md_path).parent / "parsed.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    n_prov = len(result["provisions"])
    logger.info("Wrote %s — %d provisions.", output_path, n_prov)
    return output_path
