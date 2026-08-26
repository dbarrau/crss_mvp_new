"""MDCG guidance document parser using LlamaParse v2.

Converts MDCG guidance PDFs into clean structured markdown with
consolidated footnotes and extracted decision-tree flowcharts.

Public API
----------
- :func:`parse_mdcg_pdf` — full pipeline: upload → parse → clean → extract.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml

if TYPE_CHECKING:  # avoid an import cycle with profiles.py at runtime
    from .profiles import GuidanceProfile

logger = logging.getLogger(__name__)

# Post-processing callables supplied by a parse profile.
CleanFn = Callable[[str], "tuple[str, dict]"]
ExtractFn = Callable[[str], "list[dict]"]

# ── LlamaParse custom prompt ───────────────────────────────────────────────

MDCG_CUSTOM_PROMPT = """
You are parsing an official MDCG (Medical Device Coordination Group) guidance
document published by the European Commission. These are regulatory guidance
PDFs with numbered sections, footnotes, and decision-tree flowcharts.

CRITICAL INSTRUCTIONS:

1. CONTINUOUS DOCUMENT — Treat the entire PDF as ONE continuous document.
   Do NOT restart headings or numbering at page boundaries.

2. REMOVE REPEATED HEADERS/FOOTERS — The following text appears on nearly
   every page as a running header or footer. REMOVE every occurrence EXCEPT
   the very first one at the start of the document:
   - "Medical Devices"
   - "Medical Device Coordination Group Document"
   - "MDCG 2020-3 Rev.1" (or any MDCG document reference that repeats)
   Remove page numbers as well.

3. SECTION HIERARCHY — Preserve the original numbered section structure
   exactly. Map section numbers to markdown heading levels:
   - Document title → # (level 1)
   - Top sections (1, 2, 3, 4) → ## (level 2)
   - Subsections (4.1, 4.2, 4.3) → ### (level 3)
   - Sub-subsections (4.3.1, 4.3.2) → #### (level 4)
   - Deeper (4.3.2.1, 4.3.2.2, 4.3.2.3) → ##### (level 5)

4. FOOTNOTES — Collect ALL footnotes from the entire document.
   Place them in a SINGLE section titled "## Footnotes" at the very end
   of the document. Preserve the original footnote numbering (superscript
   numbers become plain numbers like "1.", "2.", etc.). Do NOT scatter
   footnotes at the end of each page — consolidate them ALL at the end.

5. FLOWCHARTS (DECISION TREES) — The document contains decision-tree
   flowcharts labeled "Main Chart", "Chart A", "Chart B", "Chart C",
   "Chart D", "Chart E". For each flowchart:
   - Start with the chart title as a heading
   - Represent each decision node as a numbered step with the question
   - Show Yes/No branches using indented bullet points
   - Use → to indicate flow direction
   - Example format:
     **Step B1**: Change of built-in control mechanism, operating principles,
     source of energy or alarm systems?
       - **Yes** → The change is considered significant
       - **No** → Go to Step B2

6. FORMATTING — Preserve bold (**text**) and italic (*text*) formatting.
   Keep bullet lists as markdown lists. Keep all legal references exactly
   as written (Article numbers, Regulation references, Directive references).

7. EXAMPLES — When the document provides lists of "Non-significant" and
   "Significant" change examples, preserve them as bullet lists under clear
   subheadings.

8. TABLES — If the document contains tables, render them as markdown tables.

9. CONTENT PRESERVATION — Do NOT omit, summarize, or paraphrase any content.
   Include every sentence, every example, every footnote from the original.
"""


# ── Post-processing ───────────────────────────────────────────────────────

# Running-header/footer patterns specific to MDCG PDFs. Passed into the generic
# cleaner by the MDCG profile; other families supply their own (or none).
MDCG_HEADER_PATTERNS = [
    r"^#{1,3}\s*Medical Devices\s*$",
    r"^#{1,3}\s*Medical Device Coordination Group Document\s*"
    r"(?:MDCG\s+\d{4}-\d+\s*(?:Rev\.\d+)?)?\s*$",
    r"^Medical Devices\s*$",
]


def _html_table_to_markdown(block: str) -> str:
    """Convert a single ``<table>…</table>`` block to a markdown table."""
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", block, re.DOTALL | re.IGNORECASE)
    md_rows: list[str] = []
    for row in rows:
        cells = re.findall(
            r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row, re.DOTALL | re.IGNORECASE
        )
        cleaned = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip().replace("|", "\\|")
            for c in cells
        ]
        if any(cleaned):
            md_rows.append("| " + " | ".join(cleaned) + " |")
    if not md_rows:
        return ""
    ncol = md_rows[0].count("|") - 1
    sep = "| " + " | ".join(["---"] * ncol) + " |"
    return "\n" + md_rows[0] + "\n" + sep + "\n" + "\n".join(md_rows[1:]) + "\n"


def clean_guidance_markdown(
    raw: str,
    *,
    header_patterns: list[str] | tuple[str, ...] = (),
) -> tuple[str, dict]:
    """Clean LlamaParse output for a guidance document.

    Family-agnostic: the only publisher-specific input is *header_patterns*
    (repeated running headers/footers to dedupe). Everything else — page-break
    separators, HTML-table conversion, page-number/TOC-leader stripping,
    footnote consolidation, heading normalisation, whitespace — is common
    across guidance families.

    Returns ``(cleaned_markdown, metrics_dict)``.
    """
    metrics: dict = {}
    text = raw

    # 1. Remove page-break separators
    before = len(re.findall(r"^\s*-{3,}\s*$", text, re.MULTILINE))
    text = re.sub(r"\n\s*-{3,}\s*\n", "\n\n", text)
    metrics["separators_removed"] = before

    # 1a. Convert HTML tables to markdown (LlamaParse sometimes emits <table>
    #     despite the markdown-table instruction). Keeps tabular content usable
    #     downstream instead of embedding raw HTML.
    n_tables = len(re.findall(r"<table\b", text, re.IGNORECASE))
    if n_tables:
        text = re.sub(
            r"<table\b[^>]*>.*?</table>",
            lambda m: _html_table_to_markdown(m.group(0)),
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    metrics["html_tables_converted"] = n_tables

    # 1b. Strip table-of-contents leader lines (runs of 5+ dots are never body
    #     text — always ".... <page-number>" TOC entries).
    toc = len(re.findall(r"(?m)^.*\.{5,}.*$", text))
    text = re.sub(r"(?m)^.*\.{5,}.*$\n?", "", text)
    metrics["toc_leader_lines_removed"] = toc

    # 1c. Strip standalone page-number lines (a bare 1-3 digit line left behind
    #     when the crop-box misses a page number mid-column). Paragraph markers
    #     are "(1)" and headings are "## 1." — a lone number is always an artifact.
    pages = len(re.findall(r"(?m)^[ \t]*\d{1,3}[ \t]*$", text))
    text = re.sub(r"(?m)^[ \t]*\d{1,3}[ \t]*$\n?", "", text)
    metrics["page_number_lines_removed"] = pages

    # 2. Remove duplicate running headers (keep first occurrence)
    total_header_removals = 0
    for pat in header_patterns:
        matches = list(re.finditer(pat, text, re.MULTILINE))
        if len(matches) > 1:
            for m in reversed(matches[1:]):
                text = text[: m.start()] + text[m.end() :]
                total_header_removals += 1
    metrics["duplicate_headers_removed"] = total_header_removals

    # 3. Consolidate footnotes.
    #    A footnote block runs from a "## Footnotes" heading until the next
    #    heading *of any level* OR the next body-paragraph marker "(N)". Both
    #    boundaries matter: LlamaParse can emit per-page footnote blocks
    #    interleaved with body, and the body that resumes after them uses deep
    #    (####/#####) subheadings and "(N)" paragraph markers. Stopping only at
    #    1-3-hash headings (the old rule) silently ate real paragraphs.
    _fn_body = r"(?:(?!^#{1,6}\s)(?!^\(\d+\)).*\n?)*"
    footnote_pattern = r"^#{1,3}\s*Footnotes?\s*\n(" + _fn_body + r")"
    footnote_sections = re.findall(footnote_pattern, text, re.MULTILINE)

    all_footnotes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for section in footnote_sections:
        fns = re.findall(
            r"^\s*(?:(\d+)[\.\):]?\s+(.+?))\s*$", section, re.MULTILINE
        )
        for num, content in fns:
            norm = re.sub(r"\s+", " ", content.strip().lower())
            if norm not in seen and len(norm) > 5:
                seen.add(norm)
                all_footnotes.append((num, content.strip()))

    text = re.sub(
        r"\n*^#{1,3}\s*Footnotes?\s*\n" + _fn_body,
        "\n",
        text,
        flags=re.MULTILINE,
    )
    metrics["footnote_sections_consolidated"] = len(footnote_sections)
    metrics["unique_footnotes"] = len(all_footnotes)

    # 4. Normalize heading hierarchy
    def _fix_heading(m: re.Match) -> str:
        num = m.group(2)
        title = m.group(3)
        depth = len(num.split("."))
        level = min(depth + 1, 6)
        return f"{'#' * level} {num} {title}"

    text = re.sub(
        r"^(#{1,6})\s+(\d+(?:\.\d+)*)\s+(.+)$",
        _fix_heading,
        text,
        flags=re.MULTILINE,
    )

    # 5. Clean whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = text.strip() + "\n"

    # 6. Append consolidated footnotes
    if all_footnotes:
        block = "\n\n## Footnotes\n\n"
        for num, content in all_footnotes:
            block += f"{num}. {content}\n\n"
        text += block.rstrip() + "\n"

    return text, metrics


def clean_mdcg_markdown(raw: str) -> tuple[str, dict]:
    """MDCG-specific cleaner: the generic cleaner with MDCG header patterns.

    Kept as a named entry point for the MDCG parse profile and any existing
    callers.
    """
    return clean_guidance_markdown(raw, header_patterns=MDCG_HEADER_PATTERNS)


# ── Flowchart extraction ──────────────────────────────────────────────────

def extract_flowcharts(md_text: str) -> list[dict]:
    """Extract decision-tree flowcharts from cleaned MDCG markdown.

    Returns a list of chart dicts, each containing ``chart_id``,
    ``title``, ``reference_section``, and ``steps``.
    """
    charts: list[dict] = []

    chart_heading_pat = re.compile(
        r"^#{1,6}\s+.*?"
        r"(?:(Main\s+Chart)|(?:Chart\s+)([A-E]))"
        r"(?:\s*[:\-–—]\s*(.+?))?$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(chart_heading_pat.finditer(md_text))

    body_pat = re.compile(
        r"(?:^|\n)\s*\*\*(?:(Main\s+Chart)|Chart\s+([A-E])[^*]*)\*\*",
        re.MULTILINE | re.IGNORECASE,
    )
    for bm in body_pat.finditer(md_text):
        near = any(abs(bm.start() - hm.start()) < 200 for hm in matches)
        if not near:
            matches.append(bm)

    matches.sort(key=lambda m: m.start())

    if not matches:
        logger.warning("No flowchart sections found in the markdown.")
        return charts

    for i, m in enumerate(matches):
        chart_letter = None
        for g in range(1, (m.lastindex or 0) + 1):
            val = m.group(g)
            if val:
                chart_letter = "Main" if "main" in val.lower() else val.strip()
                break
        chart_letter = chart_letter or "Main"
        chart_title = ""
        if m.lastindex and m.lastindex >= 3 and m.group(3):
            chart_title = m.group(3).strip()

        start = m.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else min(start + 3000, len(md_text))
        )
        block = md_text[start:end]

        steps: list[dict] = []

        step_re = re.compile(
            r"\*\*(?:Step\s+)?([A-Z]?\d+|[A-Z]|\d+)\*\*[:\s]*(.+?)"
            r"(?=\n\s*[-*•]|\n\n|\Z)",
            re.DOTALL,
        )
        for sm in step_re.finditer(block):
            step_id = sm.group(1).strip()
            question = re.sub(r"\s+", " ", sm.group(2).strip()).rstrip("*").rstrip()
            if not question.endswith("?"):
                question += "?"

            after = block[sm.end() : sm.end() + 500]
            yes_m = re.search(
                r"[-*•]\s*\*?\*?Yes\*?\*?\s*[→:–\-]\s*(.+?)(?:\n|$)", after, re.I
            )
            no_m = re.search(
                r"[-*•]\s*\*?\*?No\*?\*?\s*[→:–\-]\s*(.+?)(?:\n|$)", after, re.I
            )

            if chart_letter == "Main" and len(step_id) == 1:
                sid = f"Main-{step_id}"
            elif step_id[0].isdigit() and chart_letter != "Main":
                sid = f"{chart_letter}{step_id}"
            else:
                sid = step_id

            step: dict = {"id": sid, "question": question}
            if yes_m:
                step["yes"] = yes_m.group(1).strip()
            if no_m:
                step["no"] = no_m.group(1).strip()
            steps.append(step)

        # Fallback: bold question lines
        if not steps:
            current_step = None
            idx = 0
            for line in block.split("\n"):
                s = line.strip()
                if not s:
                    continue
                qm = re.match(r"\*\*(.{10,}?)(\?)?\*\*", s)
                if qm:
                    idx += 1
                    q = qm.group(1).strip()
                    if not q.endswith("?"):
                        q += "?"
                    current_step = {
                        "id": f"{chart_letter}{idx}",
                        "question": q,
                    }
                    steps.append(current_step)
                    continue
                if current_step:
                    if re.match(r"[-*•]\s*\*?\*?Yes\b", s, re.I):
                        tail = re.sub(
                            r"^[-*•]\s*\*?\*?Yes\*?\*?\s*[→:–\-]?\s*",
                            "",
                            s,
                            flags=re.I,
                        )
                        current_step["yes"] = tail.strip() or "significant"
                    elif re.match(r"[-*•]\s*\*?\*?No\b", s, re.I):
                        tail = re.sub(
                            r"^[-*•]\s*\*?\*?No\*?\*?\s*[→:–\-]?\s*",
                            "",
                            s,
                            flags=re.I,
                        )
                        current_step["no"] = tail.strip() or "non-significant"

        ref_m = re.search(r"Section\s+(\d+(?:\.\d+)*)", block[:500])
        charts.append(
            {
                "chart_id": chart_letter,
                "title": chart_title
                or ("Main Chart" if chart_letter == "Main" else f"Chart {chart_letter}"),
                "reference_section": ref_m.group(1) if ref_m else "",
                "steps": steps,
            }
        )

    # Deduplicate: keep entry with most steps per chart_id
    deduped: dict[str, dict] = {}
    for c in charts:
        cid = c["chart_id"]
        if cid not in deduped or len(c["steps"]) > len(deduped[cid]["steps"]):
            if (
                cid in deduped
                and c["title"] == f"Chart {cid}"
                and deduped[cid]["title"] != f"Chart {cid}"
            ):
                c["title"] = deduped[cid]["title"]
            deduped[cid] = c
    return list(deduped.values())


# ── Main entry point ──────────────────────────────────────────────────────

async def _parse_pdf_async(
    pdf_path: Path,
    output_dir: Path,
    api_key: str,
    tier: str,
    custom_prompt: str,
    clean_fn: "CleanFn",
    extract_fn: "ExtractFn | None",
) -> dict:
    """Async implementation of the full guidance parsing pipeline.

    Publisher-specific behaviour is injected via *custom_prompt*, *clean_fn*
    and *extract_fn* (all supplied by the parse profile) so this engine is
    family-agnostic.
    """
    from llama_cloud import AsyncLlamaCloud

    client = AsyncLlamaCloud(api_key=api_key)

    logger.info("Uploading %s to LlamaParse…", pdf_path.name)
    file_obj = await client.files.create(file=str(pdf_path), purpose="parse")

    logger.info("Parsing with %s tier (large docs can take several minutes)…", tier)
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier=tier,
        version="latest",
        agentic_options={"custom_prompt": custom_prompt},
        crop_box={"top": 0.07, "bottom": 0.04, "left": 0.0, "right": 0.0},
        output_options={"markdown": {"annotate_links": True}},
        expand=["markdown", "text"],
    )

    raw = "\n\n".join(p.markdown for p in result.markdown.pages)
    logger.info(
        "Raw output: %d chars across %d pages",
        len(raw),
        len(result.markdown.pages),
    )

    # Clean
    cleaned_md, cleaning_metrics = clean_fn(raw)
    logger.info("Cleaned: %d chars (%.1f%% reduction)",
                len(cleaned_md),
                (len(raw) - len(cleaned_md)) / max(len(raw), 1) * 100)

    # Flowcharts (only families that have them supply an extractor)
    flowcharts = extract_fn(cleaned_md) if extract_fn is not None else []
    logger.info("Extracted %d flowchart(s)", len(flowcharts))

    # Save outputs
    stem = pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    raw_path = output_dir / f"{stem}_raw.md"
    raw_path.write_text(raw, encoding="utf-8")
    files["raw_markdown"] = str(raw_path)

    clean_path = output_dir / f"{stem}_clean.md"
    clean_path.write_text(cleaned_md, encoding="utf-8")
    files["clean_markdown"] = str(clean_path)

    if flowcharts:
        yaml_path = output_dir / f"{stem}_flowcharts.yaml"
        yaml_path.write_text(
            yaml.dump(
                flowcharts,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        files["flowcharts_yaml"] = str(yaml_path)

    meta = {
        "source_pdf": pdf_path.name,
        "tier": tier,
        "pages": len(result.markdown.pages),
        "raw_chars": len(raw),
        "cleaned_chars": len(cleaned_md),
        "cleaning_metrics": cleaning_metrics,
        "flowcharts_count": len(flowcharts),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    files["metadata"] = str(meta_path)

    return {
        "markdown": cleaned_md,
        "flowcharts": flowcharts,
        "metrics": cleaning_metrics,
        "output_files": files,
    }


def parse_guidance_pdf(
    pdf_path: Path,
    output_dir: Path,
    profile: "GuidanceProfile",
    api_key: str | None = None,
    tier: str | None = None,
) -> dict:
    """Parse a guidance PDF end-to-end, driven by a parse *profile*.

    1. Upload & parse via LlamaParse v2 (profile's tier)
    2. Post-process via the profile's ``clean_fn`` (dedup headers, consolidate
       footnotes, fix hierarchy)
    3. Extract structures (e.g. decision-tree flowcharts) via the profile's
       ``extract_fn`` if it has one
    4. Save all outputs to *output_dir*

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory where outputs are written.
        profile: The parse profile (prompt + post-processing) for this family.
        api_key: LlamaParse API key. Falls back to ``LLAMA_CLOUD_API_KEY`` env var.
        tier: Override the profile's LlamaParse tier.

    Returns:
        Dict with ``markdown``, ``flowcharts``, ``metrics``, ``output_files``.
    """
    key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY", "")
    if not key:
        raise RuntimeError(
            "No LLAMA_CLOUD_API_KEY provided. Set it in .env or pass api_key=."
        )
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    effective_tier = tier or profile.tier

    args = (
        pdf_path,
        output_dir,
        key,
        effective_tier,
        profile.custom_prompt,
        profile.clean_fn,
        profile.extract_fn,
    )

    # Run the async pipeline in the current or new event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an async context (e.g. Jupyter) — run in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(asyncio.run, _parse_pdf_async(*args)).result()
    else:
        result = asyncio.run(_parse_pdf_async(*args))

    return result


def parse_mdcg_pdf(
    pdf_path: Path,
    output_dir: Path,
    api_key: str | None = None,
    tier: str = "agentic",
    custom_prompt: str | None = None,
) -> dict:
    """Parse an MDCG guidance PDF (backward-compatible wrapper).

    Delegates to :func:`parse_guidance_pdf` with the MDCG profile. A
    *custom_prompt* override is honoured for callers that already pass one.
    """
    from .profiles import GuidanceProfile, get_profile

    profile = get_profile("mdcg")
    if custom_prompt is not None:
        profile = GuidanceProfile(
            name=profile.name,
            custom_prompt=custom_prompt,
            clean_fn=profile.clean_fn,
            extract_fn=profile.extract_fn,
            tier=profile.tier,
        )
    return parse_guidance_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        profile=profile,
        api_key=api_key,
        tier=tier,
    )
