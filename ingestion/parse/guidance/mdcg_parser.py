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

import yaml

logger = logging.getLogger(__name__)

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

def clean_mdcg_markdown(raw: str) -> tuple[str, dict]:
    """Clean LlamaParse output for MDCG documents.

    Returns ``(cleaned_markdown, metrics_dict)``.
    """
    metrics: dict = {}
    text = raw

    # 1. Remove page-break separators
    before = len(re.findall(r"^\s*-{3,}\s*$", text, re.MULTILINE))
    text = re.sub(r"\n\s*-{3,}\s*\n", "\n\n", text)
    metrics["separators_removed"] = before

    # 2. Remove duplicate running headers (keep first occurrence)
    header_patterns = [
        r"^#{1,3}\s*Medical Devices\s*$",
        r"^#{1,3}\s*Medical Device Coordination Group Document\s*"
        r"(?:MDCG\s+\d{4}-\d+\s*(?:Rev\.\d+)?)?\s*$",
        r"^Medical Devices\s*$",
    ]
    total_header_removals = 0
    for pat in header_patterns:
        matches = list(re.finditer(pat, text, re.MULTILINE))
        if len(matches) > 1:
            for m in reversed(matches[1:]):
                text = text[: m.start()] + text[m.end() :]
                total_header_removals += 1
    metrics["duplicate_headers_removed"] = total_header_removals

    # 3. Consolidate footnotes
    footnote_pattern = r"^#{1,3}\s*Footnotes?\s*\n((?:(?!^#{1,3}\s).*\n?)*)"
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
        r"\n*^#{1,3}\s*Footnotes?\s*\n(?:(?!^#{1,3}\s).*\n?)*",
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
) -> dict:
    """Async implementation of the full MDCG parsing pipeline."""
    from llama_cloud import AsyncLlamaCloud

    client = AsyncLlamaCloud(api_key=api_key)

    logger.info("Uploading %s to LlamaParse…", pdf_path.name)
    file_obj = await client.files.create(file=str(pdf_path), purpose="parse")

    logger.info("Parsing with %s tier (this may take 1-3 minutes)…", tier)
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
    cleaned_md, cleaning_metrics = clean_mdcg_markdown(raw)
    logger.info("Cleaned: %d chars (%.1f%% reduction)",
                len(cleaned_md),
                (len(raw) - len(cleaned_md)) / max(len(raw), 1) * 100)

    # Flowcharts
    flowcharts = extract_flowcharts(cleaned_md)
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


def parse_mdcg_pdf(
    pdf_path: Path,
    output_dir: Path,
    api_key: str | None = None,
    tier: str = "agentic",
    custom_prompt: str | None = None,
) -> dict:
    """Parse an MDCG guidance PDF end-to-end.

    1. Upload & parse via LlamaParse v2 (agentic tier)
    2. Post-process (dedup headers, consolidate footnotes, fix hierarchy)
    3. Extract decision-tree flowcharts
    4. Save all outputs to *output_dir*

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory where outputs are written.
        api_key: LlamaParse API key. Falls back to ``LLAMA_CLOUD_API_KEY`` env var.
        tier: LlamaParse tier (default ``"agentic"``).
        custom_prompt: Override for the MDCG-specific prompt.

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

    prompt = custom_prompt or MDCG_CUSTOM_PROMPT

    # Run the async pipeline in the current or new event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an async context (e.g. Jupyter) — use nest_asyncio
        # or create a new thread. We use a simple thread approach.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                asyncio.run,
                _parse_pdf_async(pdf_path, output_dir, key, tier, prompt),
            ).result()
    else:
        result = asyncio.run(
            _parse_pdf_async(pdf_path, output_dir, key, tier, prompt)
        )

    return result
