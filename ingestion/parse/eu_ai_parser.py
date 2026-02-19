"""EU AI Act parser that emits RAG-optimised graph nodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from domain.regulations_catalog import REGULATIONS
from domain.ontology.roles import eu_ai_role_detector
from .base.hierarchy import LEVEL_ORDER
from ...domain.ontology.lang_keywords import LANG_KEYWORDS
from .base.parser_core import (
    ParserConfig,
    build_provision_record,
    canonicalize_numbered_level,
    flatten_relations,
)
from .base.parser_utils import classify_block, detect_numbering, normalize_text

# Force-load EU AI Act role detector (registers via side-effect)
CELEX_ID = "32024R1689"
SOURCE_NAME = "EU_AI_ACT_2024"
REGULATION_NAME = REGULATIONS.get(CELEX_ID, {}).get("name", SOURCE_NAME)
REGULATION_ID = REGULATION_NAME or SOURCE_NAME
LEVEL_NORMALIZATION = {
    "section": "paragraph",   # numbering "1." within articles
    "subsection": "paragraph",
    "point": "letter",        # (a) should align with LEVEL_ORDER's "letter"
}


CONFIG = ParserConfig(
    celex_id=CELEX_ID,
    source_name=SOURCE_NAME,
    regulation_id=REGULATION_ID,
    parser_name="eu_ai_parser.parse_eu_ai_act",
    level_normalization=LEVEL_NORMALIZATION,
    context_levels=("title", "chapter", "section", "article"),
    non_requirement_levels={"title", "chapter", "section", "annex", "recital"},
    reference_excluded_levels=set(),
    role_detector=eu_ai_role_detector,
)


def parse_eu_ai_act(html_file: Path, lang: str = "EN") -> Tuple[List[Dict], List[Tuple[str, str, str]]]:
    """
    Parse the EU AI Act HTML into a structured list of provisions and explicit relations.

    Parameters
    ----------
    html_file: Path
        Path to HTML file of the EU AI Act.
    lang : str, optional
        Language code for parsing (default is "EN").

    Returns
    -------
    Tuple[List[Dict], List[Tuple[str, str, str]]]
        - provisions: list of provision dictionaries, each containing:
            * id, parent_id, level, kind, item_number, title, text, intro_text
            * path (hierarchy path from root)
            * is_requirement, requirement_type
            * metadata: celex_id, source, lang and current hierarchy context
            * references: cross-references extracted from text
        - relations: explicit triples of the form (source_id, relation_type, target_id)
            e.g., ("Article_3", "HAS_CHILD", "Article_3_Paragraph_1")
    """

    lang = lang.upper()
    keywords = LANG_KEYWORDS.get(lang, LANG_KEYWORDS["EN"])

    # Load HTML with BeautifulSoup and keep raw HTML for provenance
    raw_html = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "lxml")
    blocks = soup.find_all(["p", "table", "h1", "h2", "h3", "h4"])

    provisions: List[Dict] = []
    relations: List[Tuple[str, str, str]] = []
    stack: List[Dict] = []
    recital_counter = 0
    in_preamble = True

    for tag in blocks:
        block = classify_block(tag, lang)
        if not block:
            continue

        text = normalize_text(block["text"])
        if not text:
            continue

        block_type = block["type"]
        level: Optional[str] = None
        marker: Optional[str] = None
        body = text
        title_text: Optional[str] = None
        intro_text = ""

        if block_type in ("title", "chapter_title", "section_title", "article_title", "annex_title"):
            if block_type == "title":
                level = "title"
                title_text = text
                m = re.search(rf"{keywords.get('title','TITLE')}\s*[:\-.–—]?\s*([IVXLCDM]+)\b", text, re.I)
                m = m or re.search(r"\b([IVXLCDM]+)\b", text, re.I)
                marker = m.group(1) if m else "TITLE"
            elif block_type == "chapter_title":
                level = "chapter"
                title_text = text
                m = re.search(rf"{keywords.get('chapter','CHAPTER')}\s*[:\-.–—]?\s*([IVXLCDM]+)\b", text, re.I)
                m = m or re.search(r"\b([IVXLCDM]+)\b", text, re.I)
                marker = m.group(1) if m else "CHAPTER"
            elif block_type == "section_title":
                level = "section"
                title_text = text
                m = re.search(rf"{keywords.get('section','SECTION')}\s*[:\-.–—]?\s*([IVXLCDM]+|\d+)\b", text, re.I)
                if not m:
                    m = re.search(r"\b([IVXLCDM]+|\d+)\b", text, re.I)
                marker = m.group(1) if m else "SECTION"
            elif block_type == "article_title":
                level = "article"
                m = re.match(rf"^{keywords.get('article','Article')}\s*[:\-.–—]?\s*(\d+)\b\s*(.*)$", text, re.I)
                if m:
                    marker = m.group(1)
                    short_title = m.group(2).strip()
                    title_text = f"{keywords.get('article','Article')} {marker}"
                    if short_title:
                        intro_text = short_title
                else:
                    title_text = text
                    m2 = re.search(r"\b(\d+)\b", text)
                    marker = m2.group(1) if m2 else "ARTICLE"
            elif block_type == "annex_title":
                level = "annex"
                title_text = text
                m = re.search(rf"{keywords.get('annex','ANNEX')}\s*[:\-.–—]?\s*([IVXLCDM]+)\b", text, re.I)
                m = m or re.search(r"\b([IVXLCDM]+)\b", text, re.I)
                marker = m.group(1) if m else keywords.get("annex", "ANNEX")

        if level and level in {"title", "chapter", "section", "article", "annex"}:
            in_preamble = False

        if not level:
            numbered_level, numbered_marker, body_candidate = detect_numbering(text)
            if body_candidate:
                body = body_candidate
            level = canonicalize_numbered_level(numbered_level, CONFIG)
            marker = marker or numbered_marker

        if level == "paragraph" and in_preamble and not stack:
            level = "recital"
            if marker and marker.isdigit():
                recital_counter = max(recital_counter, int(marker))
            else:
                recital_counter += 1
                marker = marker or f"PREAMBLE_{recital_counter}"
            title_text = title_text or text

        if not level and re.match(r"^(Whereas|Considérant|Erwägungsgrund)", text, re.I):
            level = "recital"
            m = re.search(r"\b(\d+)\b", text)
            if m and m.group(1).isdigit():
                marker = m.group(1)
                recital_counter = max(recital_counter, int(marker))
            else:
                recital_counter += 1
                marker = marker or f"PREAMBLE_{recital_counter}"
            title_text = title_text or text

        if not level:
            if stack:
                parent_intro = stack[-1].get("intro_text", "")
                stack[-1]["intro_text"] = (parent_intro + " " + text).strip()
            continue

        current_rank = LEVEL_ORDER.get(level, 99)
        while stack and (
            LEVEL_ORDER.get(stack[-1]["level"], 99) >= current_rank
            or stack[-1]["level"] == "recital"
        ):
            stack.pop()

        parent_id = stack[-1]["id"] if stack else None

        provision = build_provision_record(
            config=CONFIG,
            stack=stack,
            level=level,
            marker=marker,
            lang=lang,
            title=title_text,
            text=body,
            intro_text=intro_text,
            parent_id=parent_id,
            raw_html=raw_html,
            tag=tag,
            html_file=html_file,
            text_for_analysis=body,
        )

        provisions.append(provision)

        if parent_id:
            relations.append((parent_id, "HAS_CHILD", provision["id"]))
        for ref in provision["references"]:
            relations.append((provision["id"], "REFERENCES", ref))

        # Recitals are top-level siblings — do NOT clear stack, just append
        stack.append(provision)

    relations_obj = flatten_relations(relations)

    return provisions, relations_obj
