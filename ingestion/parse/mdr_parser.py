
"""
Parser for Regulation (EU) 2017/745 (MDR).

This module provides parsing utilities for extracting provisions and relations from the MDR (Medical Device Regulation) HTML document, producing outputs suitable for graph-based analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from domain.regulations_catalog import REGULATIONS
from domain.ontology.roles import mdr_role_detector
from .base.hierarchy import LEVEL_ORDER
from .base.lang_keywords import LANG_KEYWORDS
from .base.parser_core import (
    ParserConfig,
    build_provision_record,
    canonicalize_numbered_level,
    flatten_relations,
)
from .base.parser_utils import classify_block, detect_numbering, normalize_text

CELEX_ID = "32017R0745"
SOURCE_NAME = "MDR_2017_745"
REGULATION_NAME = REGULATIONS.get(CELEX_ID, {}).get("name", SOURCE_NAME)
REGULATION_ID = REGULATION_NAME or SOURCE_NAME


LEVEL_NORMALIZATION = {
    "section": "paragraph",
    "subsection": "paragraph",
    "point": "letter",
}


CONFIG = ParserConfig(
    celex_id=CELEX_ID,
    source_name=SOURCE_NAME,
    regulation_id=REGULATION_ID,
    parser_name="mdr_parser.parse_mdr",
    level_normalization=LEVEL_NORMALIZATION,
    context_levels=("title", "chapter", "section", "article", "annex"),
    non_requirement_levels={"title", "chapter", "section", "article", "annex"},
    reference_excluded_levels={"title", "chapter", "section", "article", "annex"},
    role_detector=mdr_role_detector,
)


def parse_mdr(html_file: Path, lang: str = "EN") -> Tuple[List[Dict], List[Dict]]:

    """
    Parse MDR HTML into provisions and relations matching the graph schema.

    Parameters
    ----------
    html_file : Path
        Path to the HTML file containing the MDR regulation.
    lang : str, optional
        Language code (default is 'EN'). Used to select language-specific keywords.

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        A tuple containing:
        - List of provision dictionaries, each representing a parsed provision.
        - List of relation dictionaries, each representing a relationship between provisions.

    Notes
    -----
    The function uses BeautifulSoup to parse the HTML and extract relevant blocks (paragraphs, tables, headers).
    It classifies and normalizes these blocks, builds provision records, and establishes parent-child and reference relations.
    """

    lang = (lang or "EN").upper()
    raw_html = html_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "lxml")
    blocks = soup.find_all(["p", "table", "h1", "h2", "h3", "h4"])

    provisions: List[Dict] = []
    relations: List[Tuple[str, str, str]] = []
    stack: List[Dict] = []
    keywords = LANG_KEYWORDS.get(lang, LANG_KEYWORDS["EN"])

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

        if not level:
            numbered_level, numbered_marker, body_candidate = detect_numbering(text)
            if body_candidate:
                body = body_candidate
            level = canonicalize_numbered_level(numbered_level, CONFIG)
            marker = marker or numbered_marker

        if not level:
            if stack:
                parent_intro = stack[-1].get("intro_text", "")
                stack[-1]["intro_text"] = (parent_intro + " " + text).strip()
            continue

        current_rank = LEVEL_ORDER.get(level, 99)
        while stack and LEVEL_ORDER.get(stack[-1]["level"], 99) >= current_rank:
            stack.pop()

        parent_id = stack[-1]["id"] if stack else None
        stored_text = text if level in CONFIG.non_requirement_levels else body

        provision = build_provision_record(
            config=CONFIG,
            stack=stack,
            level=level,
            marker=marker,
            lang=lang,
            title=title_text,
            text=stored_text,
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

        stack.append(provision)

    relations_obj = flatten_relations(relations)

    return provisions, relations_obj
