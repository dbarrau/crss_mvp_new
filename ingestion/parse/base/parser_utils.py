# /ingestion/parsers/base/parser_utils.py

"""
Common utilities for EU regulation parsers.
Includes:
- Text normalization
- Cross-reference extraction
- Block classification
- Numbering detection
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from bs4 import Tag
from ....domain.ontology.lang_keywords import LANG_KEYWORDS
import hashlib
from pathlib import Path
from .requirement_patterns import (is_requirement_text,
                                  classify_requirement_type)

# Role detector hook (can be registered by a regulation-specific parser)
ROLE_DETECTOR = lambda text, lang: []

def register_role_detector(fn):
    """Register a callable that detects roles in provision text.

    This avoids importing regulation-specific parsers from the base
    utilities and prevents circular imports. The callable should accept
    (text: str, lang: str) and return List[str].
    """
    global ROLE_DETECTOR
    ROLE_DETECTOR = fn


# ============================================================
# Text normalization STAYS
# ============================================================
def normalize_text(text: str) -> str:
    """Normalize and clean text for parser consumption.

    Performs Unicode normalization (NFKD), collapses consecutive
    whitespace into single spaces and trims leading/trailing space.

    :param text: Raw text extracted from HTML blocks.
    :return: Normalized, trimmed text safe for pattern matching.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# Block classification STAYS
# ============================================================
def classify_block(tag: Tag, lang: str) -> Optional[Dict]:
    """Classify a BeautifulSoup tag into a block type.

    Recognised types include ``annex_title``, ``chapter_title``,
    ``article_title``, ``table`` and ``paragraph``. The classifier uses
    language-specific structural keywords from :data:`LANG_KEYWORDS`.

    :param tag: A BeautifulSoup ``Tag`` object (e.g. ``<p>``, ``<h2>``).
    :param lang: Language code for keyword lookup.
    :return: ``dict`` with keys ``type`` and ``text`` or ``None`` if empty.
    """
    keywords = LANG_KEYWORDS.get(lang.upper(), LANG_KEYWORDS["EN"])
    article_kw = keywords["article"]
    annex_kw = keywords["annex"]
    chapter_kw = keywords["chapter"]

    text = normalize_text(tag.get_text(" ", strip=True))
    if not text:
        return None

    if re.match(rf"^{annex_kw}\s+[IVXLCDM]+\b", text, re.I):
        return {"type": "annex_title", "text": text}

    if re.match(rf"^{chapter_kw}\s+[IVXLCDM]+\b", text, re.I):
        return {"type": "chapter_title", "text": text}

    # Section titles may use Roman or Arabic numerals, e.g. "SECTION 1" or "SECTION I"
    if re.match(rf"^{keywords.get('section','SECTION')}\s+([IVXLCDM]+|\d+)\b", text, re.I):
        return {"type": "section_title", "text": text}

    if re.match(rf"^{article_kw}\s+\d+", text, re.I):
        return {"type": "article_title", "text": text}

    if tag.name == "table":
        return {"type": "table", "text": text}

    return {"type": "paragraph", "text": text}


# ============================================================
# Numbering detection STAYS
# ============================================================
def detect_numbering(text: str) -> Tuple[Optional[str], Optional[str], str]:
    """Detect numbered provisions in the text and classify hierarchy level.

    Recognises numbering patterns such as ``1.``, ``(1)``, ``(a)``,
    Roman numerals and dotted numeric hierarchies. Returns a tuple of
    ``(level, marker, body)`` where ``level`` is one of
    ``subsection``, ``section`, ``paragraph``, ``point``, ``subpoint``
    or ``None`` when no numbering was detected.

    :param text: Text to inspect for numbering prefixes.
    :return: Tuple of ``(level, marker, body)``.
    """
    patterns = [
        (r"^(\d+\.\d+\.\d+)\s+(.+)", "subsection"),
        (r"^(\d+\.\d+)\s+(.+)", "subsection"),
        (r"^(\d+)\.\s+(.+)", "section"),
        (r"^\((\d+)\)\s*(.+)", "paragraph"),
        (r"^\(([a-z])\)\s*(.+)", "point"),
        (r"^\(([ivxlcdm]+)\)\s*(.+)", "subpoint"),
    ]
    for pat, level in patterns:
        m = re.match(pat, text, re.I)
        if m:
            return level, m.group(1), m.group(2)
    return None, None, text


# STAYS
def make_provenance(html_text: str, tag: Tag, parser: str, parser_version: str, html_file: Path) -> Dict:
    """Create a small provenance dict for a parsed tag.

    Attempts to locate the tag's HTML within the raw document to provide
    start/end offsets and includes a hash of the raw document for
    integrity checks.

    :param html_text: full HTML document as a string
    :param tag: BeautifulSoup Tag that produced the provision
    :param parser: parser identifier (e.g. 'mdr_parser.parse_mdr')
    :param parser_version: version string for the parser
    :param html_file: source file Path
    :return: dict with provenance metadata
    """
    snippet = str(tag)
    start = html_text.find(snippet)
    end = start + len(snippet) if start != -1 else None
    raw_hash = hashlib.sha256(html_text.encode("utf-8")).hexdigest()

    prov = {
        "parser": parser,
        "parser_version": parser_version,
        "source_path": str(html_file),
        "raw_hash": raw_hash,
        "html_start": start,
        "html_end": end,
    }
    return prov

# STAYS
def extract_obligations(text: str, lang: str) -> List[Dict]:
    """Extract simple obligation structures from provision text.

    This is a light-weight extractor intended to produce a small
    structured representation agents can use. It is conservative:
    - If the provision is identified as a requirement, create an obligation
      entry with detected roles (actors) and the short action text.
    - Modality is derived from requirement classification.
    """
    obligations: List[Dict] = []
    if not text:
        return obligations

    # Use existing detectors where available
    req = is_requirement_text(text, lang) if 'is_requirement_text' in globals() else False
    req_type = classify_requirement_type(text, lang) if 'classify_requirement_type' in globals() else "other"

    # detect roles by calling the registered role detector (if any)
    roles = ROLE_DETECTOR(text, lang)

    if req:
        action = text.strip()
        # make short action (first sentence or 120 chars)
        first_sentence = action.split('.')
        short = (first_sentence[0] if first_sentence else action)[:240]
        obligations.append({
            "actors": roles or [],
            "action": short,
            "modality": req_type,
            "timing": None,
        })

    return obligations
