# ingestion/parse/base/requirement_patterns.py
"""
Enhanced requirement patterns for EU regulations (MDR, IVDR, EU AI Act)
Supports EN, DE, FR with conflict resolution, punctuation handling, and better definitions.
"""

import re

REQ_PATTERNS = {
    "EN": {
        "obligation": [
            r"\bshall\b(?=\s|:|,|\.|$)",
            r"\bmust\b(?=\s|:|,|\.|$)",
            r"\bis required to\b",
            r"\bis obliged to\b",
            r"\bhas to\b",
            r"\bshall ensure\b",
        ],
        "prohibition": [
            r"\bshall not\b",
            r"\bmay not\b",
            r"\bmust not\b",
            r"\bis prohibited\b",
            r"\bshall refrain from\b",
        ],
        "permission": [
            r"\bmay\b(?=\s|:|,|\.|$)",
            r"\bis permitted\b",
            r"\bis allowed to\b",
        ],
        "definition": [
            r"'[^']+'\s+means\b",
            r'"[^"]+"\s+means\b',
            r"\brefers to\b",
            r"\bdenotes\b",
        ],
    },
    "DE": {
        "obligation": [
            r"\bmuss\b",
            r"\bverpflichtet\b",
            r"\bhat sicherzustellen\b",
            r"\bist verpflichtet\b",
        ],
        "prohibition": [
            r"\bdarf nicht\b",
            r"\bist untersagt\b",
            r"\bverboten\b",
        ],
        "permission": [
            r"\bdarf\b",
            r"\bist erlaubt\b",
        ],
        "definition": [
            r"'[^']+'\s+bezeichnet\b",
            r'"[^"]+"\s+bezeichnet\b',
            r"\bbezeichnet\b",
            r"\bsteht für\b",
        ],
    },
    "FR": {
        "obligation": [
            r"\bdoit\b",
            r"\best tenu de\b",
            r"\best obligé de\b",
        ],
        "prohibition": [
            r"\bne doit pas\b",
            r"\best interdit\b",
        ],
        "permission": [
            r"\bpeut\b",
            r"\best autorisé à\b",
        ],
        "definition": [
            r"'[^']+'\s+signifie\b",
            r'"[^"]+"\s+signifie\b',
            r"\bsignifie\b",
            r"\bdésigne\b",
        ],
    },
}

# ============================================================
# Requirement detection functions
# ============================================================

def is_requirement_text(text: str, lang: str) -> bool:
    """
    Returns True if the text contains any requirement keyword for the given language.
    """
    lang = lang.upper()
    patterns = REQ_PATTERNS.get(lang, REQ_PATTERNS["EN"])

    for p_list in patterns.values():
        for pat in p_list:
            if re.search(pat, text, re.I):
                return True
    return False


def classify_requirement_type(text: str, lang: str) -> str:
    """
    Classifies the text into one of: obligation, prohibition, permission, definition, or 'other'.
    Longer patterns are matched first to avoid conflicts (e.g., 'may not' vs 'may').
    """
    lang = lang.upper()
    patterns = REQ_PATTERNS.get(lang, REQ_PATTERNS["EN"])

    # Flatten patterns as (req_type, pattern) preserving order
    flat_patterns = []
    for req_type in ["prohibition", "obligation", "permission", "definition"]:
        flat_patterns.extend([(req_type, pat) for pat in patterns.get(req_type, [])])

    for req_type, pat in flat_patterns:
        if re.search(pat, text, re.I):
            return req_type
    return "other"
