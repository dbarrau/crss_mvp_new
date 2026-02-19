import re

REQ_PATTERNS = {
    "EN": {
        "obligation": [r"\bshall\b", r"\bmust\b", r"\bis required to\b"],
        "prohibition": [r"\bshall not\b", r"\bmay not\b"],
        "permission": [r"\bmay\b"],
        "definition": [r"\bmeans\b", r"\brefers to\b"],
    },
    "DE": {
        "obligation": [r"\bmuss\b", r"\bverpflichtet\b", r"\bhat sicherzustellen\b"],
        "prohibition": [r"\bdarf nicht\b"],
        "permission": [r"\bdarf\b"],
        "definition": [r"\bbezeichnet\b", r"\bist\b"],
    },
    "FR": {
        "obligation": [r"\bdoit\b", r"\best tenu de\b"],
        "prohibition": [r"\bne doit pas\b"],
        "permission": [r"\bpeut\b"],
        "definition": [r"\bsignifie\b"],
    },
}

# ============================================================
# Requirement detection
# ============================================================

def is_requirement_text(text: str, lang: str) -> bool:
    lang = lang.upper()
    patterns = REQ_PATTERNS.get(lang, REQ_PATTERNS["EN"])

    for p_list in patterns.values():
        for pat in p_list:
            if re.search(pat, text, re.I):
                return True
    return False


def classify_requirement_type(text: str, lang: str) -> str:
    lang = lang.upper()
    patterns = REQ_PATTERNS.get(lang, REQ_PATTERNS["EN"])

    for req_type, p_list in patterns.items():
        for pat in p_list:
            if re.search(pat, text, re.I):
                return req_type
    return "other"
