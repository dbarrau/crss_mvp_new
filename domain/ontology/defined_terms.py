"""Domain knowledge for EU legislative definition extraction.

Contains the linguistic pattern and ontological signals used to identify
and classify defined terms in EU regulatory text, e.g.:

    'provider' means a natural or legal person …

Curly/typographic single quotes (\u2018, \u2019) are also supported.
"""
from __future__ import annotations

import re

# Matches: opening quote (straight or curly), the term, closing quote,
# optional whitespace, then literal 'means'.
TERM_PATTERN = re.compile(
    r"^['\u2018\u2019\u201a\u201b]([^'\u2018\u2019\u201a\u201b]+)"
    r"['\u2018\u2019\u201a\u201b]\s+means\b",
    re.IGNORECASE,
)

# Phrases in the first clause of a definition body that signal an "actor" —
# an economic operator (natural or legal person) bearing obligations/rights.
ACTOR_SIGNALS = (
    "natural or legal person",
    "public authority",
    "agency or other body",
)

# Keywords in the term *name* itself that identify institutional bodies
# (authorities, boards, offices).  Checked against the term, not the body,
# to avoid false positives where a definition merely mentions such a body.
BODY_TERM_KEYWORDS = ("authority", "body", "office", "board")


def classify_category(term: str, definition_body: str) -> str:
    """Return the semantic category for a defined term: 'actor', 'body', or 'other'.

    Parameters
    ----------
    term:
        The raw term text (between quotes), e.g. ``"notifying authority"``.
    definition_body:
        The provision text *after* the word ``means``.
    """
    first_clause = re.split(r"[;.]", definition_body, maxsplit=1)[0]
    first_clause = first_clause.replace("\xa0", " ").lower()

    if any(sig in first_clause for sig in ACTOR_SIGNALS):
        return "actor"

    if any(kw in term.lower() for kw in BODY_TERM_KEYWORDS):
        return "body"

    return "other"
