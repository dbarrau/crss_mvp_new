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
#
# This is a DIFFERENT axis from ``actor_roles.CANONICAL_ACTOR_ROLES`` (the named
# role registry) and ``provision_roles._ACTOR_SUBJECTS`` (the obligation-rule
# subject list): these are definition-body *signal phrases* used to classify a
# freshly parsed DefinedTerm as an actor, not a list of role names. It is
# intentionally not reconciled with the role registry.
ACTOR_SIGNALS = (
    "natural or legal person",
    "public authority",
    "agency or other body",
)

# Keywords in the term *name* itself that identify institutional bodies
# (authorities, boards, offices).  Checked against the term, not the body,
# to avoid false positives where a definition merely mentions such a body.
BODY_TERM_KEYWORDS = ("authority", "body", "office", "board")

# Maps CELEX IDs to their official definitions article.
# Used to distinguish formal (regulation-wide) definitions from contextual
# (scoped) ones that appear elsewhere in the text.
#
# Must stay in sync with ``DEFINITIONS_ARTICLE_IDS`` in
# ``domain/ontology/provision_roles.py`` (same four regulations). Omitting a
# regulation here silently marks all its definitions ``contextual`` even when
# they are the regulation's formal Article-N definitions — which excludes them
# from the formal-definition retrieval path (see ``application/_definitions.py``).
DEFINITIONS_ARTICLES: dict[str, dict[str, str]] = {
    "32017R0745": {"article_id": "32017R0745_art_2",  "display_ref": "Article 2"},
    "32017R0746": {"article_id": "32017R0746_art_2",  "display_ref": "Article 2"},
    "32024R1689": {"article_id": "32024R1689_art_3",  "display_ref": "Article 3"},
    "32016R0679": {"article_id": "32016R0679_art_4",  "display_ref": "Article 4"},  # GDPR
}


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
