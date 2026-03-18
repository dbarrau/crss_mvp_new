"""Definition extraction for EU legislative provisions.

Extracts :DefinedTerm nodes and DEFINED_BY edges from definition articles
(e.g., EU AI Act Article 3, MDR Article 2, IVDR Article 2).

A "definition point" matches when its text starts with a term in single
quotes followed by 'means':

    'provider' means a natural or legal person …

Curly/typographic single quotes (\u2018, \u2019) are also supported.

Extracted DefinedTerm properties
---------------------------------
id                  – ``{celex}_defterm_{term_normalized}``
term                – raw term between the quotes, whitespace-stripped
term_normalized     – lowercased, consecutive whitespace → single underscore
category            – inferred from definition body keyword matching
celex               – source regulation CELEX identifier
regulation          – human-readable regulation name
source_provision_id – id of the :Point/:AnnexPoint provision containing
                      the full definition text
"""
from __future__ import annotations

import re
from typing import Any

# Matches: opening quote (straight or curly), the term, closing quote,
# optional whitespace, then literal 'means'
_TERM_RE = re.compile(
    r"^['\u2018\u2019\u201a\u201b]([^'\u2018\u2019\u201a\u201b]+)"
    r"['\u2018\u2019\u201a\u201b]\s+means\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------
# Only the first clause (text before the first ';' or '.') is checked for actors.
# Body classification uses the term name itself — if the term is called an
# "authority", "body", "office", or "board", it is an institutional body.
# This avoids false positives where an unrelated definition merely *mentions*
# a competent authority in its text.
#
# Two meaningful categories for agent use:
#   actor — economic operators (natural/legal persons) bearing obligations/rights
#   body  — institutional actors (authorities, boards, offices)
# Everything else is "other"; exact-term lookup via find_by_term() works for all.

_ACTOR_SIGNALS = (
    "natural or legal person",
    "public authority",
    "agency or other body",
)

# Keywords present in the term *name* that identify institutional bodies.
_BODY_TERM_KEYWORDS = ("authority", "body", "office", "board")


def _classify_category(term: str, definition_body: str) -> str:
    """Return the semantic category for a defined term.

    Parameters
    ----------
    term:
        The raw term text (between quotes), e.g. ``"notifying authority"``.
    definition_body:
        The provision text *after* the word ``means``.
    """
    first_clause = re.split(r"[;.]", definition_body, maxsplit=1)[0]
    first_clause = first_clause.replace("\xa0", " ").lower()

    if any(sig in first_clause for sig in _ACTOR_SIGNALS):
        return "actor"

    if any(kw in term.lower() for kw in _BODY_TERM_KEYWORDS):
        return "body"

    return "other"


def _normalize_term(term: str) -> str:
    """Return a stable, filesystem-safe identifier fragment for a term.

    Lowercases the term and replaces any run of whitespace with a single
    underscore, e.g. 'AI system' → 'ai_system'.
    """
    return re.sub(r"\s+", "_", term.strip().lower())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_defined_terms(
    provisions: list[dict[str, Any]],
    celex: str,
    regulation: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan a flat provisions list for definition patterns.

    Any provision whose ``text`` field starts with a quoted term followed by
    *means* is treated as a defining provision.  One :DefinedTerm node and
    one DEFINED_BY relation is produced per unique ``(celex, term_normalized)``
    pair.

    Parameters
    ----------
    provisions:
        Flat provisions list as produced by the structural parsers.
    celex:
        CELEX identifier of the source regulation (e.g. ``"32024R1689"``).
    regulation:
        Human-readable regulation name (e.g. ``"EU AI Act"``).  Used as the
        ``regulation`` property on each DefinedTerm node.

    Returns
    -------
    (defined_terms, relations)
        defined_terms – list of DefinedTerm node dicts ready for Neo4j
        relations     – list of DEFINED_BY relation dicts using the same
                        schema as the ``relations`` array in ``parsed.json``
    """
    defined_terms: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    seen: set[str] = set()

    for prov in provisions:
        text = (prov.get("text") or "").strip()
        if not text:
            continue

        m = _TERM_RE.match(text)
        if not m:
            continue

        raw_term = m.group(1)
        term_normalized = _normalize_term(raw_term)

        # Deduplicate within the same regulation (first occurrence wins)
        dedup_key = f"{celex}:{term_normalized}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Definition body starts immediately after the 'means' keyword
        definition_body = text[m.end():].strip()
        category = _classify_category(raw_term, definition_body)

        node_id = f"{celex}_defterm_{term_normalized}"

        defined_terms.append(
            {
                "id":                  node_id,
                "term":                raw_term.strip(),
                "term_normalized":     term_normalized,
                "category":            category,
                "celex":               celex,
                "regulation":          regulation,
                "source_provision_id": prov["id"],
            }
        )

        relations.append(
            {
                "source":     node_id,
                "type":       "DEFINED_BY",
                "target":     prov["id"],
                "properties": {},
            }
        )

    return defined_terms, relations
