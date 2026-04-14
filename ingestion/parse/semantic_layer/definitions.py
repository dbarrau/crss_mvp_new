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

from domain.ontology.defined_terms import (
    DEFINITIONS_ARTICLES as _DEFINITIONS_ARTICLES,
    TERM_PATTERN as _TERM_RE,
    classify_category as _classify_category,
)


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

        # Determine whether the definition comes from the regulation's
        # official definitions article (formal) or from elsewhere (contextual).
        def_article = _DEFINITIONS_ARTICLES.get(celex, {})
        def_article_prefix = def_article.get("article_id", "")
        is_formal = bool(
            def_article_prefix
            and (
                prov["id"] == def_article_prefix
                or prov["id"].startswith(def_article_prefix + "_")
            )
        )

        defined_terms.append(
            {
                "id":                  node_id,
                "term":                raw_term.strip(),
                "term_normalized":     term_normalized,
                "category":            category,
                "definition_type":     "formal" if is_formal else "contextual",
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
