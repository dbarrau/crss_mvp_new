"""Shared constants, lookup mappings and lightweight detection helpers.

This module contains the configuration values (limits, regex patterns, CELEX
maps) that are referenced by multiple other submodules in the ``application``
package.  No Mistral / retriever I/O happens here.
"""
from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv

from domain.legislation_catalog import LEGISLATION as _LEGISLATION
from domain.mdcg_catalog import MDCG_DOCUMENTS as _MDCG_DOCS

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# ---------------------------------------------------------------------------
# Context-assembly limits
# ---------------------------------------------------------------------------

# Truncate rolled-up article/annex body to prevent definition-heavy articles
# (e.g. Article 2 with 65 definitions) from flooding the context window and
# crowding out actually-relevant children.
_BODY_LIMIT = 4000

# Maximum number of definition terms to inject into the context.
_MAX_DEFINITIONS = 5
_MAX_RELATED_DEFINITIONS = 15
_RELATED_DEFINITION_SCAN_LIMIT = 10

# ---------------------------------------------------------------------------
# Regulation name → CELEX lookup for multi-regulation retrieval.
# Derived from domain/legislation_catalog.py — add new legislation there.
# ---------------------------------------------------------------------------

_REG_NAME_TO_CELEX: dict[str, str] = {
    meta["name"]: celex for celex, meta in _LEGISLATION.items()
}

# Curated keyword patterns for detecting which legislation a question targets.
# Keys must match the "name" field in domain/legislation_catalog.py.
_LEGISLATION_PATTERNS: dict[str, list[str]] = {
    "EU AI Act": [
        "ai act", "2024/1689", "eu ai",
        "high-risk ai", "ai system", "artificial intelligence",
    ],
    "MDR 2017/745": [
        "mdr", "2017/745", "medical device regulation",
        "class i ", "class iia", "class iib", "class iii",
    ],
    "IVDR 2017/746": [
        "ivdr", "2017/746", "in vitro",
        "class a ", "class b ", "class c ", "class d ",
    ],
    "General Data Protection Regulation (GDPR) 2016/679": [
        "gdpr", "2016/679", "general data protection",
        "data protection regulation", "data subject", "personal data",
        "data controller", "data processor",
    ],
}

# Validate that every curated legislation pattern key exists in the catalog.
_unknown = set(_LEGISLATION_PATTERNS) - set(_REG_NAME_TO_CELEX)
if _unknown:
    raise KeyError(
        f"_LEGISLATION_PATTERNS has keys not in legislation catalog: {_unknown}"
    )

# Regulation name patterns for detecting which regulations a question targets.
# Starts with the curated legislation patterns; MDCG entries are added below
# by _build_mdcg_mappings().
_REG_PATTERNS: dict[str, list[str]] = dict(_LEGISLATION_PATTERNS)

# Extra keyword patterns for MDCG documents that benefit from implicit topic
# detection (beyond their ID).  Keyed by catalog ID (e.g. "MDCG_2020_3").
# Validate keys against the MDCG catalog.
_MDCG_EXTRA_PATTERNS: dict[str, list[str]] = {
    "MDCG_2020_3": [
        "significant changes", "significant change", "article 120",
    ],
    "MDCG_2019_11": [
        "software qualification", "software classification", "mdsw",
        "software update", "software change", "algorithm",
    ],
    "MDCG_2025_6": [
        "ai act interplay", "mdr ai act", "aia mdr", "dual regulation",
        "mdr ivdr ai", "medical device ai act",
    ],
}

_unknown_mdcg = set(_MDCG_EXTRA_PATTERNS) - set(_MDCG_DOCS)
if _unknown_mdcg:
    raise KeyError(
        f"_MDCG_EXTRA_PATTERNS has keys not in MDCG catalog: {_unknown_mdcg}"
    )


def _build_mdcg_mappings() -> None:
    """Auto-register every MDCG catalog entry in the agent lookup dicts."""
    for celex_key in _MDCG_DOCS:
        # Derive a human-readable name from the catalog key:
        #   MDCG_2020_3  → "MDCG 2020-3"
        #   MDCG_2025_6  → "MDCG 2025-6"
        parts = celex_key.split("_")        # ["MDCG", "2020", "3"]
        human_name = f"{parts[0]} {parts[1]}-{'_'.join(parts[2:])}"

        _REG_NAME_TO_CELEX[human_name] = celex_key

        # Base patterns: the dash and slash forms of the identifier
        dash_form = human_name.lower()                     # "mdcg 2020-3"
        slash_form = dash_form.replace("-", "/")           # "mdcg 2020/3"
        patterns: list[str] = [dash_form, slash_form]

        # Merge any manually-curated extra keywords
        extras = _MDCG_EXTRA_PATTERNS.get(celex_key, [])
        patterns.extend(extras)

        _REG_PATTERNS[human_name] = patterns


_build_mdcg_mappings()

# ---------------------------------------------------------------------------
# Provision reference regexes
# ---------------------------------------------------------------------------

# Regex for detecting explicit provision references in a question.
# Matches "Annex I", "Annex XIV", "Article 5", "Article 26a", "Recital 47".
_PROVISION_REF_RE = re.compile(
    r"\b(annex\s+[IVX]{1,5}"
    r"|article\s+\d{1,3}[a-z]?(?:\(\d+\))?"  # catches Article 26(3)
    r"|recital\s+\d{1,4})\b",
    re.IGNORECASE,
)

# Regex for inline provision pointers found inside retrieved provision text.
# Matches references like "Annex VII", "Article 17", "Annex IV", "Section 2",
# "Chapter III" that appear inside the body or children of retrieved provisions.
_INLINE_REF_RE = re.compile(
    r"\b(Annex(?:es)?\s+[IVX]{1,5}"
    r"|Article\s+\d{1,3}[a-z]?"
    r"|Recital\s+\d{1,4}"
    r"|Section\s+\d{1,3}"
    r"|Chapter\s+[IVX]{1,5})\b",
)

# ---------------------------------------------------------------------------
# Lightweight detection helpers (no LLM, no retriever)
# ---------------------------------------------------------------------------


def _detect_mentioned_regulations(question: str) -> set[str]:
    """Return regulation names mentioned in the question."""
    q_lower = question.lower()
    found: set[str] = set()
    for reg_name, patterns in _REG_PATTERNS.items():
        if any(p in q_lower for p in patterns):
            found.add(reg_name)
    return found


# ---------------------------------------------------------------------------
# Obligation master articles — authoritative statutory checklists per actor
#
# Values are lists so multiple anchor articles can be force-retrieved for a
# single actor (e.g. AI Act providers span two separate obligation regimes:
# Article 16 covers High-Risk AI systems, Article 53 covers GPAI models).
# Force-retrieving all anchors for obligation-breadth questions gives the LLM
# a complete statutory skeleton across every tier, not just the densest one.
# ---------------------------------------------------------------------------

_OBLIGATION_MASTER_ARTICLES: dict[tuple[str, str], list[str]] = {
    # AI Act providers span two obligation regimes: High-Risk (Art. 16) and
    # GPAI model providers (Art. 53).  Both are needed for full coverage.
    ("provider", "32024R1689"):                     ["Article 16", "Article 53"],
    ("deployer", "32024R1689"):                     ["Article 26"],
    ("importer", "32024R1689"):                     ["Article 23"],
    ("distributor", "32024R1689"):                  ["Article 24"],
    ("manufacturer", "32017R0745"):                 ["Article 10"],
    ("manufacturer", "32017R0746"):                 ["Article 10"],
    ("authorised representative", "32017R0745"):    ["Article 11"],
    ("authorised representative", "32017R0746"):    ["Article 11"],
    ("importer", "32017R0745"):                     ["Article 13"],
    ("importer", "32017R0746"):                     ["Article 13"],
    ("distributor", "32017R0745"):                  ["Article 14"],
    ("distributor", "32017R0746"):                  ["Article 14"],
}


def _extract_provision_refs(question: str) -> list[str]:
    """Extract and normalise explicit provision references from *question*.

    Returns references like ``['Annex I', 'Article 26']`` ready for direct
    lookup.  Roman numerals are uppercased; article/recital numbers are
    preserved as-is.

    Examples
    --------
    >>> _extract_provision_refs("What does Annex I of the EU AI Act contain?")
    ['Annex I']
    >>> _extract_provision_refs("What are the obligations under Article 26?")
    ['Article 26']
    """
    seen: set[str] = set()
    result: list[str] = []
    for m in _PROVISION_REF_RE.finditer(question):
        parts = m.group(0).strip().split(None, 1)  # ["annex", "i"] or ["article", "5"]
        if len(parts) == 2:
            normalized = parts[0].capitalize() + " " + parts[1].upper()
        else:
            normalized = parts[0].capitalize()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
