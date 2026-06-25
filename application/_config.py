"""Shared constants, lookup mappings and lightweight detection helpers.

This module contains the configuration values (limits, regex patterns, CELEX
maps) that are referenced by multiple other submodules in the ``application``
package.  No Mistral / retriever I/O happens here.
"""
from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv

from domain.legislation_catalog import (
    LEGISLATION as _LEGISLATION,
    AI_ACT_CELEX,
    MDR_CELEX,
    GDPR_CELEX,
)
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
    "Commission Implementing Regulation (EU) 2026/977": [
        "2026/977", "implementing regulation (eu) 2026/977",
        "notified body fee", "notified body fees", "notified body quotation",
        "conformity assessment quotation",
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
        "significant changes", "significant change",
        # "article 120" intentionally omitted: it caused false multi-reg detection
        # when a question explicitly said "MDR Article 120", inflating mentioned_regs
        # to {MDR, MDCG_2020_3} and routing to cross_regulation instead of provision_lookup.
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


def _build_core_mappings() -> None:
    """Safety net: ensure every core (non-MDCG) catalog entry is detectable.

    ``_LEGISLATION_PATTERNS`` is a hand-curated subset.  If a regulation is added
    to ``domain/legislation_catalog.py`` without a matching pattern entry here,
    ``_detect_mentioned_regulations`` would silently never fire for it — the
    retrieval scope would exclude it even when the question names it explicitly
    (this is exactly what happened to CIR 2026/977).  To prevent that class of
    gap, register each catalog entry's bare ``number`` (e.g. "2026/977") as a
    minimal fallback detection pattern when no curated entry exists.
    """
    for _celex_key, meta in _LEGISLATION.items():
        name = meta["name"]
        number = meta.get("number")
        if name in _REG_PATTERNS:
            continue  # already curated above
        if number:
            _REG_PATTERNS[name] = [number]


_build_mdcg_mappings()
_build_core_mappings()

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
# Implicit provision reference inference
#
# Maps well-known regulatory topic keywords to their canonical provision when
# the user does not name the article explicitly.  Each entry is a 3-tuple:
#   (compiled_pattern, celex_required, article_ref)
# The pattern is matched against the question; the ref is only added when the
# required CELEX is already in scope (target_celexes) to avoid false positives
# on generic vocabulary shared across regulations.
# ---------------------------------------------------------------------------

_IMPLICIT_PROVISION_REFS: list[tuple[re.Pattern, str, str]] = [
    # "lawful basis / lawful bases" → GDPR Article 6 (the six-ground enumeration)
    (re.compile(r"\blawful\s+bas(?:is|es)\b", re.I), GDPR_CELEX, "Article 6"),
    # "prohibited / prohibition" in AI Act context → Article 5 (sole prohibition article)
    (re.compile(r"\bprohibit(?:ed|ion|ions|s)?\b", re.I), AI_ACT_CELEX, "Article 5"),
    # "technical documentation" in MDR context → Annex II (primary tech-doc annex)
    (re.compile(r"\btechnical\s+documentation\b", re.I), MDR_CELEX, "Annex II"),
]


def _extract_implicit_provision_refs(
    question: str,
    *,
    target_celexes: set[str] | None,
) -> list[str]:
    """Return implicit provision refs inferred from well-known topic keywords.

    Only fires when the required regulation is already in scope
    (i.e. *target_celexes* is not None and contains the entry's CELEX).
    Returns an empty list when no regulation is in scope so callers never need
    to guard against None.
    """
    if target_celexes is None:
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for pattern, celex, ref in _IMPLICIT_PROVISION_REFS:
        if celex in target_celexes and pattern.search(question) and ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


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
