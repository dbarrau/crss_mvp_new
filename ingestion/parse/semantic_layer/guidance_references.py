"""Extract regulation cross-references from MDCG guidance provision text.

MDCG guidance documents cite EU regulations using three patterns:

1. **Full form**: ``Regulation (EU) 2017/745``
2. **Short name**: ``MDR``, ``IVDR``
3. **Qualified**: ``Article 120(3) of … Regulation (EU) 2017/745``
   or ``Annex VIII of Regulation (EU) 2017/745``

This module scans each provision's text, identifies regulation mentions, and
associates nearby ``Article`` / ``Annex`` references with the regulation to
produce ``CITES_EXTERNAL`` relations.  The crosslinker then resolves these to
concrete ``CITES`` edges between graph nodes.

Public API
----------
- :func:`extract_guidance_relations` — process a full provisions list.
"""
from __future__ import annotations

import re
from typing import Any

from domain.legislation_catalog import LEGISLATION

# ---------------------------------------------------------------------------
# Short-name → regulation number mapping
# ---------------------------------------------------------------------------
_SHORT_NAMES: dict[str, str] = {
    "MDR":  "2017/745",
    "IVDR": "2017/746",
}

# Regulation numbers that we have loaded (can be resolved by crosslinker)
_KNOWN_NUMBERS: frozenset[str] = frozenset(
    meta["number"] for meta in LEGISLATION.values()
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "Regulation (EU) YYYY/NNN"
_FULL_REG_RE = re.compile(
    r"Regulation\s+\(EU\)\s+(?P<number>\d{4}/\d+)",
    re.IGNORECASE,
)

# Short-name references: standalone "MDR" or "IVDR"
_SHORT_REG_RE = re.compile(r"\b(?P<short>MDR|IVDR)\b")

# "Article N(P)(sub), point (x)(y)" — captures article + optional qualifiers
_ARTICLE_RE = re.compile(
    r"Articles?\s+(?P<article>\d+)"
    r"(?:\((?P<para>\d+[a-z]?)\))?"
    r"(?:\((?P<sub>\d+)\))?"
    r"(?:,?\s*point\s+\((?P<point>[a-z0-9]+)\)"
    r"(?:\((?P<subpoint>[a-z0-9]+)\))?"
    r")?",
    re.IGNORECASE,
)

# "Annex VIII" / "Annex XVI" (Roman numerals)
_ANNEX_RE = re.compile(
    r"Annex\s+(?P<annex>[IVX]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_guidance_relations(
    provisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract regulation cross-references from guidance provisions.

    Returns a list of ``CITES_EXTERNAL`` relation dicts compatible with
    the ``parsed.json`` schema and resolvable by the crosslinker.
    """
    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()          # (source, number, ref_text)

    for prov in provisions:
        prov_id = prov["id"]
        text = prov.get("text", "") or ""
        if not text:
            continue

        # Collect all regulation mentions (position + number)
        reg_mentions = _find_regulation_mentions(text)
        if not reg_mentions:
            continue

        # Collect article / annex references
        article_matches = list(_ARTICLE_RE.finditer(text))
        annex_matches = list(_ANNEX_RE.finditer(text))

        # Associate each article/annex with the nearest regulation mention
        for am in article_matches:
            reg = _nearest_regulation(am.start(), reg_mentions)
            if reg is None:
                continue
            number = reg["number"]
            if number not in _KNOWN_NUMBERS:
                continue
            ref_text = _format_article_ref(am)
            key = (prov_id, number, ref_text)
            if key in seen:
                continue
            seen.add(key)
            relations.append(_make_cites_ext(prov_id, number, ref_text))

        for am in annex_matches:
            reg = _nearest_regulation(am.start(), reg_mentions)
            if reg is None:
                continue
            number = reg["number"]
            if number not in _KNOWN_NUMBERS:
                continue
            ref_text = f"Annex {am.group('annex')}"
            key = (prov_id, number, ref_text)
            if key in seen:
                continue
            seen.add(key)
            relations.append(_make_cites_ext(prov_id, number, ref_text))

        # Document-level citation for any regulation mentioned without
        # an associated article/annex
        art_annex_regs = set()
        for am in article_matches:
            r = _nearest_regulation(am.start(), reg_mentions)
            if r:
                art_annex_regs.add(r["number"])
        for am in annex_matches:
            r = _nearest_regulation(am.start(), reg_mentions)
            if r:
                art_annex_regs.add(r["number"])

        for reg in reg_mentions:
            number = reg["number"]
            if number not in _KNOWN_NUMBERS:
                continue
            if number in art_annex_regs:
                continue          # already have provision-level refs
            ref_text = f"Regulation (EU) {number}"
            key = (prov_id, number, ref_text)
            if key in seen:
                continue
            seen.add(key)
            relations.append(_make_cites_ext(prov_id, number, ref_text))

    return relations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_regulation_mentions(text: str) -> list[dict[str, Any]]:
    """Return all regulation mentions (full and short-name) with positions."""
    mentions: list[dict[str, Any]] = []
    for m in _FULL_REG_RE.finditer(text):
        mentions.append({
            "number": m.group("number"),
            "start":  m.start(),
            "end":    m.end(),
        })
    for m in _SHORT_REG_RE.finditer(text):
        mentions.append({
            "number": _SHORT_NAMES[m.group("short")],
            "start":  m.start(),
            "end":    m.end(),
        })
    return mentions


def _nearest_regulation(
    pos: int, mentions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the regulation mention closest to *pos* in the text."""
    if not mentions:
        return None
    return min(mentions, key=lambda m: min(abs(pos - m["start"]), abs(pos - m["end"])))


def _format_article_ref(m: re.Match) -> str:
    """Build a human-readable ref_text from an article regex match."""
    parts = [f"Article {m.group('article')}"]
    if m.group("para"):
        parts.append(f"({m.group('para')})")
    if m.group("point"):
        parts.append(f", point ({m.group('point')})")
    if m.group("subpoint"):
        parts[-1] += f"({m.group('subpoint')})"
    return "".join(parts)


def _make_cites_ext(
    source_id: str, number: str, ref_text: str,
) -> dict[str, Any]:
    """Build a CITES_EXTERNAL relation dict for the crosslinker."""
    ext_id = f"ext_regulation_eu_{number.replace('/', '_')}"
    return {
        "source": source_id,
        "type": "CITES_EXTERNAL",
        "target": ext_id,
        "properties": {
            "ref_text": ref_text,
            "number": number,
        },
    }
