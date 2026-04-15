"""Extract regulation cross-references from MDCG guidance provision text.

MDCG guidance documents cite EU regulations using several patterns:

1. **Full form**: ``Regulation (EU) 2017/745``
2. **Short name**: ``MDR``, ``IVDR``, ``AIA``
3. **Multi-word short name**: ``AI Act``
4. **Qualified article**: ``Article 6(1) AIA``, ``Art. 3 (29) AIA``
5. **Qualified annex**: ``Annex VI AIA``, ``Annex XIV MDR``
6. **Compound short name**: ``MDR/IVDR`` (expands to both)
7. **Proximity-based**: ``Article 120(3) of … Regulation (EU) 2017/745``

Detection priority: qualified patterns (4-6) first, then proximity (7).
This avoids misattribution when multiple regulations appear in the same
paragraph.

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
    "MDR":    "2017/745",
    "IVDR":   "2017/746",
    "AIA":    "2024/1689",
    "AI Act": "2024/1689",
}

# Build regex alternation from short names, longest first to avoid
# partial matches (e.g. "AI Act" before "AI").
_SHORT_ALTS = "|".join(
    re.escape(k) for k in sorted(_SHORT_NAMES, key=len, reverse=True)
)

# Lower-cased set of known short names — used by the false-positive filter.
_SHORT_NAMES_LC: frozenset[str] = frozenset(k.lower() for k in _SHORT_NAMES)

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

# Short-name references (standalone): MDR, IVDR, AIA, AI Act
_SHORT_REG_RE = re.compile(
    rf"\b(?P<short>{_SHORT_ALTS})\b",
)

# Compound short names: "MDR/IVDR"
_COMPOUND_RE = re.compile(
    r"\b(?P<first>MDR|IVDR|AIA)(?:/(?P<second>MDR|IVDR|AIA))\b",
)

# ---------------------------------------------------------------------------
# Article / Annex base patterns (accept "Art." and "Articles?")
# ---------------------------------------------------------------------------
_ART_PREFIX = r"(?:Articles?|Art\.)\s*"

# "Article N(P)(sub), point (x)(y)" — unqualified (no trailing short name)
_ARTICLE_RE = re.compile(
    _ART_PREFIX
    + r"(?P<article>\d+)"
    r"(?:\s*\((?P<para>\d+[a-z]?)\))?"
    r"(?:\s*\((?P<sub>\d+)\))?"
    r"(?:,?\s*point\s+\((?P<point>[a-z0-9]+)\)"
    r"(?:\((?P<subpoint>[a-z0-9]+)\))?"
    r")?",
    re.IGNORECASE,
)

# "Annex VIII" / "Annex XVI" (Roman numerals) — unqualified
_ANNEX_RE = re.compile(
    r"Annex\s+(?P<annex>[IVX]+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Qualified patterns: Article/Annex immediately followed by a short name.
# These are high-precision — the regulation is unambiguous.
# ---------------------------------------------------------------------------

# "Article 6(1) AIA", "Art. 3 (29) AIA", "Article 6(1) of the MDR"
_QUALIFIED_ARTICLE_RE = re.compile(
    _ART_PREFIX
    + r"(?P<article>\d+)"
    r"(?:\s*\((?P<para>\d+[a-z]?)\))?"
    r"(?:\s*\((?P<sub>\d+[a-z]?|[a-z])\))?"
    r"(?:,?\s*point\s+\((?P<point>[a-z0-9]+)\)"
    r"(?:\((?P<subpoint>[a-z0-9]+)\))?"
    r")?"
    r"\s+(?:of\s+(?:the\s+)?)?(?P<short>" + _SHORT_ALTS + r")\b",
    re.IGNORECASE,
)

# "Annex VI AIA", "Annex XIV MDR", "Annex I of the MDR"
_QUALIFIED_ANNEX_RE = re.compile(
    r"Annex\s+(?P<annex>[IVX]+)"
    r"\s+(?:of\s+(?:the\s+)?)?(?P<short>" + _SHORT_ALTS + r")\b",
    re.IGNORECASE,
)

# Detect articles/annexes qualified by an unknown regulation name ("of the EHDS")
# so the proximity fallback can skip them.
_OF_UNKNOWN_RE = re.compile(
    r"\s+of\s+(?:the\s+)?(?P<name>[A-Z][A-Za-z]+)\b",
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

        # Track character spans consumed by qualified matches so the
        # proximity fallback doesn't double-count them.
        qualified_spans: list[tuple[int, int]] = []

        # --- Phase 1: qualified article patterns (highest precision) ---
        for m in _QUALIFIED_ARTICLE_RE.finditer(text):
            numbers = _resolve_short(m.group("short"))
            ref_text = _format_article_ref(m)
            for number in numbers:
                if number not in _KNOWN_NUMBERS:
                    continue
                key = (prov_id, number, ref_text)
                if key not in seen:
                    seen.add(key)
                    relations.append(_make_cites_ext(prov_id, number, ref_text))
            qualified_spans.append((m.start(), m.end()))

        # --- Phase 2: qualified annex patterns ---
        for m in _QUALIFIED_ANNEX_RE.finditer(text):
            numbers = _resolve_short(m.group("short"))
            ref_text = f"Annex {m.group('annex')}"
            for number in numbers:
                if number not in _KNOWN_NUMBERS:
                    continue
                key = (prov_id, number, ref_text)
                if key not in seen:
                    seen.add(key)
                    relations.append(_make_cites_ext(prov_id, number, ref_text))
            qualified_spans.append((m.start(), m.end()))

        # --- Phase 3: proximity-based fallback for remaining refs ---
        reg_mentions = _find_regulation_mentions(text)
        if not reg_mentions:
            continue

        article_matches = [
            am for am in _ARTICLE_RE.finditer(text)
            if not _overlaps(am.start(), am.end(), qualified_spans)
            and not _qualified_by_unknown(text, am.end())
        ]
        annex_matches = [
            am for am in _ANNEX_RE.finditer(text)
            if not _overlaps(am.start(), am.end(), qualified_spans)
            and not _qualified_by_unknown(text, am.end())
        ]

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

        # --- Phase 4: document-level citation for regulations without
        # any provision-level article/annex ref ---
        regs_with_provision_refs: set[str] = set()
        for am in article_matches:
            r = _nearest_regulation(am.start(), reg_mentions)
            if r:
                regs_with_provision_refs.add(r["number"])
        for am in annex_matches:
            r = _nearest_regulation(am.start(), reg_mentions)
            if r:
                regs_with_provision_refs.add(r["number"])
        # Also count qualified matches
        for m in _QUALIFIED_ARTICLE_RE.finditer(text):
            for n in _resolve_short(m.group("short")):
                regs_with_provision_refs.add(n)
        for m in _QUALIFIED_ANNEX_RE.finditer(text):
            for n in _resolve_short(m.group("short")):
                regs_with_provision_refs.add(n)

        for reg in reg_mentions:
            number = reg["number"]
            if number not in _KNOWN_NUMBERS:
                continue
            if number in regs_with_provision_refs:
                continue
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

def _resolve_short(short: str) -> list[str]:
    """Resolve a short name (or compound like 'MDR/IVDR') to regulation numbers."""
    if "/" in short:
        return [_SHORT_NAMES[p] for p in short.split("/") if p in _SHORT_NAMES]
    return [_SHORT_NAMES[short]] if short in _SHORT_NAMES else []


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """Check whether *[start, end)* overlaps any span in *spans*."""
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def _qualified_by_unknown(text: str, end_pos: int) -> bool:
    """True if the text right after *end_pos* says 'of (the)? <ABBR>'
    where <ABBR> is an all-caps abbreviation (e.g. EHDS, GDPR) that is
    NOT a known regulation short name.  This filters false positives like
    'Article 2(2)(k) of the EHDS' without blocking legitimate patterns
    like 'Article 120(3) of the Medical Device Regulation …'."""
    tail = text[end_pos:]
    # Skip optional extra parenthesised sub-parts the article regex
    # didn't consume, e.g. "(k)" after Article 2(2).
    m = re.match(r"(?:\([a-z0-9]+\))*\s+of\s+(?:the\s+)?([A-Z]{2,})\b", tail)
    if m is None:
        return False
    abbr = m.group(1)
    return abbr.lower() not in _SHORT_NAMES_LC


def _find_regulation_mentions(text: str) -> list[dict[str, Any]]:
    """Return all regulation mentions (full, short, and compound) with positions."""
    mentions: list[dict[str, Any]] = []
    for m in _FULL_REG_RE.finditer(text):
        mentions.append({
            "number": m.group("number"),
            "start":  m.start(),
            "end":    m.end(),
        })
    # Compound forms like "MDR/IVDR" — expand to two mentions at same position
    compound_spans: list[tuple[int, int]] = []
    for m in _COMPOUND_RE.finditer(text):
        first_num = _SHORT_NAMES.get(m.group("first"))
        second_num = _SHORT_NAMES.get(m.group("second"))
        if first_num:
            mentions.append({"number": first_num, "start": m.start(), "end": m.end()})
        if second_num:
            mentions.append({"number": second_num, "start": m.start(), "end": m.end()})
        compound_spans.append((m.start(), m.end()))
    # Standalone short names (skip positions already consumed by compounds)
    for m in _SHORT_REG_RE.finditer(text):
        if _overlaps(m.start(), m.end(), compound_spans):
            continue
        number = _SHORT_NAMES.get(m.group("short"))
        if number:
            mentions.append({"number": number, "start": m.start(), "end": m.end()})
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
