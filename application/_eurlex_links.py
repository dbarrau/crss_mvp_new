"""Deterministic EUR-Lex hyperlinks for provision references (render layer only).

Turns the plain-prose provision references the model writes ("Article 25(2) AI
Act", "Annex III", "Recital 81") into clickable EUR-Lex links so a reader can
jump straight to the actual provision.  No LLM, no graph change: this replaces
the bold-only pass (:func:`_bold_references`) for the user-facing answer, wrapping
the same references in a link *and* the existing bold.

A link needs two facts per reference:

* **anchor** — derived from the reference shape.  Verified against the real
  EUR-Lex HTML (both base and consolidated documents share the scheme):
  ``Article 25`` / ``Article 25(2)`` -> ``#art_25`` (article is the finest anchor
  grain — there are no per-paragraph anchors), ``Annex III`` -> ``#anx_III``,
  ``Recital 81`` -> ``#rct_81``.
* **CELEX** — which regulation.  The bold pass is CELEX-blind ("Article 6" of
  *what*?); we recover it from the regulation name the model wrote next to the
  reference ("... Article 6 **AI Act**"), disambiguated against the regulations
  actually in scope.  When the CELEX cannot be resolved unambiguously the
  reference is left **bold-only** — a missing link is harmless, a link to the
  wrong regulation is not.

Link target follows amendment status:

* MDR / IVDR / GDPR carry a ``source_celex`` (their official EUR-Lex
  consolidation) — the current law CRSS displays — so they link there and match
  exactly.
* The AI Act, CIR and Digital Omnibus have no official consolidation yet, so
  their references link to the base act.  For AI Act articles the Omnibus has
  *replaced* in part, the base ``#art_N`` still shows most of the article; the
  amended-source affordance (linking the Omnibus itself) is Phase 2.
* **Inserted** whole articles (e.g. Article 4a, 75a–d — introduced by the
  Omnibus) have no anchor in the base act at all, so Phase 1 leaves them
  bold-only rather than emit a link that lands nowhere.  They are detected from
  the retrieved subtree: only an *inserted* article node carries ``amended_by``
  at the article-root level (a merely amended article carries it on its
  sub-paragraphs, never the root).
"""
from __future__ import annotations

import re
import time
from typing import Any, Iterable

from application._grounded_citation import _BOLD_REF_RE
from domain.legislation_catalog import LEGISLATION
from domain.eurlex import (
    AMENDER_ARTICLE_ANCHOR,
    anchor_for_ref,
    display_from_anchor,
    is_known_celex,
    provision_url,
)

# All EUR-Lex URL construction — the viewer form, the qid, which CELEX to point
# at, the anchor grammar — lives in ``domain.eurlex`` (single source of truth).
# This module only parses answers and decides *what* to link.

# A CELEX code (e.g. 32024R1689) — filters MDCG guidance keys ("MDCG_2020_3")
# out of the name→CELEX table, since those have no EUR-Lex CELEX article anchors.
_CELEX_RE = re.compile(r"^\d{5}[A-Z]\d{4}$")

# High-precision regulation IDENTIFIERS for link disambiguation — deliberately
# NOT application._config._REG_PATTERNS. Those patterns include subject-matter
# concepts ("class iib", "ai system", "personal data", "data subject") tuned for
# retrieval-SCOPE detection ("does this answer mention the MDR at all?"). Reused
# for per-reference disambiguation they are catastrophic: an "Article 6" near the
# words "Class IIb" or "personal data" would link to the MDR/GDPR regardless of
# which act the article actually belongs to (observed: AI-Act Article 6(1)(b)
# linked to MDR Article 6 because the sentence said "under the MDR"). Only an
# actual act identifier — acronym, official short name, or number — may say which
# regulation a bare "Article N" belongs to. Numbers are added from the catalog.
_LINK_ALIASES: dict[str, list[str]] = {
    "32024R1689": ["ai act", "eu ai act", "artificial intelligence act",
                   "artificial intelligence regulation"],
    "32017R0745": ["mdr", "medical device regulation", "medical devices regulation"],
    "32017R0746": ["ivdr", "in vitro diagnostic regulation",
                   "in-vitro diagnostic regulation"],
    "32016R0679": ["gdpr", "general data protection regulation"],
    "32026R1744": ["digital omnibus", "ai omnibus", "omnibus regulation", "omnibus"],
    "32026R0977": ["common implementing regulation"],
}


# (name-pattern, base CELEX) pairs for adjacency resolution, longest pattern
# first so a specific name ("medical device regulation") wins over a short one
# ("mdr") and short/noisy patterns do not shadow it.
def _build_name_patterns() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for celex, aliases in _LINK_ALIASES.items():
        if not is_known_celex(celex):
            continue
        for pat in aliases:
            p = pat.strip().lower()
            if len(p) >= 3:
                pairs.append((p, celex))
        number = (LEGISLATION.get(celex, {}).get("number") or "").strip().lower()
        if number:                                # e.g. "2024/1689"
            pairs.append((number, celex))
    # de-dup, keep first (a pattern maps to one reg), longest first
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for p, c in sorted(pairs, key=lambda x: len(x[0]), reverse=True):
        if p not in seen:
            seen.add(p)
            uniq.append((p, c))
    return uniq


_NAME_PATTERNS: list[tuple[str, str]] = _build_name_patterns()

_ADJACENCY_WINDOW = 48  # chars scanned each side of a reference for a reg name

# A regulation name only disambiguates a reference when it is BOUND to it as part
# of the same citation phrase ("Article 6 of the AI Act", "the GDPR's Article 6",
# "MDR Article 10") — never when a content word sits between them ("under the MDR,
# *satisfying* Article 6(1)"). The gap between the name and the reference may hold
# only citation scaffolding; any other word means the name describes something
# else in the sentence, so the reference is left unresolved (→ bold-only).
_CONNECTOR_WORDS = frozenset(
    {"of", "the", "a", "an", "under", "in", "to", "as", "per", "pursuant",
     "regulation", "eu", "ec", "no", "council", "european", "directive", "and", "s"}
)


def _is_bound(gap: str) -> bool:
    """True when *gap* (text between a reg name and a reference) is only citation
    scaffolding — so the name genuinely labels the reference."""
    return all(t in _CONNECTOR_WORDS for t in re.findall(r"[a-z]+", gap.lower()))

# Only a letter-suffixed article (75d) is *inserted* and absent from the base act;
# a purely-numeric amended article (75) still lives in the base and must keep its
# base link, so the suffix is required here.
_ARTICLE_ID_NUM = re.compile(r"_art_(\d+[a-z]+)$")


def _resolve_celex(
    line: str,
    start: int,
    end: int,
    in_scope: frozenset[str],
    default_celex: str | None = None,
) -> str | None:
    """Recover the CELEX for a reference at ``line[start:end]``.

    Nearest regulation name *after* the reference wins ("Article 6 AI Act");
    then nearest *before* ("the AI Act's Article 6"); then the answer's dominant
    regulation (*default_celex*, set only when the question targets a single
    regulation); then, if exactly one regulation is in scope, that one. ``None``
    when still ambiguous — the caller leaves the reference bold-only.

    The default is what closes the single-regulation gap: in an all-AI-Act answer
    the model drops the "AI Act" qualifier after the first mentions ("Article 53
    …", "Article 96 …"), leaving nothing for adjacency to catch, so those
    references (and the whole "Provisions cited" footer) fell through. An
    explicitly-named cross-reference still resolves by adjacency first, so a
    "Article 9 GDPR" inside an AI-Act answer is never mislabelled.
    """
    after = line[end : end + _ADJACENCY_WINDOW]
    after_l = after.lower()
    best: str | None = None
    best_pos = len(after) + 1
    for pat, celex in _NAME_PATTERNS:
        i = after_l.find(pat)
        # bound only if nothing but scaffolding sits between the ref and the name
        if i != -1 and i < best_pos and _is_bound(after[:i]):
            best_pos, best = i, celex
    if best is not None:
        return best

    before = line[max(0, start - _ADJACENCY_WINDOW) : start]
    before_l = before.lower()
    best_end = -1
    for pat, celex in _NAME_PATTERNS:
        i = before_l.rfind(pat)
        if i != -1 and (i + len(pat)) > best_end and _is_bound(before[i + len(pat):]):
            best_end, best = i + len(pat), celex
    if best is not None:
        return best

    if default_celex is not None:
        return default_celex
    if len(in_scope) == 1:
        return next(iter(in_scope))
    return None


def build_link_scope(
    provisions: list[dict[str, Any]] | None,
    target_celexes: Iterable[str] | None,
) -> tuple[frozenset[str], dict[tuple[str, str], str]]:
    """Return ``(in_scope_celexes, inserted_articles)`` for :func:`link_references`.

    ``inserted_articles`` maps ``(celex, article_number)`` -> the amending act's
    CELEX, for every article whose *root* was inserted by an amending act
    (root-level ``amended_by``). The base act has no anchor for these, so they
    link to the amending act instead of the base.
    """
    in_scope: set[str] = {c for c in (target_celexes or []) if c}
    inserted: dict[tuple[str, str], str] = {}
    for p in provisions or []:
        celex = p.get("celex")
        if celex:
            in_scope.add(celex)
        subtree = p.get("subtree") or []
        if subtree and celex:
            root = subtree[0]
            amender = root.get("amended_by")
            if amender and root.get("kind") == "article":
                m = _ARTICLE_ID_NUM.search(root.get("id") or "")
                if m:
                    inserted[(celex, m.group(1).lower())] = amender
    return frozenset(in_scope), inserted


def link_references(
    text: str,
    *,
    in_scope_celexes: frozenset[str] = frozenset(),
    inserted_articles: dict[tuple[str, str], str] | None = None,
    default_celex: str | None = None,
    cited: dict[tuple[str, str], str] | None = None,
) -> str:
    """Bold every provision reference and link the *first* mention of each article.

    Same reference detection and heading/quote-line skipping as
    :func:`_bold_references`; a reference whose CELEX cannot be resolved (or is an
    inserted article) is wrapped in ``**bold**`` exactly as before.

    Because EUR-Lex anchors are article-grained ("Article 50(1)", "Article 50(2)"
    and "Article 50(3)" all resolve to ``#art_50``), only the **first** mention of
    a given article *within a section* is linked — later mentions stay bold-only,
    so the prose reads like a memo, not a wall of identical links. The section
    resets at each heading so a long answer keeps a nearby link per section.

    When *cited* is provided it is filled (in first-seen order) with
    ``{(celex, anchor): "Article 50"}`` for every distinct provision linked
    anywhere in the answer — the source for :func:`build_provisions_footer` — so
    the footer stays complete even though inline links are deduped.
    """
    inserted_articles = inserted_articles or {}
    qid = int(time.time() * 1000)                   # one fresh EUR-Lex qid per answer
    linked_here: set[tuple[str, str]] = set()       # (celex, anchor) linked in this section

    def _sub(m: "re.Match[str]") -> str:
        ref = m.group(1)
        anchor = anchor_for_ref(ref)
        if anchor is None:
            return f"**{ref}**"
        celex = _resolve_celex(
            m.string, m.start(), m.end(), in_scope_celexes, default_celex
        )
        if celex is None or not is_known_celex(celex):
            return f"**{ref}**"
        # Inserted article (no base anchor) → link to the amending act instead.
        amender = inserted_articles.get((celex, anchor[4:])) if anchor.startswith("art_") else None
        if amender and is_known_celex(amender):
            key = (amender, AMENDER_ARTICLE_ANCHOR)
            if key in linked_here:
                return f"**{ref}**"
            linked_here.add(key)
            return f"[**{ref}**]({provision_url(amender, AMENDER_ARTICLE_ANCHOR, qid=qid)})"
        if amender:                                 # amender not linkable → bold-only
            return f"**{ref}**"
        key = (celex, anchor)
        if cited is not None and key not in cited:  # footer records every provision
            cited[key] = display_from_anchor(anchor)
        if key in linked_here:
            return f"**{ref}**"                     # already linked this section — dedupe
        linked_here.add(key)
        return f"[**{ref}**]({provision_url(celex, anchor, qid=qid)})"

    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            linked_here.clear()                     # new section — first mention links again
            out.append(line)
            continue
        if stripped.startswith(">"):
            out.append(line)                        # verbatim quote — untouched
            continue
        out.append(_BOLD_REF_RE.sub(_sub, line))
    return "\n".join(out)


# ── "Provisions cited" footer (the RA table of authorities) ──────────────────

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN_VALUES.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _footer_sort_key(anchor: str) -> tuple[int, int, str]:
    """Order the footer in document order: articles (numeric) → annexes → recitals."""
    kind, _, rest = anchor.partition("_")
    rank = {"art": 0, "anx": 1, "rct": 2}.get(kind, 3)
    if kind == "art":
        m = re.match(r"(\d+)([a-z]*)", rest)
        return (rank, int(m.group(1)) if m else 0, m.group(2) if m else "")
    if kind == "anx":
        return (rank, _roman_to_int(rest), "")
    if kind == "rct":
        return (rank, int(rest) if rest.isdigit() else 0, "")
    return (rank, 0, rest)


def _reg_label(celex: str, reg_names: dict[str, str]) -> str:
    """Display label for a regulation in the footer: the answer's own label, else
    the catalog name, else its number, else the raw CELEX."""
    meta = LEGISLATION.get(celex, {})
    return reg_names.get(celex) or meta.get("name") or meta.get("number") or celex


def build_provisions_footer(
    cited: dict[tuple[str, str], str],
    reg_names: dict[str, str] | None = None,
    amender_celexes: Iterable[str] | None = None,
) -> str:
    """Return a "Provisions cited" appendix — one EUR-Lex link per distinct
    provision, grouped by regulation and ordered within each group, plus an
    "Amending act" line per amending act the answer relies on (its provisions are
    consolidated into the base act, so the amending act itself is where the change
    is authoritatively found).

    Empty when nothing was linked.  ``reg_names`` maps a base CELEX to its display
    label (e.g. "EU AI Act"); a CELEX with no label falls back to its number.
    """
    amender_celexes = [c for c in (amender_celexes or []) if c]
    if not cited and not amender_celexes:
        return ""
    reg_names = reg_names or {}
    qid = int(time.time() * 1000)

    groups: dict[str, list[str]] = {}               # celex -> anchors, first-seen celex order
    for (celex, anchor) in cited:
        groups.setdefault(celex, []).append(anchor)
    cited_bases = set(groups)                        # base regulations the answer actually cites

    lines = ["", "---", "", "**Provisions cited**", ""]
    for celex, anchors in groups.items():
        items = [
            f"[{display_from_anchor(a)}]({provision_url(celex, a, qid=qid)})"
            for a in sorted(set(anchors), key=_footer_sort_key)
        ]
        lines.append(f"- **{_reg_label(celex, reg_names)}** — " + " · ".join(items))

    for amender in dict.fromkeys(amender_celexes):  # de-dup, keep order
        if amender in groups:
            continue                                # already listed as a cited regulation
        # An amending act belongs here only if the answer cites the base act it
        # amends. Otherwise it is a cross-reference that got pulled into retrieval
        # (e.g. an AI Act provision surfacing in an MDR answer dragged in the
        # Digital Omnibus) — showing "Amending act — Digital Omnibus on AI" on an
        # MDR answer is just noise.
        amends = LEGISLATION.get(amender, {}).get("amends")
        if amends and amends not in cited_bases:
            continue
        url = provision_url(amender, AMENDER_ARTICLE_ANCHOR, qid=qid)
        lines.append(
            f"- **Amending act — {_reg_label(amender, reg_names)}** — "
            f"[Article 1]({url}) *(amendments applied above)*"
        )
    return "\n".join(lines)
