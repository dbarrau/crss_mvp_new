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
from typing import Any, Iterable

from application._config import _REG_NAME_TO_CELEX, _REG_PATTERNS
from application._grounded_citation import _BOLD_REF_RE
from domain.legislation_catalog import LEGISLATION

_EURLEX_TMPL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}#{anchor}"
)

# A CELEX code (e.g. 32024R1689) — filters MDCG guidance keys ("MDCG_2020_3")
# out of the name→CELEX table, since those have no EUR-Lex CELEX article anchors.
_CELEX_RE = re.compile(r"^\d{5}[A-Z]\d{4}$")

# base CELEX -> the CELEX to link to (its official consolidation when one exists,
# else itself).  Consolidated documents share the base's #art_N anchor scheme.
_URL_CELEX: dict[str, str] = {
    celex: (meta.get("source_celex") or celex) for celex, meta in LEGISLATION.items()
}

# (name-pattern, base CELEX) pairs for adjacency resolution, longest pattern
# first so a specific name ("medical device regulation") wins over a short one
# ("mdr") and short/noisy patterns do not shadow it.
def _build_name_patterns() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, celex in _REG_NAME_TO_CELEX.items():
        if not _CELEX_RE.match(celex):
            continue
        for pat in _REG_PATTERNS.get(name, [name]):
            p = pat.strip().lower()
            if len(p) >= 3:                       # skip ultra-generic 1-2 char noise
                pairs.append((p, celex))
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

_ANCHOR_ARTICLE = re.compile(r"Articles?\s+(\d+[a-z]?)", re.IGNORECASE)
_ANCHOR_ANNEX = re.compile(r"Annex(?:es)?\s+([IVXLC]+)", re.IGNORECASE)
_ANCHOR_RECITAL = re.compile(r"Recitals?\s+(\d+)", re.IGNORECASE)

_ARTICLE_ID_NUM = re.compile(r"_art_(\d+[a-z]?)$")


def _anchor_from_ref(ref: str) -> str | None:
    """``"Article 25(2)"`` -> ``"art_25"``; ``"Annex III"`` -> ``"anx_III"``;
    ``"Recital 81"`` -> ``"rct_81"``.  ``None`` for an unrecognised shape."""
    m = _ANCHOR_ARTICLE.search(ref)
    if m:
        return f"art_{m.group(1).lower()}"
    m = _ANCHOR_ANNEX.search(ref)
    if m:
        return f"anx_{m.group(1).upper()}"
    m = _ANCHOR_RECITAL.search(ref)
    if m:
        return f"rct_{m.group(1)}"
    return None


def _resolve_celex(
    line: str, start: int, end: int, in_scope: frozenset[str]
) -> str | None:
    """Recover the CELEX for a reference at ``line[start:end]``.

    Nearest regulation name *after* the reference wins ("Article 6 AI Act");
    then nearest *before* ("the AI Act's Article 6"); then, if exactly one
    regulation is in scope, that one.  ``None`` when still ambiguous — the caller
    leaves the reference bold-only.
    """
    after = line[end : end + _ADJACENCY_WINDOW].lower()
    best: str | None = None
    best_pos = len(after) + 1
    for pat, celex in _NAME_PATTERNS:
        i = after.find(pat)
        if i != -1 and i < best_pos:
            best_pos, best = i, celex
    if best is not None:
        return best

    before = line[max(0, start - _ADJACENCY_WINDOW) : start].lower()
    best_end = -1
    for pat, celex in _NAME_PATTERNS:
        i = before.rfind(pat)
        if i != -1 and (i + len(pat)) > best_end:
            best_end, best = i + len(pat), celex
    if best is not None:
        return best

    if len(in_scope) == 1:
        return next(iter(in_scope))
    return None


def build_link_scope(
    provisions: list[dict[str, Any]] | None,
    target_celexes: Iterable[str] | None,
) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    """Return ``(in_scope_celexes, inserted_articles)`` for :func:`link_references`.

    ``inserted_articles`` is the set of ``(celex, article_number)`` whose article
    *root* was inserted by an amending act (root-level ``amended_by``); Phase 1
    suppresses links on these because the base act has no such anchor.
    """
    in_scope: set[str] = {c for c in (target_celexes or []) if c}
    inserted: set[tuple[str, str]] = set()
    for p in provisions or []:
        celex = p.get("celex")
        if celex:
            in_scope.add(celex)
        subtree = p.get("subtree") or []
        if subtree and celex:
            root = subtree[0]
            if root.get("amended_by") and root.get("kind") == "article":
                m = _ARTICLE_ID_NUM.search(root.get("id") or "")
                if m:
                    inserted.add((celex, m.group(1).lower()))
    return frozenset(in_scope), frozenset(inserted)


def link_references(
    text: str,
    *,
    in_scope_celexes: frozenset[str] = frozenset(),
    inserted_articles: frozenset[tuple[str, str]] = frozenset(),
) -> str:
    """Bold every provision reference and, where the CELEX resolves, link it.

    Same reference detection and heading/quote-line skipping as
    :func:`_bold_references`; a reference whose CELEX cannot be resolved (or is an
    inserted article) is wrapped in ``**bold**`` exactly as before, so this is a
    strict superset of the bold-only behaviour.
    """

    def _sub(m: "re.Match[str]") -> str:
        ref = m.group(1)
        anchor = _anchor_from_ref(ref)
        if anchor is None:
            return f"**{ref}**"
        celex = _resolve_celex(m.string, m.start(), m.end(), in_scope_celexes)
        if celex is None or celex not in _URL_CELEX:
            return f"**{ref}**"
        if anchor.startswith("art_") and (celex, anchor[4:]) in inserted_articles:
            return f"**{ref}**"                     # inserted: no base anchor
        url = _EURLEX_TMPL.format(celex=_URL_CELEX[celex], anchor=anchor)
        return f"[**{ref}**]({url})"

    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith(">"):
            out.append(line)                        # heading or verbatim quote
            continue
        out.append(_BOLD_REF_RE.sub(_sub, line))
    return "\n".join(out)
