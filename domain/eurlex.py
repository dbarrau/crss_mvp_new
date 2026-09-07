# domain/eurlex.py

"""Single source of truth for EUR-Lex provision URLs.

Everything about how a provision reference becomes a clickable EUR-Lex link lives
here — the viewer endpoint, the ``qid`` the viewer needs to honour a ``#anchor``,
which CELEX to point at (the official consolidation when one exists, else the base
act), the anchor grammar, and where an amending act's changes live.

The application citation layer (``application/_eurlex_links.py``) parses answers
and decides *what* to link and *how* to render it; this module decides *where*
each link points. So when EUR-Lex behaviour changes — a different viewer form, a
new anchor scheme — there is exactly one file to edit.

Verified against the live site and the cached EUR-Lex HTML:

* Articles anchor as ``#art_50`` (``#art_75d`` for inserted, letter-suffixed
  articles); annexes as ``#anx_III`` (roman); recitals as ``#rct_81``. Article
  level is the finest grain EUR-Lex exposes — there are no per-paragraph anchors.
* The ``/TXT/`` viewer loads its body client-side; on a URL with no ``qid`` it
  navigates to append one and drops the ``#anchor`` (landing at the top). A
  ``qid`` already present suppresses that redirect. It is only a millisecond
  timestamp and is not validated, so a fresh one is minted per URL.
* Consolidated documents (``source_celex``) share the base act's anchor scheme,
  so pointing at the consolidated CELEX shows current law at the same ``#anchor``.
"""
from __future__ import annotations

import re
import time

from domain.legislation_catalog import LEGISLATION

# ── the viewer + URL template ────────────────────────────────────────────────
# The single knob for the EUR-Lex viewer form. "EN/TXT" is the standard viewer;
# "EN/TXT/HTML" is the server-rendered plain-HTML view. Change it here only.
_VIEW = "EN/TXT"
_URL_TEMPLATE = (
    "https://eur-lex.europa.eu/legal-content/{view}/?uri=CELEX:{celex}&qid={qid}#{anchor}"
)

# ── which CELEX a base act links to ──────────────────────────────────────────
# Base CELEX -> the CELEX to point at: its official consolidation (current law)
# when the catalog declares one via ``source_celex``, else the base act itself.
_DISPLAY_CELEX: dict[str, str] = {
    celex: (meta.get("source_celex") or celex) for celex, meta in LEGISLATION.items()
}

# Amending acts place a base act's amendments in their first article (the Digital
# Omnibus amends the AI Act entirely within its Article 1). An inserted article
# has no anchor of its own anywhere, so it links there.
AMENDER_ARTICLE_ANCHOR = "art_1"


def is_known_celex(celex: str) -> bool:
    """True when *celex* is a catalog regulation we can build a link for."""
    return celex in _DISPLAY_CELEX


def display_celex(celex: str) -> str:
    """The CELEX to put in a URL for *celex* — its consolidation when one exists."""
    return _DISPLAY_CELEX.get(celex, celex)


def provision_url(celex: str, anchor: str, *, qid: int | None = None) -> str:
    """Full EUR-Lex URL for *anchor* within *celex* (consolidated when available).

    A fresh ``qid`` is minted unless one is supplied (pass a shared one to keep a
    single answer's links uniform)."""
    return _URL_TEMPLATE.format(
        view=_VIEW,
        celex=display_celex(celex),
        qid=qid if qid is not None else int(time.time() * 1000),
        anchor=anchor,
    )


# ── anchor grammar (reference text <-> EUR-Lex anchor) ───────────────────────
_ANCHOR_ARTICLE = re.compile(r"Articles?\s+(\d+[a-z]?)", re.IGNORECASE)
_ANCHOR_ANNEX = re.compile(r"Annex(?:es)?\s+([IVXLC]+)", re.IGNORECASE)
_ANCHOR_RECITAL = re.compile(r"Recitals?\s+(\d+)", re.IGNORECASE)


def anchor_for_ref(ref: str) -> str | None:
    """``"Article 25(2)"`` → ``"art_25"`` (article is the finest grain),
    ``"Annex III"`` → ``"anx_III"``, ``"Recital 81"`` → ``"rct_81"``.  ``None``
    for an unrecognised shape."""
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


def display_from_anchor(anchor: str) -> str:
    """``"art_50"`` → ``"Article 50"``, ``"anx_III"`` → ``"Annex III"``,
    ``"rct_81"`` → ``"Recital 81"`` — the human label for a footer / list."""
    kind, _, rest = anchor.partition("_")
    return {"art": "Article", "anx": "Annex", "rct": "Recital"}.get(kind, kind) + f" {rest}"
