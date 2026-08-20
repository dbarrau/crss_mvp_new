#!/usr/bin/env python
"""Audit for SILENT partial text loss: content that exists in a regulation's
raw EUR-Lex HTML but never made it into the parsed article body.

This is the general text-conservation guarantee that the two narrower sweeps do
NOT provide:

  * audit_render_coverage.py     catches only FULLY EMPTY provisions (a binary).
  * audit_orphan_subparagraphs.py catches only ONE loss shape (a trailing <p>
                                  emitted as a sibling of a paragraph-id div).

An article can be non-empty AND have every trailing <p> present yet still drop a
whole obligation mid-body — e.g. a second subparagraph inside a quoted amendment,
or a numbered paragraph that follows a point-list. Confirmed Aug 2026 on MDR
Article 117 (a combination-products conformity duty) and IVDR Articles 112 / 110
/ 42 / 38, none of which the two sweeps above flag.

Method (parse-time, mirrors audit_orphan_subparagraphs' raw-vs-parsed contract):
for every article div in the raw HTML, compare its visible text against the
freshly PARSED article subtree (data/legislation/<celex>/EN/parsed.json). The
comparison is an ORDER-INDEPENDENT multiset difference of meaningful content
tokens, not a contiguous-substring match — the parser legitimately reorders and
reformats structure (heading labels move to `number`/`title` fields, list
markers "(a)" are normalised, definition sub-items are re-sequenced), so a
substring check drowns in false positives. Only tokens ABSENT from the parsed
subtree entirely count as lost; relocated text does not. A difflib span is shown
purely for human context.

Amending regulations (catalog type ending "_amending_regulation", e.g. the
Digital Omnibus 32026R1744) are skipped: their operative text is a list of
amendment instructions applied into the TARGET regulation's graph by the
consolidation pipeline, so it is expected NOT to appear verbatim as stored
provision text — flagging them would be a guaranteed false positive.

    python scripts/audit_text_conservation.py
    python scripts/audit_text_conservation.py --min 10 32017R0746
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

from domain.legislation_catalog import LEGISLATION
from domain.ontology.eurlex_html import ARTICLE_ID_RE
from ingestion.parse.normalizer import normalize_consolidated_html

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "legislation"
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z0-9]+")
_ARTICLE_LABEL = re.compile(r"^Article\s+\d+[a-z]?\s*")

# Structural / ubiquitous tokens carry no evidentiary weight for "was text lost":
# they occur everywhere, so a raw count that exceeds the parsed count usually
# reflects reformatting, not a dropped obligation. "article" is here because the
# raw "Article N" heading label is dropped on purpose (the number lives in
# `number`, the caption in `title`).
_STOP = set(
    "the of to a and or in for on by with that this as be is are shall may not "
    "an at from which such it its any other article".split()
)


def _ctoks(text: str) -> list[str]:
    return [w for w in _TOK.findall(text.lower()) if w not in _STOP]


def _subtree_text(byid: dict, node_id: str, _depth: int = 0) -> str:
    """Faithful reconstruction of a provision's rendered body: marker + caption
    + text, then every descendant in order (matches what the display renders)."""
    if _depth > 12 or node_id not in byid:
        return ""
    n = byid[node_id]
    parts = [str(n.get("number") or ""), n.get("title") or "", n.get("text") or ""]
    parts.extend(_subtree_text(byid, c, _depth + 1) for c in n.get("children", []))
    return " ".join(p for p in parts if p)


def _is_amending(celex: str) -> bool:
    return str(LEGISLATION.get(celex, {}).get("type", "")).endswith("_amending_regulation")


def _check_doc(celex: str, min_absent: int) -> list[tuple[int, str, str]]:
    """(absent_token_count, article_id, longest_dropped_span) per lossy article."""
    raw_path = _DATA_DIR / celex / "EN" / "raw" / "raw.html"
    parsed_path = _DATA_DIR / celex / "EN" / "parsed.json"
    if not raw_path.is_file() or not parsed_path.is_file():
        return []

    soup = BeautifulSoup(
        normalize_consolidated_html(raw_path.read_text(encoding="utf-8", errors="replace")),
        "html.parser",
    )
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    provisions = parsed["provisions"] if isinstance(parsed, dict) else parsed
    byid = {n["id"]: n for n in provisions}

    flags: list[tuple[int, str, str]] = []
    for art_div in soup.find_all("div", id=ARTICLE_ID_RE):
        art_id = f"{celex}_{art_div['id']}"
        src = _ARTICLE_LABEL.sub("", _WS.sub(" ", art_div.get_text(" ", strip=True)))
        got = _subtree_text(byid, art_id)

        absent = Counter(_ctoks(src)) - Counter(_ctoks(got))  # order-independent: truly missing
        n_absent = sum(absent.values())
        if n_absent < min_absent:
            continue

        # Largest contiguous raw span not matched in the parsed body — for display only.
        sm = difflib.SequenceMatcher(None, src.split(), got.split(), autojunk=False)
        span = max(
            (" ".join(src.split()[i1:i2]) for tag, i1, i2, _, _ in sm.get_opcodes()
             if tag in ("delete", "replace")),
            key=len,
            default="",
        )
        flags.append((n_absent, art_id, span))
    return flags


def main(min_absent: int = 6, only: list[str] | None = None) -> int:
    if not _DATA_DIR.is_dir():
        print("No data/legislation/ directory found — nothing to audit.")
        return 0

    docs = sorted(d.name for d in _DATA_DIR.iterdir() if d.is_dir())
    if only:
        docs = [d for d in docs if d in set(only)]

    total = 0
    for celex in docs:
        if _is_amending(celex):
            print(f"— {celex}: skipped (amending regulation; operative text applied into target graph)")
            continue
        flags = _check_doc(celex, min_absent)
        total += len(flags)
        mark = "⚠" if flags else "✓"
        print(f"{mark} {celex}: {len(flags)} article(s) with content in raw HTML but ABSENT from parsed body")
        for n_absent, art_id, span in sorted(flags, reverse=True):
            print(f"      {art_id}: {n_absent} tokens absent — {span[:110]!r}")

    if total:
        print(
            f"\n⚠ {total} article(s) silently drop substantive text at parse time. "
            "Re-parse and check the enacting-terms / annex parser for these shapes."
        )
    else:
        print("\n✓ No partial text loss — every raw article's content survives into the parsed body.")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audit for partial (mid-body) text loss at parse time.")
    ap.add_argument("docs", nargs="*", help="Limit to these CELEX ids (default: all loaded).")
    ap.add_argument("--min", type=int, default=6,
                    help="Min absent meaningful-token occurrences to flag an article (default 6).")
    args = ap.parse_args()
    raise SystemExit(0 if main(args.min, args.docs or None) == 0 else 1)
