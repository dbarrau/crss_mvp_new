#!/usr/bin/env python
"""Audit for silently truncated paragraphs: text EUR-Lex emits outside its
owning paragraph's wrapper div, dropped if nothing goes looking for it.

Some EUR-Lex layouts wrap only a numbered paragraph's FIRST subparagraph in
its <div id="NNN.MMM">; every subparagraph after that — the text following a
point-list, or a plain second/third subparagraph — is emitted as a bare
<p class="oj-normal"> SIBLING of that div, untagged, with no id linking it
back. Unlike a fully empty provision (see audit_render_coverage.py), the
owning article still has substantial text, so it is invisible to that sweep —
the paragraph just silently ends early. Confirmed Aug 2026 on MDR Article 14:
CRSS answered "show me Article 14" and dropped two real obligations (a
sampling-method allowance and a non-conformity duty) with no error, anywhere.
A follow-up sweep found the SAME pattern in 100 articles across MDR/IVDR/GDPR.

This is a parse-time check, not a runtime graph check: it re-derives the
orphan <p> text directly from each document's raw HTML and confirms every
occurrence is present verbatim in that document's freshly PARSED provisions
(data/legislation/<celex>/EN/parsed.json) — so it catches both a regression in
enacting_terms_parser and a future regulation whose raw markup doesn't fit the
parser's assumptions, not just "does this exact bug's shape still exist".

    python scripts/audit_orphan_subparagraphs.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

from domain.ontology.eurlex_html import ARTICLE_ID_RE, CLASS_OJ_NORMAL, PARAGRAPH_ID_RE
from ingestion.parse.normalizer import normalize_consolidated_html

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "legislation"
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def _orphan_texts_by_article(raw_html: str) -> dict[str, list[str]]:
    """{article_html_id: [orphan subparagraph text, ...]} from raw EUR-Lex HTML."""
    soup = BeautifulSoup(normalize_consolidated_html(raw_html), "html.parser")
    out: dict[str, list[str]] = {}
    for art_div in soup.find_all("div", id=ARTICLE_ID_RE):
        pending = False
        texts: list[str] = []
        for child in art_div.find_all(["div", "p"], recursive=False):
            cid = child.get("id") if child.name == "div" else None
            if child.name == "div" and cid and PARAGRAPH_ID_RE.match(cid):
                pending = True
            elif pending and child.name == "p" and CLASS_OJ_NORMAL in (child.get("class") or []):
                texts.append(child.get_text(" ", strip=True))
        if texts:
            out[art_div["id"]] = texts
    return out


def _subtree_text(byid: dict, node_id: str, _depth: int = 0) -> str:
    if _depth > 8 or node_id not in byid:
        return ""
    n = byid[node_id]
    parts = [n.get("text") or ""]
    parts.extend(_subtree_text(byid, c, _depth + 1) for c in n.get("children", []))
    return " ".join(parts)


def _check_doc(celex: str) -> list[tuple[str, str]]:
    """Missing (article_id, text_snippet) pairs for one document; [] if clean."""
    raw_path = _DATA_DIR / celex / "EN" / "raw" / "raw.html"
    parsed_path = _DATA_DIR / celex / "EN" / "parsed.json"
    if not raw_path.is_file() or not parsed_path.is_file():
        return []

    orphans = _orphan_texts_by_article(raw_path.read_text(encoding="utf-8", errors="replace"))
    if not orphans:
        return []

    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    provisions = parsed["provisions"] if isinstance(parsed, dict) else parsed
    byid = {n["id"]: n for n in provisions}

    missing: list[tuple[str, str]] = []
    for art_local_id, texts in orphans.items():
        art_id = f"{celex}_{art_local_id}"
        full_text = _norm(_subtree_text(byid, art_id))
        for t in texts:
            norm_t = _norm(t)
            if norm_t and norm_t not in full_text:
                missing.append((art_id, norm_t[:100]))
    return missing


def main() -> int:
    if not _DATA_DIR.is_dir():
        print("No data/legislation/ directory found — nothing to audit.")
        return 0

    all_missing: dict[str, list[tuple[str, str]]] = {}
    for doc_dir in sorted(_DATA_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        missing = _check_doc(doc_dir.name)
        if missing:
            all_missing[doc_dir.name] = missing

    if not all_missing:
        print("✓ No orphaned subparagraphs — every EUR-Lex trailing <p> is present in its parsed article.")
        return 0

    total = sum(len(v) for v in all_missing.values())
    print(f"⚠ {total} subparagraph(s) present in the raw HTML but MISSING from the parsed output:\n")
    for celex, missing in all_missing.items():
        print(f"  {celex}:")
        for art_id, snippet in missing:
            print(f"    {art_id}: {snippet!r}…")
    print("\nThese are silently truncated, not empty — re-parse and check enacting_terms_parser.py.")
    return total


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
