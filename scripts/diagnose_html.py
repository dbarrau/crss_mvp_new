#!/usr/bin/env python3
"""Diagnose structural differences between legal-basis and consolidated EUR-Lex HTML.

Compares two HTML files and reports differences in the elements the parser relies on:
- Section container IDs (tit_1, pbl_1, enc_1, fnp_1)
- Article/chapter/section ID patterns
- Paragraph div ID patterns
- CSS classes used by the parser
- Extra wrapper elements that could break recursive=False searches
- Amendment markers specific to consolidated documents

Usage:
    python scripts/diagnose_html.py <legal_basis.html> <consolidated.html>

Example:
    python scripts/diagnose_html.py \
        data/legislation/32017R0745/EN/raw/raw_legal_basis.html \
        data/legislation/32017R0745/EN/raw/raw.html
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# ── Patterns the parser relies on ────────────────────────────────────────────

SECTION_IDS = ["tit_1", "pbl_1", "enc_1", "fnp_1"]
ARTICLE_RE = re.compile(r"^art_(\d+[a-z]?)$")
CHAPTER_RE = re.compile(r"^cpt_([IVXLCDM]+)$")
SECTION_RE = re.compile(r"^cpt_([IVXLCDM]+)\.sct_(\d+)$")
PARAGRAPH_RE = re.compile(r"^(\d{3})\.(\d{3})$")
ANNEX_RE = re.compile(r"^anx_[A-Za-z0-9]+$")
CITATION_RE = re.compile(r"^cit_\d+")
RECITAL_RE = re.compile(r"^rct_\d+")

# Known amendment markers in consolidated EUR-Lex HTML
AMENDMENT_MARKER_RE = re.compile(r"[▼►][A-Z0-9]+")


def analyze(soup: BeautifulSoup, label: str) -> dict:
    """Extract structural summary from a parsed HTML document."""
    info: dict = {"label": label}

    # 1. Section containers
    for sid in SECTION_IDS:
        el = soup.find(id=sid)
        info[f"has_{sid}"] = el is not None
        if el:
            info[f"{sid}_tag"] = el.name
            info[f"{sid}_classes"] = el.get("class", [])

    # 2. Articles
    articles = soup.find_all(id=ARTICLE_RE)
    info["article_count"] = len(articles)
    info["article_ids"] = sorted(
        [a["id"] for a in articles],
        key=lambda x: (int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0),
    )

    # 3. Chapters & Sections
    info["chapter_count"] = len(soup.find_all(id=CHAPTER_RE))
    info["section_count"] = len(soup.find_all(id=SECTION_RE))

    # 4. Paragraphs
    paragraphs = soup.find_all(id=PARAGRAPH_RE)
    info["paragraph_count"] = len(paragraphs)

    # 5. Annexes
    annexes = soup.find_all(id=ANNEX_RE)
    info["annex_count"] = len(annexes)
    info["annex_ids"] = [a["id"] for a in annexes]

    # 6. Citations & Recitals
    info["citation_count"] = len(soup.find_all(id=CITATION_RE))
    info["recital_count"] = len(soup.find_all(id=RECITAL_RE))

    # 7. CSS classes used – count all unique class names
    class_counter: Counter = Counter()
    for tag in soup.find_all(True):
        for cls in tag.get("class", []):
            class_counter[cls] += 1
    info["class_counts"] = class_counter

    # 8. Amendment markers (text nodes containing ▼B, ▼M1, ►M1, etc.)
    markers: list[str] = []
    for text_node in soup.find_all(string=AMENDMENT_MARKER_RE):
        parent = text_node.parent
        if parent:
            markers.append(f"{parent.name}.{','.join(parent.get('class', []))}: {text_node.strip()[:80]}")
    info["amendment_markers"] = markers
    info["amendment_marker_count"] = len(markers)

    # 9. Check article→paragraph nesting: are paragraph divs direct children of article divs?
    nesting_issues = []
    for art in articles[:5]:  # Sample first 5 articles
        art_id = art["id"]
        # Find paragraph divs that are descendants but NOT direct children
        all_paras = art.find_all(id=PARAGRAPH_RE)
        direct_paras = art.find_all(id=PARAGRAPH_RE, recursive=False)
        if len(all_paras) != len(direct_paras):
            # Find what's between article and paragraph
            for para in all_paras:
                if para not in direct_paras:
                    chain = []
                    p = para.parent
                    while p and p != art:
                        chain.append(f"{p.name}#{p.get('id', '')}[{','.join(p.get('class', []))}]")
                        p = p.parent
                    nesting_issues.append({
                        "article": art_id,
                        "paragraph": para["id"],
                        "wrapper_chain": " → ".join(reversed(chain)),
                    })
    info["nesting_issues"] = nesting_issues

    # 10. Check chapter→article nesting similarly
    chapter_nesting_issues = []
    enc_root = soup.find(id="enc_1")
    if enc_root:
        chapters = enc_root.find_all(id=CHAPTER_RE, recursive=False)
        direct_articles_in_enc = enc_root.find_all(id=ARTICLE_RE, recursive=False)
        info["articles_directly_in_enc"] = len(direct_articles_in_enc)
        info["chapters_directly_in_enc"] = len(chapters)

        for chap in chapters[:3]:
            all_arts = chap.find_all(id=ARTICLE_RE)
            direct_arts = chap.find_all(id=ARTICLE_RE, recursive=False)
            if len(all_arts) != len(direct_arts):
                for a in all_arts:
                    if a not in direct_arts:
                        chain = []
                        p = a.parent
                        while p and p != chap:
                            chain.append(f"{p.name}#{p.get('id', '')}[{','.join(p.get('class', []))}]")
                            p = p.parent
                        chapter_nesting_issues.append({
                            "chapter": chap["id"],
                            "article": a["id"],
                            "wrapper_chain": " → ".join(reversed(chain)),
                        })
    info["chapter_nesting_issues"] = chapter_nesting_issues

    # 11. Direct children tag types of enc_1
    if enc_root:
        child_tags = Counter()
        for child in enc_root.children:
            if isinstance(child, Tag):
                tag_desc = child.name
                cid = child.get("id", "")
                cls = ",".join(child.get("class", []))
                if cid:
                    tag_desc += f"#{cid}"
                if cls:
                    tag_desc += f".{cls}"
                child_tags[tag_desc] += 1
        info["enc_direct_children"] = child_tags

    return info


def compare(a: dict, b: dict) -> None:
    """Print side-by-side comparison of two analysis dicts."""
    print("=" * 80)
    print(f"  LEGAL BASIS: {a['label']}")
    print(f"  CONSOLIDATED: {b['label']}")
    print("=" * 80)

    # Section containers
    print("\n── Section Container IDs ──")
    for sid in SECTION_IDS:
        a_has = a.get(f"has_{sid}", False)
        b_has = b.get(f"has_{sid}", False)
        mark = "✓" if a_has == b_has else "✗ DIFF"
        print(f"  {sid}: basis={a_has}  consolidated={b_has}  {mark}")

    # Counts
    print("\n── Element Counts ──")
    for key in ["article_count", "chapter_count", "section_count",
                "paragraph_count", "annex_count", "citation_count", "recital_count"]:
        av = a.get(key, 0)
        bv = b.get(key, 0)
        mark = "✓" if av == bv else f"DIFF ({bv - av:+d})"
        print(f"  {key:25s}: basis={av:4d}  consolidated={bv:4d}  {mark}")

    # Article IDs comparison
    a_arts = set(a.get("article_ids", []))
    b_arts = set(b.get("article_ids", []))
    only_basis = a_arts - b_arts
    only_consol = b_arts - a_arts
    if only_basis or only_consol:
        print("\n── Article ID Differences ──")
        if only_basis:
            print(f"  Only in legal basis ({len(only_basis)}): {sorted(only_basis)[:20]}")
        if only_consol:
            print(f"  Only in consolidated ({len(only_consol)}): {sorted(only_consol)[:20]}")
    else:
        print("\n── Article IDs: IDENTICAL ──")

    # Annex IDs comparison
    a_anx = set(a.get("annex_ids", []))
    b_anx = set(b.get("annex_ids", []))
    if a_anx != b_anx:
        print("\n── Annex ID Differences ──")
        print(f"  Only in legal basis: {sorted(a_anx - b_anx)}")
        print(f"  Only in consolidated: {sorted(b_anx - a_anx)}")
    else:
        print(f"\n── Annex IDs: IDENTICAL ({len(a_anx)} annexes) ──")

    # CSS class differences
    print("\n── CSS Class Differences (top 20 by delta) ──")
    a_cls = a.get("class_counts", Counter())
    b_cls = b.get("class_counts", Counter())
    all_classes = set(a_cls) | set(b_cls)
    diffs = []
    for cls in all_classes:
        ac = a_cls.get(cls, 0)
        bc = b_cls.get(cls, 0)
        if ac != bc:
            diffs.append((cls, ac, bc, bc - ac))
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    only_basis_cls = [c for c in all_classes if c in a_cls and c not in b_cls]
    only_consol_cls = [c for c in all_classes if c not in a_cls and c in b_cls]
    if only_consol_cls:
        print(f"  ** Classes ONLY in consolidated: {sorted(only_consol_cls)} **")
    if only_basis_cls:
        print(f"  ** Classes ONLY in legal basis: {sorted(only_basis_cls)} **")
    for cls, ac, bc, delta in diffs[:20]:
        print(f"  {cls:40s}: basis={ac:5d}  consolidated={bc:5d}  delta={delta:+d}")

    # Amendment markers
    print(f"\n── Amendment Markers ──")
    print(f"  Legal basis:   {a.get('amendment_marker_count', 0)}")
    print(f"  Consolidated:  {b.get('amendment_marker_count', 0)}")
    if b.get("amendment_markers"):
        print(f"  First 10 consolidated markers:")
        for m in b["amendment_markers"][:10]:
            print(f"    {m}")

    # Nesting issues (THE KEY DIAGNOSTIC)
    print(f"\n── Article→Paragraph Nesting (recursive=False breakage) ──")
    a_issues = a.get("nesting_issues", [])
    b_issues = b.get("nesting_issues", [])
    print(f"  Legal basis issues:   {len(a_issues)}")
    print(f"  Consolidated issues:  {len(b_issues)}")
    for issue in b_issues[:10]:
        print(f"    art={issue['article']}  para={issue['paragraph']}")
        print(f"      wrapper chain: {issue['wrapper_chain']}")

    print(f"\n── Chapter→Article Nesting ──")
    a_ci = a.get("chapter_nesting_issues", [])
    b_ci = b.get("chapter_nesting_issues", [])
    print(f"  Legal basis issues:   {len(a_ci)}")
    print(f"  Consolidated issues:  {len(b_ci)}")
    for issue in b_ci[:10]:
        print(f"    chap={issue['chapter']}  art={issue['article']}")
        print(f"      wrapper chain: {issue['wrapper_chain']}")

    # enc_1 direct children
    print(f"\n── enc_1 Direct Children Types ──")
    print("  Legal basis:")
    for k, v in sorted(a.get("enc_direct_children", {}).items()):
        print(f"    {k}: {v}")
    print("  Consolidated:")
    for k, v in sorted(b.get("enc_direct_children", {}).items()):
        print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legal_basis", type=Path, help="Path to legal-basis raw.html")
    parser.add_argument("consolidated", type=Path, help="Path to consolidated raw.html")
    args = parser.parse_args()

    print(f"Loading legal basis: {args.legal_basis} ({args.legal_basis.stat().st_size:,} bytes)")
    soup_a = BeautifulSoup(args.legal_basis.read_text(encoding="utf-8"), "html.parser")
    print(f"Loading consolidated: {args.consolidated} ({args.consolidated.stat().st_size:,} bytes)")
    soup_b = BeautifulSoup(args.consolidated.read_text(encoding="utf-8"), "html.parser")

    info_a = analyze(soup_a, str(args.legal_basis))
    info_b = analyze(soup_b, str(args.consolidated))

    compare(info_a, info_b)


if __name__ == "__main__":
    main()
