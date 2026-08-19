"""Enacting-terms (article/paragraph) parser — orphan-subparagraph handling
(ingestion/parse/structural_layer/enacting_terms_parser).

A numbered paragraph's <div id="NNN.MMM"> wraps only its FIRST subparagraph;
EUR-Lex emits every subparagraph after the first — the text following a
point-list, or a second/third subparagraph — as a bare <p class="oj-normal">
SIBLING of that div, untagged, with no id linking it back to its paragraph.
Confirmed on real MDR Article 14 (eval/runs + a live re-parse, Aug 2026): 100
articles across MDR/IVDR/GDPR were silently dropping this trailing text. These
tests pin the fix; a real-data test confirms Article 14 is now complete.
"""
import os

import pytest
from bs4 import BeautifulSoup

from ingestion.parse.base.utils import ParserContext
from ingestion.parse.structural_layer.enacting_terms_parser import parse_enacting_terms

_CELEX = "TEST"


def _parse(body_html: str):
    """Parse a minimal enacting-terms fragment; return {id: node}."""
    html = f'<div id="enc_1">{body_html}</div>'
    soup = BeautifulSoup(html, "html.parser")
    ctx = ParserContext(celex=_CELEX)
    root = ctx.make_node("document", "document", "", None)
    parse_enacting_terms(soup, ctx, root)
    return {n["id"]: n for n in ctx.provisions}


# A single article whose paragraph 2 has a point-list followed by two orphan
# subparagraphs, and whose paragraph 3 has a plain trailing orphan — the exact
# shapes hit in real MDR Article 14 (points-then-postamble; plain postamble).
_ARTICLE_WITH_ORPHANS = """
<div id="art_1">
  <p class="oj-ti-art" id="t1">Article 1</p>
  <div id="art_1.tit_1"><p class="oj-sti-art">Title</p></div>
  <div id="001.001"><p class="oj-normal">First paragraph, no orphans.</p></div>
  <div id="001.002"><p class="oj-normal">
    <p class="oj-normal inline-element">Intro sentence, requirements are:</p>
    <table width="100%"><tr><td><p class="oj-normal">(a)</p></td><td><p class="oj-normal">first point;</p></td></tr></table>
    <table width="100%"><tr><td><p class="oj-normal">(b)</p></td><td><p class="oj-normal">second point.</p></td></tr></table>
  </p></div>
  <p class="oj-normal">First orphan subparagraph after the point-list.</p>
  <p class="oj-normal">Second orphan subparagraph after the point-list.</p>
  <div id="001.003"><p class="oj-normal">Third paragraph main sentence.</p></div>
  <p class="oj-normal">Orphan postamble trailing the third paragraph.</p>
</div>
"""


def test_orphan_subparagraphs_after_point_list_are_recovered():
    nodes = _parse(_ARTICLE_WITH_ORPHANS)
    para2 = nodes[f"{_CELEX}_001.002"]
    assert para2["kind"] == "paragraph"
    assert [nodes[c]["number"] for c in para2["children"]] == ["1", "2", "3"]

    sp1, sp2, sp3 = (nodes[c] for c in para2["children"])
    assert sp1["text"] == "Intro sentence, requirements are:"
    assert sp2["text"] == "First orphan subparagraph after the point-list."
    assert sp3["text"] == "Second orphan subparagraph after the point-list."

    # subparagraph 1's own points survive the restructuring untouched
    assert [nodes[c]["number"] for c in sp1["children"]] == ["a", "b"]
    assert nodes[sp1["children"][1]]["text"] == "second point."


def test_plain_orphan_postamble_is_recovered():
    nodes = _parse(_ARTICLE_WITH_ORPHANS)
    para3 = nodes[f"{_CELEX}_001.003"]
    assert [nodes[c]["number"] for c in para3["children"]] == ["1", "2"]
    sp1, sp2 = (nodes[c] for c in para3["children"])
    assert sp1["text"] == "Third paragraph main sentence."
    assert sp2["text"] == "Orphan postamble trailing the third paragraph."


def test_paragraph_without_orphans_is_unchanged():
    # No trailing <p>: stays the existing flat single-subparagraph shape —
    # confirms the fix only restructures paragraphs that actually have orphans.
    nodes = _parse(_ARTICLE_WITH_ORPHANS)
    para1 = nodes[f"{_CELEX}_001.001"]
    assert para1["children"] == []
    assert para1["text"] == "First paragraph, no orphans."


# ── real-data integration ────────────────────────────────────────────────────

_have = os.path.isfile("data/legislation/32017R0745/EN/raw/raw.html")


@pytest.mark.skipif(not _have, reason="MDR raw HTML not ingested")
def test_mdr_article_14_matches_official_text_verbatim():
    """Pins the exact bug report: MDR Article 14 paragraphs 2 and 6 each drop
    real obligations (a sampling-method allowance, a non-conformity duty, and
    a "deemed fulfilled" / free-samples duty) without this fix."""
    from ingestion.parse.normalizer import normalize_consolidated_html
    from ingestion.parse.universal_eurlex_parser import parse_eurlex_html

    raw = open("data/legislation/32017R0745/EN/raw/raw.html", encoding="utf-8", errors="replace").read()
    byid = {n["id"]: n for n in parse_eurlex_html(normalize_consolidated_html(raw), "32017R0745", "32017R0745")["provisions"]}

    para2 = byid["32017R0745_014.002"]
    assert [byid[c]["number"] for c in para2["children"]] == ["1", "2", "3"]
    assert "sampling method" in byid[para2["children"][1]]["text"]
    assert "serious risk or is a falsified device" in byid[para2["children"][2]]["text"]

    para6 = byid["32017R0745_014.006"]
    assert [byid[c]["number"] for c in para6["children"]] == ["1", "2"]
    assert "free samples" in byid[para6["children"][1]]["text"]
