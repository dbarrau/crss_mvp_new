"""Base annex parser — older-OJ-format handling (ingestion/parse/structural_layer).

The MDR/IVDR use an older EUR-Lex layout than the AI Act: numbered items are
`<p class="oj-normal">` WRAPPED in a class-less `<div>`, the numbering has no
space ("1.Contact"), and correlation annexes are marker-less data `<table>`s.
The parser dropped all three, leaving 5 annexes empty (→ "show me" fabricates).
These tests pin the fixes; a real-data test confirms 0 empty annexes.
"""
import os

import pytest
from bs4 import BeautifulSoup

from ingestion.parse.structural_layer.annex_parser import (
    _CELL_SEP,
    _collect_elements,
    _is_data_table,
    _parse_dotted,
)


def _rows(html: str):
    t = BeautifulSoup(html, "html.parser").find("table")
    return (t.find("tbody") or t).find_all("tr", recursive=False)


# ── no-space numbering ───────────────────────────────────────────────────────

def test_parse_dotted_accepts_no_space_numbering():
    assert _parse_dotted("1.Contact lenses") == ("1", "Contact lenses", 1)   # older OJ, no space
    assert _parse_dotted("1.   Contact lenses") == ("1", "Contact lenses", 1)  # AI-Act spacing
    assert _parse_dotted("1.1.1.Sub") == ("1.1.1", "Sub", 3)


def test_parse_dotted_does_not_mistake_a_decimal_for_a_marker():
    assert _parse_dotted("1.5 million devices") is None       # "1.5" is a value, not "1."


# ── plain <div> wrapper flattening ───────────────────────────────────────────

def test_collect_elements_unwraps_plain_div_wrappers():
    # older-OJ format wraps each item in a class-less <div>
    div = BeautifulSoup(
        '<div id="anx_XVI">'
        '<p class="oj-doc-ti">ANNEX XVI</p>'
        '<div><p class="oj-normal">1.Contact lenses</p></div>'
        '<div><p class="oj-normal">2.Products</p></div>'
        '</div>', "html.parser").find("div")
    els = _collect_elements(div)
    # the two wrapped <p>s are surfaced; the title is dropped
    assert [e.name for e in els] == ["p", "p"]
    assert all("oj-normal" in (e.get("class") or []) for e in els)


def test_collect_elements_keeps_enumeration_spacing_div():
    div = BeautifulSoup(
        '<div id="anx_X"><div class="oj-enumeration-spacing"><p>1.</p><p>x</p></div></div>',
        "html.parser").find("div")
    els = _collect_elements(div)
    assert [e.name for e in els] == ["div"]                   # kept for its own handler


# ── data-table (correlation table) detection ─────────────────────────────────

def test_is_data_table_true_for_marker_less_correlation_table():
    rows = _rows(
        "<table><tbody>"
        "<tr><td>Directive 90/385/EEC</td><td>Directive 93/42/EEC</td><td>This Regulation</td></tr>"
        "<tr><td>Article 1(1)</td><td>Article 1(1)</td><td>Article 2</td></tr>"
        "</tbody></table>")
    assert _is_data_table(rows) is True


def test_is_data_table_false_for_enumerated_table():
    rows = _rows(
        "<table><tbody>"
        "<tr><td>1.</td><td>The requirement text.</td></tr>"
        "</tbody></table>")
    assert _is_data_table(rows) is False


# ── real-data integration ────────────────────────────────────────────────────

_have = all(os.path.isfile(f"data/legislation/{c}/EN/raw/raw.html")
            for c in ("32017R0745", "32017R0746"))


@pytest.mark.skipif(not _have, reason="MDR/IVDR raw HTML not ingested")
@pytest.mark.parametrize("celex,correlation", [("32017R0745", "XVII"), ("32017R0746", "XV")])
def test_older_oj_annexes_no_longer_empty(celex, correlation):
    from ingestion.parse.normalizer import normalize_consolidated_html
    from ingestion.parse.universal_eurlex_parser import parse_eurlex_html
    raw = open(f"data/legislation/{celex}/EN/raw/raw.html", encoding="utf-8", errors="replace").read()
    byid = {n["id"]: n for n in parse_eurlex_html(normalize_consolidated_html(raw), celex, celex)["provisions"]}

    def kids(nid):
        return [byid[c] for c in byid[nid].get("children", []) if c in byid]

    # no annex is empty
    empty = [n["number"] for n in byid.values() if n.get("kind") == "annex"
             and not kids(n["id"]) and len(n.get("text") or "") < 120]
    assert empty == []

    # the correlation annex is a real data table (annex_row leaves, cell-separated)
    rows = kids(f"{celex}_anx_{correlation}")
    assert len(rows) > 20 and all(r["kind"] == "annex_row" for r in rows)
    assert _CELL_SEP in rows[0]["text"]                       # header row carries multiple cells
