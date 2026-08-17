"""Stage 1 amendment parser (consolidation/amendment_parser.py).

An amending regulation ("Article 1 — Amendments to Regulation (EU) N") is a list
of surgical directives interleaved with the quoted text they enact.  These tests
pin the recursive directive grammar, the structured locator that threads scope
across nesting, and the content-payload parser — the foundation the Stage 2
applier mutates the base graph from, so it must be exact.
"""
import os

import pytest

from consolidation.amendment_parser import (
    Locator,
    _classify_verb,
    _parse_locator,
    _split_enumerator,
    parse_amendments_html,
    parse_amending_regulation,
)

# EUR-Lex renders every point as a two-column table: [enumerator | content].
# These builders assemble that structure so fixtures read like the directives.


def _art1(*point_tables: str) -> str:
    return f'<div id="art_1">{"".join(point_tables)}</div>'


def _row(enum: str, content_html: str) -> str:
    return (f'<table><tbody><tr><td><p>{enum}</p></td>'
            f'<td>{content_html}</td></tr></tbody></table>')


def _pt(n, content_html: str) -> str:      # a top-level numbered point (n)
    return _row(f'({n})', content_html)


def _parse_one(content_html: str):
    """Parse a single top-level point and return its operations."""
    return parse_amendments_html(_art1(_pt(1, content_html)), "TESTCELEX")[0].operations


# ── verb classification ──────────────────────────────────────────────────────

def test_classify_all_verbs():
    assert _classify_verb("Article 5 is amended as follows:") == "amend"
    assert _classify_verb("paragraph 2 is replaced by the following:") == "replace"
    assert _classify_verb("the following paragraphs are inserted:") == "insert"
    assert _classify_verb("the following paragraph is added:") == "add"
    assert _classify_verb("point 1 is deleted") == "delete"


def test_classify_handles_are_replaced_plural():
    # regression: "are replaced" (plural subject) must classify as replace, not unknown
    assert _classify_verb("paragraphs 2 and 3 are replaced by the following:") == "replace"


# ── structured locator ───────────────────────────────────────────────────────

def test_locator_article_paragraph_render():
    loc = _parse_locator("in Article 17, paragraph 2 is replaced by the following:")
    assert loc.article == "17" and loc.para == "2"
    assert loc.render() == "Article 17(2)"


def test_locator_ordinal_paragraph_is_not_numbered():
    loc = _parse_locator("in Article 113, the third paragraph is amended as follows:")
    assert loc.article == "113" and loc.para_ord == "third" and loc.para == ""
    assert loc.render() == "Article 113, third paragraph"


def test_locator_ignores_the_following_new_item():
    # "the following point" is the enacted kind, not a locator segment
    loc = _parse_locator("in the first subparagraph, the following point is added:")
    assert loc.point == "" and loc.subpara == "first"


def test_locator_merge_threads_container_scope():
    # "point (a)" inside "Article 113, third paragraph" keeps the paragraph scope
    parent = _parse_locator("in Article 113, the third paragraph is amended as follows:")
    child = _parse_locator("point (a) is replaced by the following:")
    assert child.merged_under(parent).render() == "Article 113, third paragraph, point (a)"


def test_locator_child_paragraph_overrides_parent():
    parent = Locator(article="57")
    child = _parse_locator("in paragraph 9, point (e) is replaced by the following:")
    assert child.merged_under(parent).render() == "Article 57(9), point (e)"


# ── content enumerator splitting ─────────────────────────────────────────────

def test_split_enumerator_forms():
    assert _split_enumerator("‘1a. For the purposes") == ("1a", "For the purposes")
    assert _split_enumerator("(ba) the placing") == ("ba", "the placing")
    assert _split_enumerator("(i) that generation") == ("i", "that generation")
    assert _split_enumerator("Article 4a Some heading") == ("Article 4a", "Some heading")


# ── replace / insert / add / delete end-to-end ───────────────────────────────

def test_replace_paragraph():
    ops = _parse_one("<p>in Article 17, paragraph 2 is replaced by the following:</p>"
                     "<div><p>‘2. The new paragraph text.’</p></div>")
    assert len(ops) == 1
    op = ops[0]
    assert (op.op, op.item_kind, op.target_ref) == ("replace", "paragraph", "Article 17(2)")
    assert op.content[0].enumerator == "2"
    assert op.content[0].text == "The new paragraph text."


def test_insert_flat_paragraphs():
    # Article 6 shape: a run of paragraphs inserted, no sub-points
    ops = _parse_one(
        "<p>in Article 6, the following paragraphs are inserted:</p>"
        "<div><p>‘1a. First inserted.</p></div>"
        "<div><p>1b. Second inserted.’</p></div>")
    op = ops[0]
    assert (op.op, op.item_kind, op.target_ref) == ("insert", "paragraph", "Article 6")
    assert [c.enumerator for c in op.content] == ["1a", "1b"]


def test_insert_points_into_subparagraph():
    # Article 5(1) first subparagraph shape: points (ba)/(bb)
    ops = _parse_one(
        "<p>in paragraph 1, the first subparagraph, the following points are inserted:</p>"
        + _row("‘(ba)", "<p>first new point</p>")
        + _row("(bb)", "<p>second new point</p>"))
    # scope inherits Article from the CONTAINER, so at top level (no container) the
    # article is absent — assert the sub-locator resolved:
    op = ops[0]
    assert op.op == "insert" and op.item_kind == "point"
    assert "first subparagraph" in op.target_ref
    assert [c.enumerator for c in op.content] == ["ba", "bb"]


def test_nested_amend_threads_scope_to_points():
    # The Article 113 dates case: nested amend-as-follows → point (c) with romans
    inner = (_row("(a)", "<p>point (a) is replaced by the following:</p>"
                          "<div><p>‘(a) Chapters I and II apply from 2 Feb 2025;</p></div>")
             + _row("(c)", "<p>point (c) is replaced by the following:</p>"
                           "<div><p>‘(c) Chapter III applies from:</p>"
                           + _row("(i)", "<p>2 December 2027 as regards Annex III; and</p>")
                           + _row("(ii)", "<p>2 August 2028 as regards Annex I;’</p>")
                           + "</div>"))
    ops = _parse_one("<p>in Article 113, the third paragraph is amended as follows:</p>" + inner)
    assert [o.target_ref for o in ops] == [
        "Article 113, third paragraph, point (a)",
        "Article 113, third paragraph, point (c)",
    ]
    c = ops[1].content[0]
    assert [ch.text for ch in c.children] == [
        "2 December 2027 as regards Annex III; and",
        "2 August 2028 as regards Annex I",
    ]


def test_multi_target_replace_fans_out_and_splits_content():
    # As it really appears: under "Article 97 is amended as follows:" so the
    # container supplies the article scope.
    ops = _parse_one(
        "<p>Article 97 is amended as follows:</p>"
        + _row("(a)", "<p>paragraphs 2 and 3 are replaced by the following:</p>"
                      "<div><p>‘2. New two.</p></div>"
                      "<div><p>3. New three.’</p></div>"))
    assert [o.op for o in ops] == ["replace", "replace"]
    assert [o.target_ref for o in ops] == ["Article 97(2)", "Article 97(3)"]
    # each fanned-out op carries only its own matching content unit
    assert [c.enumerator for o in ops for c in o.content] == ["2", "3"]
    assert ops[0].content[0].text == "New two." and ops[1].content[0].text == "New three."


def test_multi_target_delete_fans_out():
    ops = _parse_one("<p>in Annex VIII, section B, points 7 and 9 are deleted;</p>")
    assert [o.op for o in ops] == ["delete", "delete"]
    assert "point 7" in ops[0].target_ref and "point 9" in ops[1].target_ref


def test_degenerate_amend_is_a_replace():
    # "point (14) is amended as follows: '(14) …'" — no sub-directives, whole new
    # text: semantically a replace, not a container.
    ops = _parse_one(
        "<p>point (14) is amended as follows:</p>"
        + _row("‘(14)", "<p>“safety component” means a new definition</p>"))
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "replace" and op.item_kind == "point"
    assert op.target_ref.endswith("point (14)")
    # the defined term's own opening quote is preserved (only the block delimiter is peeled)
    assert op.content[0].text.startswith("“safety component”")


def test_insert_new_article_takes_target_from_content():
    ops = _parse_one("<p>the following Article is inserted:</p>"
                     "<div><p>‘Article 4a Bias detection</p></div>")
    op = ops[0]
    assert op.op == "insert" and op.item_kind == "article"
    assert op.target_ref == "Article 4a"


def test_separator_semicolons_are_not_nodes():
    ops = _parse_one("<p>in Article 42, the following paragraph is added:</p>"
                     "<div><p>‘4. Added paragraph.’</p></div><p>;</p>")
    assert [c.text for c in ops[0].content] == ["Added paragraph."]


# ── integration: the real Digital Omnibus (skipped if data absent) ───────────

_OMNIBUS = "32026R1744"
_have_omnibus = os.path.isfile(
    os.path.join("data", "legislation", _OMNIBUS, "EN", "raw", "raw.html"))


@pytest.mark.skipif(not _have_omnibus, reason="Omnibus raw HTML not ingested")
def test_omnibus_full_coverage_and_dates():
    ins = parse_amending_regulation(_OMNIBUS)
    ops = [o for i in ins for o in i.operations]
    # every point classified, every op has a resolved target
    assert len(ins) == 43
    assert not [o for o in ops if o.op == "unknown"]
    assert not [o for o in ops if o.op != "delete" and not o.target_ref]
    assert all(i.target_celex == "32024R1689" for i in ins)

    # the decisive Article 113 dates land under the right romans
    pt40 = next(i for i in ins if i.point_num == "40")
    c = next(o for o in pt40.operations if o.target_ref.endswith("point (c)")).content[0]
    kids = " ".join(ch.text for ch in c.children)
    assert "2 December 2027" in kids and "Annex III" in kids
    assert "2 August 2028" in kids and "Annex I" in kids

    # Article 5's nested insert keeps the (a)→(i)/(ii) + (b) structure
    pt7 = next(i for i in ins if i.point_num == "7")
    para_1a = next(o for o in pt7.operations if o.item_kind == "paragraph").content[0]
    assert para_1a.enumerator == "1a"
    assert [ch.enumerator for ch in para_1a.children] == ["a", "b"]
    assert [g.enumerator for g in para_1a.children[0].children] == ["i", "ii"]
