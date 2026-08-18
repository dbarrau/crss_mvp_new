"""Stage 2 consolidation applier (consolidation/applier.py).

The applier mutates a copy of the base regulation's parsed.json so it holds
current law.  It is the corpus-mutating step, so these tests pin the target
resolver (traversal), the node builder (base-format ids), and every tree editor
(insert / replace / add / delete / heading), then confirm the real Digital
Omnibus consolidates the decisive articles correctly.
"""
import json
import os

import pytest

from consolidation.amendment_parser import (
    ContentNode,
    Operation,
    AmendmentInstruction,
    parse_amending_regulation,
)
from consolidation.applier import consolidate, resolve


# ── a tiny synthetic base act to exercise the editors directly ───────────────

def _node(nid, kind, number="", text="", children=None, parent=None, depth=1):
    return {
        "id": nid, "kind": kind, "number": number, "text": text,
        "children": children or [], "parent_id": parent, "path": [], "lang": "EN",
        "hierarchy_depth": depth, "binding_force": "binding",
        "source_type": "regulation", "text_for_analysis": text,
    }


def _base_article6():
    """Chapter → Article 6 → paragraphs 006.001 (with points a,b), 006.002."""
    return [
        _node("X_cpt_III", "chapter", "III", "High-risk", ["X_art_6"], depth=2),
        _node("X_art_6", "article", "6", "Classification", ["X_006.001", "X_006.002"],
              parent="X_cpt_III", depth=3),
        _node("X_006.001", "paragraph", "1", "1.   Chapeau",
              ["X_006.001_pt_a", "X_006.001_pt_b"], parent="X_art_6", depth=4),
        _node("X_006.001_pt_a", "point", "a", "first point", parent="X_006.001", depth=5),
        _node("X_006.001_pt_b", "point", "b", "second point", parent="X_006.001", depth=5),
        _node("X_006.002", "paragraph", "2", "2.   Second para",
              parent="X_art_6", depth=4),
    ]


def _instr(*ops):
    return [AmendmentInstruction("1", "X", list(ops))]


def _byid(provisions):
    return {n["id"]: n for n in provisions}


# ── target resolution ────────────────────────────────────────────────────────

def test_resolve_article_paragraph_point():
    byid = _byid(_base_article6())
    node = resolve(byid, "X", {"article": "6", "para": "1", "point": "b"})
    assert node["id"] == "X_006.001_pt_b"


def test_resolve_container_for_insert():
    # an insert locator (no leaf) resolves to the container to insert into
    byid = _byid(_base_article6())
    assert resolve(byid, "X", {"article": "6"})["id"] == "X_art_6"


# ── insert ───────────────────────────────────────────────────────────────────

def test_insert_paragraph_lands_after_its_anchor():
    op = Operation("insert", "paragraph", "Article 6", target={"article": "6"},
                   content=[ContentNode("1a", "paragraph", "Inserted 1a")])
    cons, report = consolidate(_base_article6(), _instr(op), "OMNI")
    art6 = _byid(cons)["X_art_6"]
    # 1a splices between paragraph 1 and paragraph 2
    assert art6["children"] == ["X_006.001", "X_006.001a", "X_006.002"]
    new = _byid(cons)["X_006.001a"]
    assert new["kind"] == "paragraph" and new["number"] == "1a"
    assert new["text"] == "1a.   Inserted 1a"
    assert new["amended_by"] == "OMNI" and new["amendment_op"] == "new"
    assert report[0].status == "applied"


def test_insert_points_after_lettered_anchor():
    op = Operation("insert", "point", "Article 6(1)",
                   target={"article": "6", "para": "1"},
                   content=[ContentNode("ba", "point", "new ba")])
    cons, _ = consolidate(_base_article6(), _instr(op), "OMNI")
    para1 = _byid(cons)["X_006.001"]
    assert para1["children"] == ["X_006.001_pt_a", "X_006.001_pt_b", "X_006.001_pt_ba"]


# ── replace (with a child subtree) ───────────────────────────────────────────

def test_replace_point_swaps_text_and_rebuilds_children():
    op = Operation("replace", "point", "Article 6(1), point (b)",
                   target={"article": "6", "para": "1", "point": "b"},
                   content=[ContentNode("b", "point", "REPLACED",
                                        [ContentNode("i", "roman_item", "sub i")])])
    cons, _ = consolidate(_base_article6(), _instr(op), "OMNI")
    byid = _byid(cons)
    b = byid["X_006.001_pt_b"]
    assert b["text"] == "REPLACED" and b["amendment_op"] == "replace"
    assert b["children"] == ["X_006.001_pt_b_rm_i"]
    assert byid["X_006.001_pt_b_rm_i"]["text"] == "sub i"


def test_replace_heading_keeps_the_body():
    op = Operation("replace", "heading", "Article 6",
                   target={"article": "6", "heading": True},
                   content=[ContentNode("", "provision", "New Title")])
    cons, _ = consolidate(_base_article6(), _instr(op), "OMNI")
    art6 = _byid(cons)["X_art_6"]
    assert art6["text"] == "New Title" and art6.get("title") == "New Title"
    assert art6["children"] == ["X_006.001", "X_006.002"]      # body untouched


# ── add / delete ─────────────────────────────────────────────────────────────

def test_add_point_appends_at_end():
    op = Operation("add", "point", "Article 6(1)",
                   target={"article": "6", "para": "1"},
                   content=[ContentNode("c", "point", "added c")])
    cons, _ = consolidate(_base_article6(), _instr(op), "OMNI")
    assert _byid(cons)["X_006.001"]["children"][-1] == "X_006.001_pt_c"


def test_delete_removes_node_and_subtree():
    op = Operation("delete", "point", "Article 6(1), point (a)",
                   target={"article": "6", "para": "1", "point": "a"})
    cons, _ = consolidate(_base_article6(), _instr(op), "OMNI")
    byid = _byid(cons)
    assert "X_006.001_pt_a" not in byid
    assert byid["X_006.001"]["children"] == ["X_006.001_pt_b"]


def test_base_is_not_mutated():
    base = _base_article6()
    before = json.dumps(base, sort_keys=True)
    op = Operation("delete", "point", "Article 6(1), point (a)",
                   target={"article": "6", "para": "1", "point": "a"})
    consolidate(base, _instr(op), "OMNI")
    assert json.dumps(base, sort_keys=True) == before      # pure: source untouched


# ── integration: the real Digital Omnibus over the real AI Act ───────────────

_have_data = (os.path.isfile("data/legislation/32026R1744/EN/raw/raw.html")
              and os.path.isfile("data/legislation/32024R1689/EN/parsed.json"))


@pytest.mark.skipif(not _have_data, reason="AI Act / Omnibus data not ingested")
def test_omnibus_consolidates_the_decisive_articles():
    base = json.load(open("data/legislation/32024R1689/EN/parsed.json"))["provisions"]
    ins = parse_amending_regulation("32026R1744")
    cons, report = consolidate(base, ins, "32026R1744")
    byid = {n["id"]: n for n in cons}

    def kids(nid):
        return [byid[c] for c in byid[nid]["children"] if c in byid]

    # Article 6: 1a/1b/1c inserted after paragraph 1
    a6 = [c["number"] for c in kids("32024R1689_art_6") if c["kind"] == "paragraph"]
    assert a6[:5] == ["1", "1a", "1b", "1c", "2"]

    # Article 5(1) first subparagraph: (ba)/(bb) inserted after (b)
    a5pts = [c["number"] for c in kids("32024R1689_005.001_sp_1")]
    assert a5pts[:5] == ["a", "b", "ba", "bb", "c"]

    # Article 113 third paragraph point (c): the deferred dates land as (i)/(ii)
    c = byid["32024R1689_art_113_sp_3_pt_c"]
    assert c["amendment_op"] == "replace"
    dates = " ".join(byid[g]["text"] for g in c["children"])
    assert "2 December 2027" in dates and "2 August 2028" in dates

    # Annex I Section A point 1 deleted
    assert "32024R1689_anx_I_sec_A_1" not in byid

    # Whole-article ops must NEST, not leave an empty article node:
    #  - Article 4 replaced (AI literacy) with its paragraphs
    art4 = byid["32024R1689_art_4"]
    assert art4["text"] == "AI literacy"
    assert len([c for c in kids("32024R1689_art_4") if c["kind"] == "paragraph"]) == 3
    #  - Article 4a inserted with its paragraphs
    assert "AI literacy" not in byid["32024R1689_art_4a"]["text"]     # its own heading
    assert kids("32024R1689_art_4a")                                  # non-empty
    #  - the multi-article run 75a–75d all inserted (not just the first)
    for aid in ("art_75a", "art_75b", "art_75c", "art_75d"):
        assert kids(f"32024R1689_{aid}"), f"{aid} should have paragraphs"

    # the added Annex XIV is real law — consolidated (not deferred/denied) with
    # its numbered sections
    anx = byid["32024R1689_anx_XIV"]
    assert anx["number"] == "XIV"
    assert len([c for c in kids("32024R1689_anx_XIV") if c["kind"] == "annex_point"]) >= 3

    # coverage: every operation applied — nothing skipped or deferred
    from collections import Counter
    status = Counter(r.status for r in report)
    assert status["applied"] == 74
    assert status.get("skipped", 0) == 0 and status.get("deferred", 0) == 0
