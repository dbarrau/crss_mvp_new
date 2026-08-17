"""Verbatim provision display (application/_display.py).

A "show me Article X" request must render the AUTHORITATIVE corpus text — never an
LLM reconstruction (which fabricated base paragraphs). These tests cover the
intent gate, the insertion splice, and the verbatim render, all without Neo4j
(the retriever is faked) or the LLM.
"""
from application._display import (
    _extract_quoted_block,
    _insert_after_paragraph,
    _is_flat_insertion,
    _operation_of,
    _parse_inserted_paragraphs,
    render_provision_display,
    wants_verbatim_display,
)

_HEAD = "Article 1 — Amendments to Regulation (EU) 2024/1689 | "


# ── intent gate ──────────────────────────────────────────────────────────────

def test_detector_fires_on_pure_display_request():
    assert wants_verbatim_display("Show me Article 6 of the AI Act") == ("Article 6", "32024R1689")
    assert wants_verbatim_display("What does Article 6 of the AI Act say?") == ("Article 6", "32024R1689")


def test_detector_declines_analytical_questions():
    # Naming a provision is not enough — anything analytical must generate.
    assert wants_verbatim_display("How does Article 6 of the AI Act apply to my device?") is None
    assert wants_verbatim_display("What are the obligations under Article 16 of the AI Act?") is None
    assert wants_verbatim_display(
        "If a company integrates an LLM into a medical device, is it high-risk under Article 6?"
    ) is None


def test_detector_needs_exactly_one_provision_and_one_reg():
    assert wants_verbatim_display("Show me Articles 6 and 9 of the AI Act") is None   # two refs
    assert wants_verbatim_display("Show me Article 6") is None                        # no reg named


# ── amendment-instruction parsing ────────────────────────────────────────────

def test_operation_of_reads_the_eu_grammar():
    assert _operation_of("in Article 6, the following paragraphs are inserted: ‘1a. …’") == "inserted"
    assert _operation_of("in Article 43, paragraph 3 is replaced by the following: ‘3. …’") == "replaced"
    assert _operation_of("in Section A, point 1 is deleted") == "deleted"


def test_extract_quoted_block_prefers_curly_quotes():
    assert _extract_quoted_block("paragraph 3 is replaced by the following: ‘3. New text.’") == "3. New text."


def test_parse_inserted_paragraphs_splits_and_tags():
    block = ("1a. First inserted paragraph, long enough to be real. "
             "1b. Second inserted paragraph, also long enough.")
    nodes = _parse_inserted_paragraphs(block, "Article 6", "Regulation (EU) 2026/1744")
    assert [n["number"] for n in nodes] == ["1a", "1b"]
    assert nodes[0]["ref"] == "Article 6(1a)"
    assert nodes[0]["text"].startswith("1a. First inserted")
    assert all(n["_inserted_by"] == "Regulation (EU) 2026/1744" for n in nodes)


def test_insert_after_paragraph_lands_between_1_and_2():
    base = [
        {"depth": 1, "kind": "paragraph", "number": "1", "text": "1. one"},
        {"depth": 2, "kind": "point", "number": "a", "text": "point a"},
        {"depth": 1, "kind": "paragraph", "number": "2", "text": "2. two"},
    ]
    new = [{"depth": 1, "kind": "paragraph", "number": "1a", "text": "1a. inserted"}]
    out = _insert_after_paragraph(base, "1", new)
    order = [n["text"] for n in out]
    assert order == ["1. one", "point a", "1a. inserted", "2. two"]   # after 1 AND its child (a)


# ── full render ──────────────────────────────────────────────────────────────

_SUBTREE = [
    {"id": "32024R1689_art_6", "depth": 0, "kind": "article", "number": "6",
     "ref": "Article 6", "text": "Classification rules for high-risk AI systems"},
    {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 6(1)",
     "text": "1. Base paragraph one, the operative classification condition."},
    {"id": "p1a", "depth": 2, "kind": "point", "number": "a", "ref": "Article 6(1), point (a)",
     "text": "the AI system is a safety component;"},
    {"id": "p2", "depth": 1, "kind": "paragraph", "number": "2", "ref": "Article 6(2)",
     "text": "2. Base paragraph two, the Annex III route."},
]
_AMDS = [{
    "article_text": _HEAD + ("in Article 6, the following paragraphs are inserted: "
                             "‘1a. Inserted paragraph alpha, long enough to be a real unit here. "
                             "1b. Inserted paragraph beta, also long enough to be real.’"),
    "_amends_target_ref": "Article 6",
    "amending_act": "Regulation (EU) 2026/1744",
}]


class _FakeRetriever:
    def __init__(self, subtree, amds):
        self._subtree, self._amds = subtree, amds

    def retrieve_by_refs(self, refs, celex_filter):
        return [{"article_ref": "Article 6", "regulation_id": "32024R1689", "subtree": self._subtree}]

    def retrieve_amendments(self, ids):
        return self._amds


def test_render_is_verbatim_with_insertion_spliced_and_named():
    out = render_provision_display(_FakeRetriever(_SUBTREE, _AMDS), "Article 6", "32024R1689")
    # base paragraphs are the real corpus text, in order
    assert "Base paragraph one" in out
    assert "Base paragraph two" in out
    # enumerators are bolded so marked / the front-end normaliser cannot mangle them
    assert "**1.**" in out and "**(a)**" in out
    # insertions land between 1 and 2, tagged, and before paragraph 2
    assert "Inserted paragraph alpha" in out and "Inserted paragraph beta" in out
    assert out.index("Inserted paragraph alpha") < out.index("Base paragraph two")
    assert "inserted by Regulation (EU) 2026/1744" in out
    # header names the consolidating act; footer carries the pedigree
    assert "Digital Omnibus on AI" in out
    assert "AMENDMENTS APPLIED" in out


def test_render_without_amendments_is_plain_verbatim():
    out = render_provision_display(_FakeRetriever(_SUBTREE, []), "Article 6", "32024R1689")
    assert "Base paragraph one" in out
    assert "AMENDMENTS APPLIED" not in out          # no footer when nothing was amended
    assert "inserted by" not in out


def test_replace_tags_the_rendered_node_even_with_empty_chapeau():
    # Art 6(3)/43(3) store an empty depth-1 chapeau; the text lives in a
    # subparagraph. The 'replaced' note must land on the rendered node, not the
    # skipped empty container — otherwise base text ships with no inline flag.
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "43", "ref": "Article 43",
         "text": "Conformity assessment"},
        {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 43(1)",
         "text": "1. Base paragraph one."},
        {"id": "p3", "depth": 1, "kind": "paragraph", "number": "3", "ref": "Article 43(3)",
         "text": ""},                                                   # empty chapeau
        {"id": "p3s1", "depth": 2, "kind": "subparagraph", "number": "1",
         "ref": "Article 43(3), subparagraph 1", "text": "3. Original paragraph three text."},
    ]
    amds = [{"article_text": _HEAD + "in Article 43, paragraph 3 is replaced by the following: ‘3. New text.’",
             "_amends_target_ref": "Article 43", "amending_act": "Regulation (EU) 2026/1744"}]

    class _R:
        def retrieve_by_refs(self, refs, cf):
            return [{"article_ref": "Article 43", "regulation_id": "32024R1689", "subtree": subtree}]
        def retrieve_amendments(self, ids):
            return amds

    out = render_provision_display(_R(), "Article 43", "32024R1689")
    assert "Original paragraph three text." in out                     # verbatim base, not fabricated
    assert "replaced by Regulation (EU) 2026/1744" in out              # inline flag on the rendered node
    assert "AMENDMENTS APPLIED" in out


def test_point_with_inline_cross_reference_keeps_its_leading_marker_bolded():
    # A point (b) that references "point (a)" mid-sentence must keep its OWN
    # leading marker bolded, so the front-end list normaliser cannot split the
    # point in two at the inline reference (the reported rendering bug).
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "6", "ref": "Article 6", "text": "Head"},
        {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 6(1)",
         "text": "1. Both of the following conditions are fulfilled:"},
        {"id": "pa", "depth": 2, "kind": "point", "number": "a", "ref": "Article 6(1), point (a)",
         "text": "the AI system is a safety component of a product listed in Annex I;"},
        {"id": "pb", "depth": 2, "kind": "point", "number": "b", "ref": "Article 6(1), point (b)",
         "text": "the product whose safety component pursuant to point (a) is the AI system is required to undergo assessment."},
    ]

    class _R:
        def retrieve_by_refs(self, refs, cf):
            return [{"article_ref": "Article 6", "regulation_id": "32024R1689", "subtree": subtree}]
        def retrieve_amendments(self, ids):
            return []

    out = render_provision_display(_R(), "Article 6", "32024R1689")
    assert "**(b)** the product whose safety component pursuant to point (a) is the AI system" in out


def test_roman_subitems_nest_deeper_than_their_point():
    # Art 7(2)(k) has roman sub-items (i)/(ii). They must render indented DEEPER
    # than point (k) (measured from the paragraph, via non-ASCII indent), so the
    # hierarchy shows and the letter-point (i) isn't confused with the roman (i).
    nbsp = " "
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "7", "ref": "Article 7", "text": "Head"},
        {"id": "p2", "depth": 1, "kind": "paragraph", "number": "2", "ref": "Article 7(2)",
         "text": "2. The Commission shall take into account the following criteria:"},
        {"id": "pi", "depth": 2, "kind": "point", "number": "i", "ref": "Article 7(2), point (i)",
         "text": "the extent to which the outcome is corrigible;"},
        {"id": "pk", "depth": 2, "kind": "point", "number": "k", "ref": "Article 7(2), point (k)",
         "text": "the extent to which existing Union law provides for:"},
        {"id": "pki", "depth": 3, "kind": "roman_item", "number": "i", "ref": "Article 7(2), point (k)(i)",
         "text": "effective measures of redress in relation to the risks posed;"},
        {"id": "pkii", "depth": 3, "kind": "roman_item", "number": "ii", "ref": "Article 7(2), point (k)(ii)",
         "text": "effective measures to prevent or substantially minimise those risks."},
    ]

    class _R:
        def retrieve_by_refs(self, refs, cf):
            return [{"article_ref": "Article 7", "regulation_id": "32024R1689", "subtree": subtree}]
        def retrieve_amendments(self, ids):
            return []

    out = render_provision_display(_R(), "Article 7", "32024R1689")

    def indent_of(needle):
        for ln in out.splitlines():
            if needle in ln:
                body = ln[2:]                                  # strip "> "
                return len(body) - len(body.lstrip(nbsp))
        return -1

    assert indent_of("**(k)** the extent to which existing Union law") >= 0
    assert indent_of("**(i)** effective measures of redress") > indent_of("**(k)**")
    assert indent_of("**(ii)** effective measures to prevent") > indent_of("**(k)**")


def test_is_flat_insertion_distinguishes_simple_from_complex():
    # Article 6's shape (single, flat "paragraphs are inserted") is splice-able;
    # Article 5's (multi-part "amended as follows" / letter-point insert) is not.
    assert _is_flat_insertion("in Article 6, the following paragraphs are inserted: ‘1a. …’")
    assert not _is_flat_insertion(
        "Article 5 is amended as follows: in paragraph 1, the following points are inserted: ‘(ba) …’")
    assert not _is_flat_insertion("in paragraph 1, the following points are inserted: ‘(ba) …’")


def test_complex_insertion_is_flagged_not_reproduced_inline():
    # FAIL SAFE: a complex insertion must never be rendered inline as verbatim law
    # (the flat splice would flatten/garble it). Render base + a clear note.
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "5", "ref": "Article 5",
         "text": "Prohibited AI practices"},
        {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 5(1)",
         "text": "1. The following AI practices shall be prohibited:"},
    ]
    amds = [{"_amends_target_ref": "Article 5", "amending_act": "Regulation (EU) 2026/1744",
             "article_text": _HEAD + ("Article 5 is amended as follows: in paragraph 1, the following "
                                      "points are inserted: ‘(ba) some inserted point text long enough here.’")}]

    class _R:
        def retrieve_by_refs(self, refs, cf):
            return [{"article_ref": "Article 5", "regulation_id": "32024R1689", "subtree": subtree}]
        def retrieve_amendments(self, ids):
            return amds

    out = render_provision_display(_R(), "Article 5", "32024R1689")
    body = out.split("AMENDMENTS APPLIED")[0]
    assert "inserts further provisions" in body                       # flagged, not silently dropped
    assert "(ba) some inserted point text" not in body                # NOT reproduced inline as verbatim


def test_render_returns_none_when_unresolved():
    class _Empty:
        def retrieve_by_refs(self, refs, celex_filter):
            return []
        def retrieve_amendments(self, ids):
            return []
    assert render_provision_display(_Empty(), "Article 6", "32024R1689") is None
