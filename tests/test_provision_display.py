"""Verbatim provision display (application/_display.py).

A "show me Article X" request must render the AUTHORITATIVE corpus text — never an
LLM reconstruction (which fabricated base paragraphs). The graph is consolidated
(amending acts applied to the base nodes), so the display renders the ordered
HAS_PART subtree as-is and names the amending act(s) from the ``amended_by`` tag.
These tests cover the intent gate, the verbatim render, and the provenance footer
— all without Neo4j (the retriever is faked) or the LLM.
"""
from application._display import render_provision_display, wants_verbatim_display


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


# ── render helpers ───────────────────────────────────────────────────────────

class _Retriever:
    """Fakes the one method the render calls, returning a consolidated subtree."""
    def __init__(self, subtree, ref="Article 6"):
        self._subtree, self._ref = subtree, ref

    def retrieve_by_refs(self, refs, celex_filter):
        return [{"article_ref": self._ref, "regulation_id": "32024R1689", "subtree": self._subtree}]


# The graph is consolidated: the inserted paragraph 1a is a REAL node in the
# subtree, in position, tagged with the amending act's CELEX.
_CONSOLIDATED = [
    {"id": "32024R1689_art_6", "depth": 0, "kind": "article", "number": "6",
     "ref": "Article 6", "text": "Classification rules for high-risk AI systems"},
    {"id": "006.001", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 6(1)",
     "text": "1. Base paragraph one, the operative classification condition."},
    {"id": "006.001_pt_a", "depth": 2, "kind": "point", "number": "a", "ref": "Article 6(1), point (a)",
     "text": "the AI system is a safety component;"},
    {"id": "006.001a", "depth": 1, "kind": "paragraph", "number": "1a", "ref": "Article 6(1a)",
     "text": "1a. Inserted paragraph alpha, current law now.", "amended_by": "32026R1744"},
    {"id": "006.002", "depth": 1, "kind": "paragraph", "number": "2", "ref": "Article 6(2)",
     "text": "2. Base paragraph two, the Annex III route."},
]


def test_render_shows_the_consolidated_subtree_in_order():
    out = render_provision_display(_Retriever(_CONSOLIDATED), "Article 6", "32024R1689")
    # base paragraphs are the real corpus text, in order
    assert "Base paragraph one" in out and "Base paragraph two" in out
    # enumerators are bolded so marked / the front-end normaliser cannot mangle them
    assert "**1.**" in out and "**(a)**" in out
    # the inserted paragraph is already in the subtree, rendered in position (before 2)
    assert "Inserted paragraph alpha" in out
    assert out.index("Inserted paragraph alpha") < out.index("Base paragraph two")


def test_render_footer_names_the_amending_act():
    out = render_provision_display(_Retriever(_CONSOLIDATED), "Article 6", "32024R1689")
    assert "Digital Omnibus on AI" in out            # the amending act is named
    assert "As amended" in out                       # provenance footer present
    assert "shown as currently in force" in out


def test_render_without_amendments_is_plain_verbatim():
    plain = [n for n in _CONSOLIDATED if not n.get("amended_by")]
    out = render_provision_display(_Retriever(plain), "Article 6", "32024R1689")
    assert "Base paragraph one" in out
    assert "As amended" not in out                   # no footer when nothing was amended
    assert "currently in force" not in out


def test_render_shows_replaced_text_from_the_consolidated_node():
    # Article 43(3) was replaced by the Omnibus; consolidation put the NEW text on
    # the node (tagged), and the empty depth-1 chapeau (text in a subparagraph) must
    # still render the subparagraph's text.
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "43", "ref": "Article 43",
         "text": "Conformity assessment"},
        {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 43(1)",
         "text": "1. Base paragraph one."},
        {"id": "p3", "depth": 1, "kind": "paragraph", "number": "3", "ref": "Article 43(3)",
         "text": "", "amended_by": "32026R1744"},                       # empty chapeau
        {"id": "p3s1", "depth": 2, "kind": "subparagraph", "number": "1",
         "ref": "Article 43(3), subparagraph 1", "text": "3. New consolidated paragraph three text."},
    ]
    out = render_provision_display(_Retriever(subtree, "Article 43"), "Article 43", "32024R1689")
    assert "New consolidated paragraph three text." in out             # verbatim, not fabricated
    assert "As amended" in out                                          # footer names the act


def test_point_with_inline_cross_reference_keeps_its_leading_marker_bolded():
    # A point (b) that references "point (a)" mid-sentence must keep its OWN leading
    # marker bolded, so the front-end list normaliser cannot split the point in two
    # at the inline reference (the reported rendering bug).
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "6", "ref": "Article 6", "text": "Head"},
        {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1", "ref": "Article 6(1)",
         "text": "1. Both of the following conditions are fulfilled:"},
        {"id": "pa", "depth": 2, "kind": "point", "number": "a", "ref": "Article 6(1), point (a)",
         "text": "the AI system is a safety component of a product listed in Annex I;"},
        {"id": "pb", "depth": 2, "kind": "point", "number": "b", "ref": "Article 6(1), point (b)",
         "text": "the product whose safety component pursuant to point (a) is the AI system is required to undergo assessment."},
    ]
    out = render_provision_display(_Retriever(subtree), "Article 6", "32024R1689")
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
    out = render_provision_display(_Retriever(subtree, "Article 7"), "Article 7", "32024R1689")

    def indent_of(needle):
        for ln in out.splitlines():
            if needle in ln:
                body = ln[2:]                                  # strip "> "
                return len(body) - len(body.lstrip(nbsp))
        return -1

    assert indent_of("**(k)** the extent to which existing Union law") >= 0
    assert indent_of("**(i)** effective measures of redress") > indent_of("**(k)**")
    assert indent_of("**(ii)** effective measures to prevent") > indent_of("**(k)**")


def test_render_returns_none_when_unresolved():
    class _Empty:
        def retrieve_by_refs(self, refs, celex_filter):
            return []
    assert render_provision_display(_Empty(), "Article 6", "32024R1689") is None
