"""Unit tests for adjacent-reference de-duplication (the deterministic backstop
to the "citation owns the reference" prompt fix).

When the model names a provision in prose right before a cite pointer/marker,
the render paths drop the pointer's visible reference so it shows once — while
still recording the citation in ``cited_ids`` (audit trail).  Conservative: only
a near, whole-token match suppresses.  Deterministic, no Neo4j / no LLM.
"""
from application._grounded_citation import (
    _ref_already_in_prose,
    build_pointer_index,
    resolve_pointers,
)
from application._grounded_answer import (
    Citation,
    GroundedAnswer,
    render_grounded_answer,
)


def _index():
    return build_pointer_index([
        {
            "article_id": "32024R1689_art_43",
            "article_ref": "Article 43",
            "regulation": "EU AI Act",
            "binding_force": "binding",
            "article_text": "Conformity assessment procedures.",
            "children": [],
        },
        {
            "article_id": "32024R1689_art_4",
            "article_ref": "Article 4",
            "regulation": "EU AI Act",
            "binding_force": "binding",
            "article_text": "AI literacy.",
            "children": [],
        },
    ])


# --- the predicate ----------------------------------------------------------

def test_predicate_matches_adjacent_reference():
    assert _ref_already_in_prose("as required by Article 43 ", "Article 43")


def test_predicate_matches_through_markdown_emphasis():
    assert _ref_already_in_prose("as required by **Article 43** ", "Article 43")


def test_predicate_is_whole_token_not_prefix():
    # "Article 4" must NOT match inside "Article 43".
    assert not _ref_already_in_prose("as required by Article 43 ", "Article 4")


def test_predicate_ignores_far_away_mention():
    far = "Article 43 " + ("x" * 60) + " and further "
    assert not _ref_already_in_prose(far, "Article 43")


def test_predicate_false_on_empty_ref():
    assert not _ref_already_in_prose("Article 43 ", "")


# --- inline path ------------------------------------------------------------

def test_inline_suppresses_duplicate_reference():
    res = resolve_pointers(
        "The provider must run the procedure in Article 43 [cite: 32024R1689_art_43].",
        _index(),
    )
    # "Article 43" appears exactly once; the pointer's copy is dropped.
    assert res.text.count("Article 43") == 1
    assert "32024R1689_art_43" in res.cited_ids            # still attributed
    assert "32024R1689_art_43" in res.suppressed_ref_dups


def test_inline_renders_reference_when_not_in_prose():
    res = resolve_pointers(
        "The provider must run the required conformity procedure "
        "[cite: 32024R1689_art_43].",
        _index(),
    )
    assert "Article 43 EU AI Act" in res.text
    assert res.suppressed_ref_dups == []


# --- structured path --------------------------------------------------------

def test_structured_suppresses_duplicate_reference():
    ans = GroundedAnswer(
        body="The provider must run the procedure in Article 43 [[c1]].",
        citations=[Citation(marker="c1", node_id="32024R1689_art_43", mode="cite")],
    )
    res = render_grounded_answer(ans, _index())
    assert res.text.count("Article 43") == 1
    assert "32024R1689_art_43" in res.cited_ids
    assert "32024R1689_art_43" in res.suppressed_ref_dups


def test_structured_renders_reference_when_not_in_prose():
    ans = GroundedAnswer(
        body="The system must bear the CE marking [[c1]].",
        citations=[Citation(marker="c1", node_id="32024R1689_art_43", mode="cite")],
    )
    res = render_grounded_answer(ans, _index())
    assert "Article 43 EU AI Act" in res.text
    assert res.suppressed_ref_dups == []


# --- global reference-map fallback ------------------------------------------

_GLOBAL = {
    "32024R1689_art_25": ("Article 25", "EU AI Act"),
    "32024R1689_art_47": ("Article 47", "EU AI Act"),
}


def test_inline_unretrieved_but_real_provision_renders_human_reference():
    # art_25 is NOT in the retrieved index, but IS a real provision.
    res = resolve_pointers(
        "A modifier may become a provider [cite: 32024R1689_art_25].",
        _index(), fallback_refs=_GLOBAL,
    )
    assert "Article 25 EU AI Act" in res.text
    assert "32024R1689_art_25" in res.global_ref_ids
    assert res.unresolved_ids == []
    assert "DROP" not in res.text


def test_structured_unretrieved_but_real_provision_renders_human_reference():
    ans = GroundedAnswer(
        body="A modifier may become a provider [[c1]].",
        citations=[Citation(marker="c1", node_id="32024R1689_art_47", mode="cite")],
    )
    res = render_grounded_answer(ans, _index(), fallback_refs=_GLOBAL)
    assert "Article 47 EU AI Act" in res.text
    assert "32024R1689_art_47" in res.global_ref_ids


# --- husk cleanup for genuinely-nonexistent ids -----------------------------

def test_nonexistent_id_is_dropped_without_empty_husk():
    # art_23_g is invented — exists in neither the bag nor the global map.
    res = resolve_pointers(
        "This obligation ensures traceability, as required by "
        "**[cite: 32024R1689_art_23_g]**.",
        _index(), fallback_refs=_GLOBAL,
    )
    assert "****" not in res.text                    # no empty bold husk
    assert "DROP" not in res.text                    # sentinel never leaks
    assert "32024R1689" not in res.text              # no raw node id
    assert "32024R1689_art_23_g" in res.unresolved_ids
    # the dangling "as required by" connector is cleaned; sentence still closes
    assert "as required by" not in res.text
    assert res.text.rstrip().endswith("traceability.")


def test_canonicalizes_fabricated_paragraph_id_to_real_reference():
    # The model invents `art_23_1` (paragraph 1); the real id is `023.001`.
    # Canonicalising against the global map recovers a human reference.
    from application._grounded_citation import _canonicalize_id
    assert _canonicalize_id("32024R1689_art_23_1") == "32024R1689_023.001"
    assert _canonicalize_id("32024R1689_art_10a_2") == "32024R1689_010a.002"
    assert _canonicalize_id("32024R1689_art_23_pt_g") is None  # not a paragraph

    glob = {"32024R1689_023.001": ("Article 23(1)", "EU AI Act")}
    res = resolve_pointers(
        "Importers must verify provider compliance [cite: 32024R1689_art_23_1].",
        _index(), fallback_refs=glob,
    )
    assert "Article 23(1) EU AI Act" in res.text
    assert "32024R1689_023.001" in res.global_ref_ids
    assert res.unresolved_ids == []
    assert "DROP" not in res.text


def test_child_cite_renders_article_anchored_not_bare_paragraph():
    # A cite to a paragraph node must read "Article 23(1)", not the child's bare
    # display_ref "Paragraph 1" (which reads like a decontextualised variable).
    from application._grounded_citation import _human_ref_from_id
    assert _human_ref_from_id("32024R1689_023.001") == "Article 23(1)"
    assert _human_ref_from_id("32024R1689_023.001_pt_a") == "Article 23(1)(a)"
    assert _human_ref_from_id("32024R1689_art_23") == "Article 23"
    assert _human_ref_from_id("32024R1689_rct_81") is None  # falls back to display_ref

    idx = build_pointer_index([{
        "article_id": "32024R1689_art_23",
        "article_ref": "Article 23",
        "regulation": "EU AI Act",
        "article_text": "Importer obligations.",
        "children": [{
            "id": "32024R1689_023.005",
            "ref": "Paragraph 5",  # bare, decontextualised
            "text": "Importers shall keep documentation for 10 years.",
            "binding_force": "binding",
        }],
    }])
    res = resolve_pointers(
        "Importers must retain documentation [cite: 32024R1689_023.005].", idx,
    )
    assert "Article 23(5) EU AI Act" in res.text
    assert "Paragraph 5" not in res.text


def test_bare_bracket_reference_is_unwrapped():
    # The model writes "[Article 11]" as an incomplete markdown link; it must
    # render as plain "Article 11", not literal brackets.
    res = resolve_pointers(
        "Technical documentation per [Article 11] and [Annex IV].", _index(),
    )
    assert "Article 11" in res.text and "[Article 11]" not in res.text
    assert "Annex IV" in res.text and "[Annex IV]" not in res.text


def test_markdown_link_citation_resolves_and_never_leaks_id():
    # The model sometimes wraps the pointer as a markdown link, which would leak
    # the internal id as a URL. It must resolve to the human reference instead.
    res = resolve_pointers(
        "The procedure applies [Article 43](cite:32024R1689_art_43).",
        _index(),
    )
    assert "Article 43 EU AI Act" in res.text
    assert "32024R1689_art_43" not in res.text        # internal id never shown
    assert "cite:" not in res.text


def test_markdown_link_with_unresolvable_id_keeps_label_not_id():
    res = resolve_pointers(
        "See [Article 999](cite:32024R1689_art_999).",
        _index(),
    )
    assert "Article 999" in res.text                  # human label kept
    assert "32024R1689" not in res.text               # id dropped
    assert "cite:" not in res.text


def test_husk_sweep_strips_leaked_citation_url_from_structured_body():
    # Defensive backstop: even a raw citation-URL in a body is de-id'd.
    ans = GroundedAnswer(
        body="The provider must comply [Article 16](cite:32024R1689_art_16_x).",
        citations=[],
    )
    res = render_grounded_answer(ans, _index())
    assert "Article 16" in res.text
    assert "32024R1689" not in res.text
    assert "cite:" not in res.text


def test_empty_blockquote_line_removed_when_only_citation_dropped():
    res = resolve_pointers(
        "Heading\n\n> [cite: 32024R1689_art_23_g]\n\nNext paragraph.",
        _index(), fallback_refs=_GLOBAL,
    )
    assert "****" not in res.text
    assert "DROP" not in res.text
    # no orphaned "> " line left behind
    import re as _re
    assert not _re.search(r"(?m)^>[ \t]*$", res.text)
