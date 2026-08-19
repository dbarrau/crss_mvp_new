"""Unit tests for the grounded-citation resolver (application/_grounded_citation).

Deterministic, no Neo4j / no LLM. Fixtures mirror the real retriever dict shape
(``article_id`` on provisions, ``id`` on children) confirmed against a live
retrieval trace.
"""
from application._grounded_citation import (
    _clean_husks,
    build_pointer_index,
    resolve_pointers,
)


def _provisions():
    return [
        {
            "article_id": "32017R0745_art_10",
            "article_ref": "Article 10",
            "regulation": "MDR 2017/745",
            "binding_force": "binding",
            "article_text": "Manufacturers shall ensure that devices are designed "
            "and manufactured in accordance with the requirements of this "
            "Regulation.",
            "children": [
                {
                    "id": "32017R0745_010.014",
                    "ref": "Paragraph 14",
                    "text": "Manufacturers shall keep the technical documentation "
                    "up to date.",
                    "binding_force": "binding",
                },
            ],
        },
        {
            "article_id": "MDCG_2019_11_s3",
            "article_ref": "Section 3",
            "regulation": "MDCG 2019-11",
            "binding_force": "non_binding",
            "article_text": "Software which drives or influences the use of a "
            "hardware medical device.",
            "children": [],
        },
    ]


def test_index_keys_are_node_ids_not_display_refs():
    idx = build_pointer_index(_provisions())
    assert set(idx) == {
        "32017R0745_art_10",
        "32017R0745_010.014",
        "MDCG_2019_11_s3",
    }


def test_quote_pointer_renders_verbatim_stored_text():
    idx = build_pointer_index(_provisions())
    out = resolve_pointers("The rule: [quote: 32017R0745_010.014]", idx)
    assert "> Manufacturers shall keep the technical documentation up to date." in out.text
    assert out.quoted_ids == ["32017R0745_010.014"]
    # The model never authored the quoted words; they came from the index.
    assert out.unresolved_ids == []


def test_cite_pointer_renders_human_ref_no_quote():
    idx = build_pointer_index(_provisions())
    out = resolve_pointers("Per [cite: 32017R0745_art_10], this applies.", idx)
    assert "Per Article 10 MDR 2017/745, this applies." == out.text
    assert out.cited_ids == ["32017R0745_art_10"]
    assert ">" not in out.text


def test_unresolved_pointer_is_dropped_not_rendered():
    idx = build_pointer_index(_provisions())
    out = resolve_pointers(
        "Fabricated claim [quote: 99999X_art_1_fake] here.", idx
    )
    assert "99999X_art_1_fake" not in out.text
    assert ">" not in out.text  # never rendered as an unsupported quote
    assert out.unresolved_ids == ["99999X_art_1_fake"]
    assert "Fabricated claim here." == out.text


def test_multiline_quote_prefixes_every_line():
    provs = [
        {
            "article_id": "X_art_1",
            "article_ref": "Article 1",
            "regulation": "REG",
            "binding_force": "binding",
            "article_text": "First line.\nSecond line.",
            "children": [],
        }
    ]
    idx = build_pointer_index(provs)
    out = resolve_pointers("[quote: X_art_1]", idx)
    assert out.text == "> First line.\n> Second line."


def test_empty_text_node_falls_back_to_ref_not_empty_quote():
    provs = [
        {
            "article_id": "X_cpt_II",
            "article_ref": "Chapter II",
            "regulation": "REG",
            "binding_force": "binding",
            "article_text": "",
            "children": [],
        }
    ]
    idx = build_pointer_index(provs)
    out = resolve_pointers("[quote: X_cpt_II]", idx)
    assert out.text == "Chapter II REG"
    assert ">" not in out.text


def test_shared_child_id_keeps_first_binding():
    # A Chapter ancestor repeats across provisions; first binding wins, leaf
    # paragraphs (unique) are unaffected.
    provs = _provisions() + [
        {
            "article_id": "32017R0745_art_11",
            "article_ref": "Article 11",
            "regulation": "MDR 2017/745",
            "binding_force": "binding",
            "article_text": "Article 11 body.",
            "children": [
                {
                    "id": "32017R0745_010.014",  # duplicate leaf id
                    "ref": "Paragraph 14 (dup)",
                    "text": "DIFFERENT TEXT",
                    "binding_force": "binding",
                }
            ],
        }
    ]
    idx = build_pointer_index(provs)
    assert idx["32017R0745_010.014"].ref == "Paragraph 14"  # first binding kept


def test_quote_marker_on_own_bullet_leaves_no_empty_list_item():
    """A `[quote:]` on its own "- " bullet must not orphan an empty list item.

    `_render_quote` lifts the quote onto standalone lines (placement-independent
    by design), which empties the bullet.  Left in, marked.js renders it as a
    bare empty <li> (the "empty bullet points" seen in a numbered list where each
    item carries a quote).  The husk cleaner must drop the emptied bullet.
    """
    import re

    idx = build_pointer_index(_provisions())
    answer = (
        "The relevant articles:\n\n"
        "1. **Article 10** – keeps documentation current.\n"
        "   - [quote: 32017R0745_010.014]\n"
    )
    out = resolve_pointers(answer, idx).text
    # the quote still renders as a blockquote ...
    assert "> Manufacturers shall keep the technical documentation up to date." in out
    # ... and no line is left as a content-less list bullet
    assert not any(re.fullmatch(r"[ \t]*[-*+][ \t]*", ln) for ln in out.split("\n")), out


# --- Heading/next-line glue (generation-side, not a citation-resolution bug) --
# Confirmed against real production output (eval/runs/artifact_v1.json, Aug
# 2026): the model occasionally emits a heading with no newline before the
# sentence that follows it. draft and final answer were byte-identical on the
# glued line in every observed case, so resolve_pointers never touched it —
# the repair belongs in _clean_husks, not upstream. Two real shapes: the tail
# reads as ordinary prose, and the tail opens with a citation (digits right
# after the glue word), which an earlier length-gated, prose-only pattern
# missed entirely (see git history on this test's sibling in _grounded_citation).

def test_clean_husks_splits_heading_glued_to_citation_opening_sentence():
    # real case: "### 3. Treatment of Pre-Determined ChangesArticle 43(4), ..."
    line = ("### 3. Treatment of Pre-Determined ChangesArticle 43(4), second "
            "subparagraph addresses pre-determined changes:")
    out = _clean_husks(line)
    assert out == (
        "### 3. Treatment of Pre-Determined Changes\n\n"
        "Article 43(4), second subparagraph addresses pre-determined changes:"
    )


def test_clean_husks_splits_short_heading_glued_to_next_line():
    # real case: "### 3. Human OversightGoverning provision: Article 26(2)"
    # — only 56 chars; the old fix required len(line) > 70 and missed this.
    line = "### 3. Human OversightGoverning provision: Article 26(2)"
    out = _clean_husks(line)
    assert out == "### 3. Human Oversight\n\nGoverning provision: Article 26(2)"


def test_clean_husks_leaves_normal_headings_alone():
    for line in [
        "### Article 10 — Technical documentation",
        "## Classification of In Vitro Diagnostic Devices (IVDs) as Class B vs. Class C under IVDR 2017/746",
        "### eIDAS-Regulation cross-reference",   # no lower->upper->lower glue
    ]:
        assert _clean_husks(line) == line


def test_resolve_pointers_repairs_glued_heading_end_to_end():
    idx = build_pointer_index(_provisions())
    answer = ("### 3. Treatment of Pre-Determined ChangesArticle 43(4) states: "
               "[cite: 32017R0745_art_10]")
    out = resolve_pointers(answer, idx).text
    assert "### 3. Treatment of Pre-Determined Changes\n\n" in out
    assert "Article 43(4) states:" in out
