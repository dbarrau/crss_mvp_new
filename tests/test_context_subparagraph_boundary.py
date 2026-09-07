"""The model-facing context render must keep the paragraph boundary visible.

Regression guard for the paraphrase-misattribution root cause (task 2): the
direct-subject subtree renderer (`application/_context._render_subtree`) dropped
a paragraph whose text lives entirely in its subparagraphs — its number vanished
— and then labelled each subparagraph with a fabricated "(n)". So MDR Article
33(5)'s "user-friendly and easily-searchable" subparagraph reached the model as a
bare "(2)" sitting under "(4)", and answers attributed it to Article 33(4). The
fix mirrors `_display._render_nodes`, which already preserves this boundary for
the verbatim view. No Neo4j / LLM.
"""
from __future__ import annotations

from application._context import _render_subtree


def _mdr_article_33_subtree() -> list[dict]:
    # Shape of MDR Article 33 as the retriever hands it to the renderer: paragraph
    # (4) has its own text; paragraph (5) is empty (its text is split across two
    # subparagraphs, sp1/sp2 — sp2 is the "user-friendly" duty).
    return [
        {"id": "art", "depth": 0, "kind": "article", "number": "33",
         "ref": "Article 33", "text": "European database on medical devices"},
        {"id": "p4", "depth": 1, "kind": "paragraph", "number": "4",
         "ref": "Article 33(4)",
         "text": "The data shall be entered into Eudamed by the Member States, "
                 "notified bodies, economic operators and sponsors."},
        {"id": "p5", "depth": 1, "kind": "paragraph", "number": "5",
         "ref": "Article 33(5)", "text": ""},
        {"id": "p5s1", "depth": 2, "kind": "subparagraph", "number": "1",
         "ref": "Article 33(5), subparagraph 1",
         "text": "All the information collated and processed by Eudamed shall be "
                 "accessible to the Member States and to the Commission."},
        {"id": "p5s2", "depth": 2, "kind": "subparagraph", "number": "2",
         "ref": "Article 33(5), subparagraph 2",
         "text": "The Commission shall ensure that public parts of Eudamed are "
                 "presented in a user-friendly and easily-searchable format."},
    ]


def test_empty_paragraph_keeps_its_number_as_a_boundary():
    out = _render_subtree(_mdr_article_33_subtree())
    lines = out.splitlines()
    # (5) is present as its own boundary line, at the SAME indent as (4) — a
    # sibling paragraph, not a sub-item of (4).
    para_lines = {ln.strip(): ln for ln in lines}
    assert "(5)" in para_lines, out
    assert para_lines["(5)"].startswith("  (5)")            # depth-1 indent
    p4 = next(ln for ln in lines if ln.strip().startswith("(4)"))
    p5 = para_lines["(5)"]
    # same leading indent → siblings
    assert len(p4) - len(p4.lstrip()) == len(p5) - len(p5.lstrip())


def test_subparagraph_gets_no_fabricated_number():
    out = _render_subtree(_mdr_article_33_subtree())
    # The old bug rendered subparagraph 2 as "(2)"; that literal must be gone.
    assert "(2)" not in out, out
    # The user-friendly text is still present, and indented UNDER (5).
    uf = next(ln for ln in out.splitlines() if "user-friendly" in ln)
    p5 = next(ln for ln in out.splitlines() if ln.strip() == "(5)")
    assert (len(uf) - len(uf.lstrip())) > (len(p5) - len(p5.lstrip())), out


def test_user_friendly_text_is_grouped_under_five_not_four():
    out = _render_subtree(_mdr_article_33_subtree())
    lines = out.splitlines()
    idx_uf = next(i for i, ln in enumerate(lines) if "user-friendly" in ln)
    # the nearest paragraph enumerator ABOVE the user-friendly line is (5), not (4)
    para_above = next(
        ln.strip() for ln in reversed(lines[:idx_uf])
        if ln.strip().startswith("(") and ln.strip()[1:2].isdigit()
    )
    assert para_above == "(5)", out


def test_numbered_points_still_get_their_letter_labels():
    # A paragraph WITH its own text plus lettered points must be unchanged.
    subtree = [
        {"id": "art", "depth": 0, "kind": "article", "number": "33",
         "ref": "Article 33", "text": "Head"},
        {"id": "p1", "depth": 1, "kind": "paragraph", "number": "1",
         "ref": "Article 33(1)", "text": "The Commission shall set up Eudamed for:"},
        {"id": "p1a", "depth": 2, "kind": "point", "number": "a",
         "ref": "Article 33(1), point (a)", "text": "to inform the public;"},
    ]
    out = _render_subtree(subtree)
    assert "(1) The Commission shall set up" in out
    assert "(a) to inform the public;" in out
