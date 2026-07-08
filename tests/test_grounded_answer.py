"""Unit tests for the structured grounded-answer renderer.

Deterministic, no Neo4j / no LLM. Exercises the [[marker]] -> verbatim/ref
substitution and the drop-and-report behaviour for unresolved markers/ids.
"""
from application._grounded_answer import (
    Citation,
    GroundedAnswer,
    build_pointer_index,
    render_grounded_answer,
)


def _index():
    return build_pointer_index(
        [
            {
                "article_id": "32017R0745_art_10",
                "article_ref": "Article 10",
                "regulation": "MDR 2017/745",
                "binding_force": "binding",
                "article_text": "Manufacturers shall ensure conformity.",
                "children": [
                    {
                        "id": "32017R0745_010.014",
                        "ref": "Paragraph 14",
                        "text": "Manufacturers shall keep documentation up to date.",
                        "binding_force": "binding",
                    }
                ],
            }
        ]
    )


def test_quote_marker_renders_verbatim():
    ans = GroundedAnswer(
        body="The duty is clear: [[q1]].",
        citations=[Citation(marker="q1", node_id="32017R0745_010.014", mode="quote")],
    )
    out = render_grounded_answer(ans, _index())
    assert "> Manufacturers shall keep documentation up to date." in out.text
    assert out.quoted_ids == ["32017R0745_010.014"]


def test_cite_marker_renders_ref_no_quote():
    ans = GroundedAnswer(
        body="This is governed by [[c1]].",
        citations=[Citation(marker="c1", node_id="32017R0745_art_10", mode="cite")],
    )
    out = render_grounded_answer(ans, _index())
    assert out.text == "This is governed by Article 10 MDR 2017/745."
    assert out.cited_ids == ["32017R0745_art_10"]
    assert ">" not in out.text


def test_unknown_marker_is_dropped_and_reported():
    ans = GroundedAnswer(body="Dangling [[q9]] token.", citations=[])
    out = render_grounded_answer(ans, _index())
    assert "q9" not in out.text
    assert out.text == "Dangling token."
    assert out.unresolved_markers == ["q9"]


def test_citation_with_id_not_in_bag_is_dropped():
    ans = GroundedAnswer(
        body="Claim [[q1]] here.",
        citations=[Citation(marker="q1", node_id="99999X_art_1_fake", mode="quote")],
    )
    out = render_grounded_answer(ans, _index())
    assert ">" not in out.text
    assert out.text == "Claim here."
    assert out.unresolved_ids == ["99999X_art_1_fake"]


def test_multiple_markers_resolve_independently():
    ans = GroundedAnswer(
        body="Per [[c1]], and specifically [[q1]].",
        citations=[
            Citation(marker="c1", node_id="32017R0745_art_10", mode="cite"),
            Citation(marker="q1", node_id="32017R0745_010.014", mode="quote"),
        ],
    )
    out = render_grounded_answer(ans, _index())
    assert "Article 10 MDR 2017/745" in out.text
    assert "> Manufacturers shall keep documentation up to date." in out.text
    assert out.cited_ids == ["32017R0745_art_10"]
    assert out.quoted_ids == ["32017R0745_010.014"]
