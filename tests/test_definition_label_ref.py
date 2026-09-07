"""Definition context labels must cite the *defining point*, not its container.

A formal definition lives in a numbered point (e.g. MDR 'manufacturer' is
Article 2, point (30)), yet ``find_by_term`` historically returned only the
parent article's ``display_ref`` ("Article 2"). The LLM was therefore handed a
coarse label and faithfully echoed the imprecise "Article 2" in prose. The graph
already carries the precise ref on the point node (``point_ref``); the context
formatter must prefer it, falling back to the parent article only when absent.

No Neo4j / LLM — pins the label string built by ``_format_definitions``.
"""
from __future__ import annotations

from application._context import _format_definitions


def test_label_prefers_point_ref_over_parent_article():
    d = [{
        "regulation": "MDR 2017/745",
        "point_ref": "Article 2, point (30)",
        "article_ref": "Article 2",
        "term": "manufacturer",
        "definition_text": "'manufacturer' means a natural or legal person ...",
        "definition_type": "formal",
    }]
    out = _format_definitions(d)
    assert "Article 2, point (30)" in out
    # the coarse parent ref must not stand alone as the citation
    assert "— Article 2 (" not in out


def test_label_falls_back_to_article_ref_when_point_ref_absent():
    d = [{
        "regulation": "MDR 2017/745",
        "article_ref": "Article 2",
        "term": "importer",
        "definition_text": "'importer' means ...",
        "definition_type": "formal",
    }]
    out = _format_definitions(d)
    assert "Article 2 (MDR 2017/745)" in out


def test_empty_point_ref_falls_back_not_blank():
    d = [{
        "regulation": "GDPR 2016/679",
        "point_ref": "",
        "article_ref": "Article 4",
        "term": "personal data",
        "definition_text": "'personal data' means ...",
        "definition_type": "formal",
    }]
    out = _format_definitions(d)
    assert "Article 4 (GDPR 2016/679)" in out
