"""Structured-lookup completeness: the direct subject of a provision lookup must
render ALL its children in document order, so a "what does Annex III list?"
question is answerable from context.

Regression guard for the Annex III bug: the retrieval Cypher's ``collect()``
scrambled child order and the renderer capped children at ``_MAX_CHILD_LINES``,
so ~20 of 32 sub-items never reached the LLM (it dropped e.g. the elections
sub-item). Peripheral provisions stay capped to bound the prompt. No Neo4j / LLM.
"""
from __future__ import annotations

from application._context import (
    _MAX_CHILD_LINES,
    _format_one_provision,
    _natural_key,
)
from domain.legislation_catalog import AI_ACT_CELEX


def _provision_with_children(n: int, *, direct: bool, pointer: bool = False) -> dict:
    # Children supplied in SCRAMBLED order (as the retrieval collect() returns them).
    order = list(range(1, n + 1))
    scrambled = order[::2] + order[1::2]
    return {
        "article_id": f"{AI_ACT_CELEX}_anx_III",
        "celex": AI_ACT_CELEX,
        "regulation": "EU AI Act",
        "article_ref": "Annex III",
        "article_text": "High-risk AI systems.",
        "children": [
            {"id": f"{AI_ACT_CELEX}_anx_III_{i}", "ref": f"Annex point {i}",
             "text": f"Area {i} description.", "binding_force": "binding"}
            for i in scrambled
        ],
        "_direct_ref_match": direct,
        "_pointer_expansion": pointer,
        "provision_role": "SCOPE",
    }


def test_natural_key_orders_numeric_and_lettered_ids():
    ids = [f"x_{s}" for s in ("2", "10", "1", "1_b", "1_a", "3")]
    assert sorted(ids, key=_natural_key) == [
        "x_1", "x_1_a", "x_1_b", "x_2", "x_3", "x_10",
    ]


def test_direct_subject_renders_all_children_in_document_order():
    p = _provision_with_children(20, direct=True)
    out = _format_one_provision(1, p, "SCOPE")
    # every one of the 20 points is present …
    for i in range(1, 21):
        assert f"Annex point {i} (id:" in out, f"point {i} missing from subject render"
    # … and in ascending document order (1 before 2 before … before 20)
    positions = [out.index(f"Annex point {i} (id:") for i in range(1, 21)]
    assert positions == sorted(positions)


def test_peripheral_provision_stays_capped():
    # No direct-ref flag → capped (bounds prompt bloat on multi-provision routes).
    p = _provision_with_children(20, direct=False)
    out = _format_one_provision(1, p, "SCOPE")
    shown = sum(1 for i in range(1, 21) if f"Annex point {i} (id:" in out)
    assert shown == _MAX_CHILD_LINES


def test_pointer_expanded_subject_stays_capped():
    # A cross-referenced provision is a direct-ref match but pointer-expanded —
    # peripheral to the question, so it must NOT get the full-render treatment.
    p = _provision_with_children(20, direct=True, pointer=True)
    out = _format_one_provision(1, p, "SCOPE")
    shown = sum(1 for i in range(1, 21) if f"Annex point {i} (id:" in out)
    assert shown == _MAX_CHILD_LINES
