"""Confidence scoring — the retrieval-relevance component's sentinel handling.

`score` on a provision is either a similarity-ranked value (dense/hybrid hits),
or one of two sentinels that are NOT similarities: 1.0 (direct-ref / structural
match) and 0.0 (graph-expansion, no similarity computed). The relevance
component must exclude BOTH; averaging the 0.0 sentinel as a real cosine
collapsed the score to ~0 on structurally-answered routes and dragged fully
grounded answers to MEDIUM.
"""
from application.agent import compute_confidence


def _score(provisions):
    """Isolate the relevance component via the public entry point."""
    return compute_confidence(
        sufficiency={"checks": [], "ok": True},
        provisions=provisions,
        faith_report=type("F", (), {"total_quotes": 0, "unverified_count": 0})(),
        had_corrective_pass=False,
        had_pointer_expansion=False,
        had_role_provisions=False,
        role_specs=[],
        question="q",
        mentioned_regs=set(),
    )["breakdown"]["retrieval_relevance"]


def test_graph_expansion_sentinels_do_not_tank_relevance():
    # A provision_lookup-style bag: one direct-ref (1.0) + graph-expanded (0.0).
    # No similarity-ranked hits → neutral 0.75, NOT ~0.
    bag = [{"score": 1.0}, {"score": 0.0}, {"score": 0.0}, {"score": 0.0}]
    assert _score(bag) == 0.75


def test_genuine_similarity_scores_are_averaged():
    # Real dense/hybrid hits (strictly between the sentinels) are averaged
    # over the top 3; a trailing 0.0 graph sentinel is ignored, not counted.
    bag = [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}, {"score": 0.0}]
    assert abs(_score(bag) - 0.8) < 1e-9


def test_all_direct_ref_is_neutral():
    assert _score([{"score": 1.0}, {"score": 1.0}]) == 0.75
