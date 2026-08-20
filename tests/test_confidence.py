"""Confidence scoring — the retrieval-relevance component.

Relevance is scored off the ``cosine`` field: the true query→provision cosine
similarity attached by the dense/hybrid and recital retrieval paths. It is a
genuine 0–1 similarity, unlike ``score``, which is overloaded (an RRF-fused rank
for hybrid hits; the sentinels 1.0 for direct-ref and 0.0 for graph-expansion).
Provisions retrieved structurally carry no ``cosine`` and are excluded; when none
remain the route answered structurally and relevance is neutral (0.75).
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


def test_structural_only_bag_is_neutral_not_zero():
    # provision_lookup-style bag: a direct-ref match and graph-expanded provisions
    # (score sentinels 1.0 / 0.0), none carrying a cosine → neutral 0.75, NOT ~0.
    bag = [{"score": 1.0}, {"score": 0.0}, {"score": 0.0}]
    assert _score(bag) == 0.75


def test_true_cosines_are_averaged_over_top_three():
    # Only the cosine field counts; the RRF-fused `score` is ignored, and a
    # cosine-less graph-expanded provision does not drag the mean.
    bag = [
        {"score": 0.02, "cosine": 0.9},
        {"score": 0.01, "cosine": 0.8},
        {"score": 0.008, "cosine": 0.7},
        {"score": 0.0},  # graph expansion, no cosine → excluded
    ]
    assert abs(_score(bag) - 0.8) < 1e-9


def test_fused_rank_score_is_not_mistaken_for_similarity():
    # A hybrid bag whose `score` is the tiny RRF rank must be scored off `cosine`,
    # not off `score` (which would give ~0.02).
    bag = [{"score": 0.016, "cosine": 0.86}, {"score": 0.015, "cosine": 0.84}]
    assert _score(bag) > 0.8
