"""Deterministic tests for the revision-delta grader (scripts/eval_revision_delta.py).

The grader turns the shared artifact's draft/final capture into a per-case
help/hurt verdict. Its heart is pure — the classifier (fabrication-added
dominates), the fab-total, and the per-case scorer — so it is pinned here with
no artifact I/O beyond a synthetic temp file. No Neo4j, no LLM.
"""
import json

from scripts.eval_revision_delta import _classify, _conf_score, _fab_total, _score_case, run


# ── classifier: adding fabrication is always "hurt" ─────────────────────────

def test_classifier_fabrication_added_is_hurt_even_with_cite_gain():
    assert _classify(+0.5, 0.0, +1) == "hurt"     # cited more, but injected a fab quote
    assert _classify(0.0, 0.0, +2) == "hurt"


def test_classifier_correctness_regression_is_hurt():
    assert _classify(-0.5, 0.0, 0) == "hurt"       # dropped a decisive cite
    assert _classify(0.0, -0.34, 0) == "hurt"      # dropped a key fact


def test_classifier_improvement_is_helped():
    assert _classify(+0.5, 0.0, 0) == "helped"     # more correct, no new fab
    assert _classify(0.0, +0.25, 0) == "helped"
    assert _classify(0.0, 0.0, -1) == "helped"     # cleaned up a fabrication


def test_classifier_prose_only_rewrite_is_neutral():
    assert _classify(0.0, 0.0, 0) == "neutral"


# ── primitives ──────────────────────────────────────────────────────────────

def test_fab_total_sums_the_two_redacted_classes():
    assert _fab_total({"unverified": 2, "misattributed": 1, "near_verbatim": 9}) == 3
    assert _fab_total(None) == 0
    assert _fab_total({}) == 0


def test_conf_score_extracts_and_rounds():
    assert _conf_score({"confidence_score": 0.6667}) == 0.667
    assert _conf_score(None) is None
    assert _conf_score({}) is None


def test_score_case_flags_fabricating_revision():
    r = {
        "id": "X", "revised": True, "draft": "foo", "final": "bar",
        "draft_fab": {"unverified": 0, "misattributed": 0},
        "final_fab": {"unverified": 3, "misattributed": 0},
        "draft_confidence": {"confidence_score": 0.8},
        "final_confidence": {"confidence_score": 0.6},
    }
    row = _score_case(r, key=None)
    assert row["verdict"] == "hurt"
    assert row["d_fab"] == 3
    assert row["d_confidence"] == -0.2
    assert row["has_key"] is False


def test_score_case_uses_answer_key_when_present():
    # Final cites the required article; draft does not → Δcite_recall positive.
    r = {
        "id": "Y", "revised": True,
        "draft": "The system must be assessed.",
        "final": "Under Article 43 the system must undergo conformity assessment.",
        "draft_fab": {"unverified": 0}, "final_fab": {"unverified": 0},
    }
    row = _score_case(r, key={"must_cite": ["Article 43"], "must_state": []})
    assert row["has_key"] is True
    assert row["d_cite"] > 0
    assert row["verdict"] == "helped"


# ── end-to-end run() over a synthetic artifact ──────────────────────────────

def test_run_reports_net_negative_when_revision_adds_fabrication(tmp_path, capsys):
    artifact = {
        "meta": {"n": 3},
        "results": [
            # revision INTRODUCED two fabricated quotes → hurt
            {"id": "A", "revised": True, "draft": "d", "final": "f",
             "draft_fab": {"unverified": 0, "misattributed": 0},
             "final_fab": {"unverified": 2, "misattributed": 0}},
            # revision did not fire → excluded from the fired tally by default
            {"id": "B", "revised": False, "draft": "same", "final": "same",
             "draft_fab": {"unverified": 0}, "final_fab": {"unverified": 0}},
            # a case that errored → skipped
            {"id": "C", "error": "TIMEOUT", "answer": ""},
        ],
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))

    summary = run(path, include_all=False, out=None, judge=False,
                  panel_spec=None, judge_model=None, judge_runs=1)

    assert summary["fired"] == 1
    assert summary["scored"] == 1              # only the fired case (B excluded, C errored)
    assert summary["errors"] == ["C"]
    assert summary["net_fab"] == 2
    assert summary["tally"] == {"helped": 0, "hurt": 1, "neutral": 0}
    assert "NET-NEGATIVE" in summary["verdict"]


def test_run_include_all_scores_non_fired_cases(tmp_path):
    artifact = {
        "meta": {"n": 2},
        "results": [
            {"id": "A", "revised": True, "draft": "d", "final": "f",
             "draft_fab": {"unverified": 1}, "final_fab": {"unverified": 0}},   # cleaned up → helped
            {"id": "B", "revised": False, "draft": "s", "final": "s",
             "draft_fab": {"unverified": 0}, "final_fab": {"unverified": 0}},   # neutral
        ],
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))

    summary = run(path, include_all=True, out=None, judge=False,
                  panel_spec=None, judge_model=None, judge_runs=1)
    assert summary["scored"] == 2
    assert summary["tally"]["helped"] == 1     # A removed a fabrication
    assert summary["tally"]["neutral"] == 1    # B unchanged
    assert summary["net_fab"] == -1
