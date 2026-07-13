"""The critical-defect gate in the answer-quality eval.

The gate is deterministic and must be able to overrule the judge: a fabricated
or misattributed quote makes an answer unusable for compliance reliance no
matter how the prose was scored (observed failure: 8.5/10 "strong first draft"
with 4 fabricated quotes).
"""
from scripts.eval_answer_quality import (
    _apply_reliance_gate,
    _critical_defects,
    _majority_reliance,
    _verification_block,
)


def _result(score=8.5, reliance="Can be used as a strong first draft",
            fab=0, mis=0):
    return {
        "score": score,
        "reliance": reliance,
        "faithfulness": {
            "fabricated": fab,
            "misattributed": mis,
            "near_verbatim": 0,
            "total_quotes": fab + mis,
        },
    }


def test_clean_case_has_no_defects_and_keeps_judge_verdict():
    r = _result()
    _apply_reliance_gate(r)
    assert r["critical_defects"] == []
    assert r["reliance_final"] == "Can be used as a strong first draft"
    assert r["reliance_gated"] is False


def test_fabricated_quote_downgrades_a_positive_judge_verdict():
    # The HQ_020 failure mode: strong-first-draft verdict over fabricated quotes.
    r = _result(fab=4)
    _apply_reliance_gate(r)
    assert "fabricated_quotes=4" in r["critical_defects"]
    assert r["reliance_final"] == "Cannot be relied on without major revision"
    assert r["reliance_gated"] is True
    assert r["score"] == 8.5  # raw judge score stays for run-to-run comparability


def test_misattributed_quote_also_gates():
    r = _result(mis=1)
    _apply_reliance_gate(r)
    assert r["reliance_gated"] is True


def test_judge_unreliable_verdict_is_a_critical_defect_without_gating():
    # Judge already said unreliable — counts as defective, no downgrade needed.
    r = _result(score=6.0, reliance="Cannot be relied on without major revision")
    _apply_reliance_gate(r)
    assert r["critical_defects"] == ["judge_verdict_unreliable"]
    assert r["reliance_gated"] is False


def test_ungraded_case_counts_as_defective():
    r = _result(score=None, reliance="(timeout)")
    _apply_reliance_gate(r)
    assert "ungraded" in r["critical_defects"]


def test_quote_defect_on_unreliable_verdict_keeps_verdict():
    r = _result(score=6.0, reliance="Unsafe for compliance reliance", fab=2)
    _apply_reliance_gate(r)
    assert r["reliance_final"] == "Unsafe for compliance reliance"
    assert r["reliance_gated"] is False
    assert "fabricated_quotes=2" in r["critical_defects"]


def test_majority_reliance_votes_across_runs():
    strong = "Can be used as a strong first draft"
    weak = "Cannot be relied on without major revision"
    assert _majority_reliance([strong, strong, weak]) == strong
    assert _majority_reliance([weak, weak, strong]) == weak


def test_majority_reliance_tie_breaks_to_worse_verdict():
    strong = "Can be used as a strong first draft"
    weak = "Cannot be relied on without major revision"
    assert _majority_reliance([strong, weak]) == weak
    # Unparsed runs don't dilute the vote among real verdicts
    assert _majority_reliance(["(unparsed)", strong, strong]) == strong


def test_majority_reliance_all_unparsed_falls_back():
    assert _majority_reliance(["(unparsed)", "(timeout)"]) == "(timeout)"
    assert _majority_reliance([]) == "(unparsed)"


def test_verification_block_renders_counts():
    block = _verification_block(
        {"fabricated": 2, "misattributed": 1, "near_verbatim": 3, "total_quotes": 6}
    )
    assert "FABRICATED quotes" in block and ": 2" in block
    assert "MISATTRIBUTED quotes" in block and ": 1" in block
    assert "CITATION VERIFICATION" in block


def test_defect_list_is_stable_for_summary_grouping():
    assert _critical_defects(_result(fab=1, mis=1)) == [
        "fabricated_quotes=1",
        "misattributed_quotes=1",
    ]


def test_answer_key_failure_is_critical_and_gates_the_verdict():
    r = _result()
    r["answer_key_check"] = {
        "passed": False,
        "cite_recall": 0.5,
        "state_recall": 1.0,
        "missed_cites": ["Article 26"],
        "missed_states": [],
        "violations": [],
    }
    _apply_reliance_gate(r)
    assert "answer_key_failed(Article 26)" in r["critical_defects"]
    assert r["reliance_final"] == "Cannot be relied on without major revision"
    assert r["reliance_gated"] is True


def test_answer_key_pass_does_not_gate():
    r = _result()
    r["answer_key_check"] = {
        "passed": True, "cite_recall": 1.0, "state_recall": 1.0,
        "missed_cites": [], "missed_states": [], "violations": [],
    }
    _apply_reliance_gate(r)
    assert r["critical_defects"] == []
    assert r["reliance_gated"] is False


def test_missing_key_check_field_is_tolerated():
    # Cases without an answer_key (and legacy result files) have no check dict.
    r = _result()
    r["answer_key_check"] = None
    _apply_reliance_gate(r)
    assert r["critical_defects"] == []
