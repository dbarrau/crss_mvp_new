"""Unit coverage for the eval-capture primitive `_fab_counts`.

`ask_stream`'s opt-in `capture` hook records the pre-audit draft and post-audit
final on equal footing so the revision-delta harness can measure what the
audit/revision loop adds. `_fab_counts` turns a faithfulness report into the
structured fabrication figures that comparison keys on. The capture block itself
is exercised end-to-end by `scripts/generate_eval_artifact.py`; here we pin the
pure count extraction (incl. the `None` = check-disabled path).
"""
from types import SimpleNamespace

from application.agent import _fab_counts


def _report(**over) -> SimpleNamespace:
    base = dict(unverified=[], misattributed=[], near_verbatim=[], total_quotes=0)
    base.update(over)
    return SimpleNamespace(**base)


def test_fab_counts_none_reads_all_zero():
    # Faithfulness check disabled / no report → every count is zero, never an error.
    assert _fab_counts(None) == {
        "unverified": 0, "misattributed": 0, "near_verbatim": 0, "total_quotes": 0,
    }


def test_fab_counts_lengths_from_report():
    report = _report(
        unverified=["q1", "q2"],           # absent from corpus (fabricated)
        misattributed=["q3"],              # real text, wrong provision (displaced)
        near_verbatim=["q4", "q5", "q6"],  # minor wording drift
        total_quotes=6,
    )
    counts = _fab_counts(report)
    assert counts == {
        "unverified": 2, "misattributed": 1, "near_verbatim": 3, "total_quotes": 6,
    }
    # The "fabricated quotes" figure the A/B speaks in = unverified + misattributed.
    assert counts["unverified"] + counts["misattributed"] == 3


def test_fab_counts_tolerates_missing_attributes():
    # A report object that predates a field must not raise — missing → 0.
    assert _fab_counts(SimpleNamespace(total_quotes=4)) == {
        "unverified": 0, "misattributed": 0, "near_verbatim": 0, "total_quotes": 4,
    }


def test_fab_counts_none_valued_attributes_read_zero():
    # A field explicitly set to None (not a list) must read as 0, not crash.
    assert _fab_counts(_report(unverified=None, total_quotes=0))["unverified"] == 0
