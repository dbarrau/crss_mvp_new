"""Tests for the answer-key currency audit (guards against keys encoding stale law).

Two jobs:

* the **live gate** — the real ``eval/quality_set.json`` must have no STALE key,
  so a key that rots after an amending act enters the corpus (a superseded date,
  a deleted-provision citation, an inserted provision penalised as a fabrication)
  fails CI instead of silently producing false eval fails;
* the **detectors** — each check must keep firing on the bug it exists for, and
  must NOT false-fire on the parent of a deleted sub-provision (the substring
  over-match that the first draft shipped and we fixed).

Detector cases inject ``inserted`` so they are deterministic without the
gitignored ``data/`` parse present; the live gate uses the real parse when it is
on disk (Check C arms itself) and still guards dates + deletions when it is not.
"""
import json
from pathlib import Path

from scripts.audit_answer_key_currency import (
    OMNIBUS_DATE_CHANGES,
    SUPERSEDED,
    STALE,
    audit,
)

QUALITY_SET = Path(__file__).resolve().parent.parent / "eval" / "quality_set.json"

_NO_INSERT: dict[str, set[str]] = {}  # detectors that don't exercise Check C


def _stale(findings):
    return [f for f in findings if f["severity"] == STALE]


def test_live_quality_set_has_no_stale_keys():
    questions = json.loads(QUALITY_SET.read_text(encoding="utf-8"))
    stale = _stale(audit(questions))
    assert not stale, "stale answer keys (run scripts/audit_answer_key_currency.py):\n" + \
        "\n".join(f"  {f['id']}: {f['message']}" for f in stale)


def test_detects_superseded_date_in_scope():
    dc = OMNIBUS_DATE_CHANGES[0]  # Annex III high-risk: 2 Aug 2026 → 2 Dec 2027
    q = {
        "id": "T_DATE",
        "question": "When do Annex III high-risk obligations apply?",
        "answer_key": {"must_cite": [], "must_state": [[dc.superseded]], "must_not_claim": []},
    }
    stale = _stale(audit([q], inserted=_NO_INSERT))
    assert [f["id"] for f in stale] == ["T_DATE"]
    assert dc.current in stale[0]["message"]


def test_superseded_date_out_of_scope_does_not_fire():
    # 2 August 2026 is also the (unchanged) GENERAL application date; with no
    # high-risk scope marker present, requiring it must NOT be flagged.
    q = {
        "id": "T_GENERAL",
        "question": "What is the general application date of the AI Act?",
        "answer_key": {"must_cite": [], "must_state": [["2 August 2026"]], "must_not_claim": []},
    }
    assert not _stale(audit([q], inserted=_NO_INSERT))


def test_detects_required_citation_of_deleted_provision():
    deleted = SUPERSEDED[0]["deleted_ref"]  # "Article 10(5)"
    q = {
        "id": "T_DEL",
        "question": "irrelevant",
        "answer_key": {"must_cite": [deleted], "must_state": [], "must_not_claim": []},
    }
    stale = _stale(audit([q], inserted=_NO_INSERT))
    assert [f["id"] for f in stale] == ["T_DEL"]


def test_parent_of_deleted_subprovision_is_not_flagged():
    # Article 10(5) was deleted; Article 10 (its parent) is still live. Requiring
    # "Article 10" must not flag — this is the substring over-match we fixed.
    q = {
        "id": "T_PARENT",
        "question": "core manufacturer obligations",
        "answer_key": {"must_cite": ["Article 10"], "must_state": [], "must_not_claim": []},
    }
    assert not _stale(audit([q], inserted=_NO_INSERT))


def test_detects_inserted_provision_penalised_as_fabrication():
    q = {
        "id": "T_INS",
        "question": "obligations for a non-high-risk provider",
        "answer_key": {
            "must_cite": [],
            "must_state": [],
            # 4a was inserted (now real) → must flag; 4b was never enacted → keep.
            "must_not_claim": ["article 4a", "article 4b"],
        },
    }
    stale = _stale(audit([q], inserted={"any": {"article 4a"}}))
    assert len(stale) == 1
    assert stale[0]["id"] == "T_INS"
    assert "4a" in stale[0]["message"] and "4b" not in stale[0]["message"]
