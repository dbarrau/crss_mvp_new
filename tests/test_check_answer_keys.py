"""Tests for the deterministic answer-key checker (law-grounded eval signal).

Pins the matching logic so the objective correctness gate cannot silently drift:
citation matching is whole-reference (no false positives on "Article 530"), a
missing decisive provision is a hard fail, and trap phrasings are flagged.
"""
from scripts.check_answer_keys import check_answer, _cite_pattern


def test_citation_matching_is_whole_reference():
    assert _cite_pattern("Article 53").search("see Article 53(1) AI Act")
    assert _cite_pattern("Annex IX").search("via Annex IX, Chapter I")
    # must NOT match a longer number
    assert not _cite_pattern("Article 9").search("processing under Article 90 GDPR")


def test_full_pass_when_cites_and_facts_present():
    key = {
        "must_cite": ["Article 53", "Article 55"],
        "must_state": [["systemic risk"], ["adversarial", "model evaluation"]],
        "must_not_claim": [],
    }
    ans = ("A provider of a GPAI model with systemic risk must, under Article 53 and "
           "Article 55, carry out adversarial testing and risk mitigation.")
    v = check_answer(ans, key)
    assert v["passed"]
    assert v["cite_recall"] == 1.0 and v["state_recall"] == 1.0


def test_missing_decisive_provision_is_a_hard_fail():
    key = {"must_cite": ["Article 53", "Article 55"], "must_state": [], "must_not_claim": []}
    v = check_answer("Only Article 53 obligations are discussed here.", key)
    assert not v["passed"]
    assert v["missed_cites"] == ["Article 55"]


def test_partial_facts_below_threshold_fail():
    key = {
        "must_cite": [],
        "must_state": [["alpha"], ["bravo"], ["charlie"], ["delta"]],  # need >= 70%
        "must_not_claim": [],
    }
    v = check_answer("this contains alpha and bravo only", key)  # 2/4 = 50%
    assert not v["passed"]
    assert sorted(v["missed_states"]) == ["charlie", "delta"]


def test_trap_phrase_is_flagged():
    key = {"must_cite": [], "must_state": [], "must_not_claim": ["you are just a deployer"]}
    v = check_answer("Yes, you are just a deployer under the AI Act.", key)
    assert v["violations"] == ["you are just a deployer"]