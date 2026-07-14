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

def test_must_cite_accepts_alternatives():
    # A list element is satisfied if ANY alternative is cited.
    key = {"must_cite": ["Article 5", ["Article 25", "Article 3"]],
           "must_state": [], "must_not_claim": []}
    # cites Article 5 and Article 3(3) -> the [25,3] alternative is satisfied
    v = check_answer("Under Article 5 MDR and Article 3(3) AI Act, the hospital is a provider.", key)
    assert v["passed"] and v["cite_recall"] == 1.0
    # neither alternative present -> the requirement is missed
    v2 = check_answer("Only Article 5 MDR is discussed.", key)
    assert not v2["passed"] and v2["missed_cites"] == ["Article 25 or Article 3"]


def test_must_state_folds_hyphens():
    # v5 residual (HQ_025): the answer wrote "fundamental-rights impact
    # assessment" (hyphenated) where the key has the unhyphenated phrase; a raw
    # substring match reported a present fact as missing.
    key = {
        "must_cite": [],
        "must_state": [["fundamental rights impact assessment"]],
        "must_not_claim": [],
    }
    answer = "The deployer must perform a fundamental-rights impact assessment before use."
    v = check_answer(answer, key)
    assert v["passed"] and v["state_recall"] == 1.0

    # The reverse orthography is also tolerated.
    key2 = {"must_cite": [], "must_state": [["machine-readable"]], "must_not_claim": []}
    assert check_answer("outputs marked in a machine readable format", key2)["passed"]


def test_must_state_strips_markdown_emphasis():
    # CRSS bolds every provision reference by mandate, so a key phrase
    # containing an article number must match through the ** markers.
    key = {
        "must_cite": [],
        "must_state": [["in addition to the article 6"]],
        "must_not_claim": [],
    }
    answer = "In addition to the **Article 6(1)** lawful basis, satisfy Article 9(2)."
    assert check_answer(answer, key)["passed"]
