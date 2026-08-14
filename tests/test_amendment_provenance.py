"""Deterministic amendment-pedigree footer (application/_postprocessing.py).

CRSS surfaces a controlling amendment's text into context, but the model repeats
the provenance marker only unreliably. `_build_amendment_provenance` renders the
pedigree deterministically from the AMENDS-edge metadata for every amended
provision the answer actually cites — the traceability compliance teams need.
"""
from application._postprocessing import (
    _amendment_target_in_answer,
    _build_amendment_provenance,
)

_AMDS = [
    {"_amends_target_ref": "Article 6", "amending_act": "Regulation (EU) 2026/1744"},
    {"_amends_target_ref": "Annex I", "amending_act": "Regulation (EU) 2026/1744"},
    {"_amends_target_ref": "Article 43", "amending_act": "Regulation (EU) 2026/1744"},
]


# ── whole-reference matching ────────────────────────────────────────────────

def test_target_matches_amended_subparagraph_citation():
    # 'Article 6' must match the answer's 'Article 6(1a)' (the inserted paragraph).
    assert _amendment_target_in_answer("Article 6", "high-risk under Article 6(1a) applies")


def test_target_does_not_match_a_different_numbered_article():
    assert not _amendment_target_in_answer("Article 6", "see Article 60 and Article 63")


def test_annex_target_is_token_bounded():
    assert _amendment_target_in_answer("Annex I", "listed in Annex I of the Act")
    assert not _amendment_target_in_answer("Annex I", "listed in Annex III of the Act")


# ── regulation disambiguation (an AI Act amendment must not attach to an MDR
#    /GDPR citation — MDR/IVDR/GDPR each have their own Article 2, Annex I) ────

def test_reference_scoped_to_mdr_is_not_an_ai_act_citation():
    # HQ demo bug: 'Article 2(30) of the MDR' is the MDR manufacturer definition;
    # the AI Act Omnibus (which amends AI Act Article 2) must NOT claim it.
    assert not _amendment_target_in_answer("Article 2", "the manufacturer under Article 2(30) of the MDR")
    assert not _amendment_target_in_answer("Annex I", "the requirements in Annex I of the MDR")
    assert not _amendment_target_in_answer("Article 2", "under MDR Article 2 the manufacturer")


def test_incidental_mdr_mention_does_not_suppress_ai_act_annex():
    # 'Annex I (which includes the MDR)' IS the AI Act's Annex I (amended) — the
    # MDR is only named as content, not as the annex's regulation. Keep it.
    assert _amendment_target_in_answer("Annex I", "covered by Annex I (which includes the MDR)")


def test_ai_act_citation_kept_when_some_occurrences_are_mdr_scoped():
    # The answer uses 'Annex I' for both the AI Act annex and the MDR annex; as
    # long as one occurrence is an AI Act citation, the amendment applies.
    answer = ("safety requirements in Annex I of the MDR ... and the AI system is "
              "covered by Annex I (which includes the MDR)")
    assert _amendment_target_in_answer("Annex I", answer)


def test_provenance_drops_mdr_only_article_2():
    # Regression for the demo: Article 2 was surfaced (AI Act amendment) but the
    # answer only cites MDR Article 2(30) -> must NOT appear in the footer.
    amds = [
        {"_amends_target_ref": "Article 2", "amending_act": "Regulation (EU) 2026/1744"},
        {"_amends_target_ref": "Article 6", "amending_act": "Regulation (EU) 2026/1744"},
    ]
    answer = "manufacturer under **Article 2(30)** of the MDR ... high-risk under **Article 6(1a)**"
    out = _build_amendment_provenance(answer, amds)
    assert "Article 6" in out
    assert "Article 2" not in out        # MDR-scoped citation excluded


# ── provenance rendering ────────────────────────────────────────────────────

def test_provenance_lists_only_cited_amended_provisions():
    answer = ("The system is high-risk under **Article 6(1a)**. The MDR conformity "
              "route applies (**Article 43(3)**).")  # cites Article 6 + Article 43, not Annex I
    out = _build_amendment_provenance(answer, _AMDS)
    assert "AMENDMENTS APPLIED" in out
    assert "**Article 6** — amended by **Regulation (EU) 2026/1744**" in out
    assert "Article 43" in out
    assert "Annex I" not in out          # surfaced but not cited → excluded (no noise)


def test_provenance_dedupes_target_act_pairs():
    answer = "Article 6(1a) and Article 6(1b) both matter."
    dup = _AMDS[:1] * 3                   # same Article 6 amendment surfaced thrice
    out = _build_amendment_provenance(answer, dup)
    assert out.count("**Article 6**") == 1


def test_provenance_empty_when_no_amended_provision_cited():
    assert _build_amendment_provenance("A general answer citing Article 99.", _AMDS) == ""


def test_provenance_empty_without_amendments_or_answer():
    assert _build_amendment_provenance("Article 6 applies", []) == ""
    assert _build_amendment_provenance("", _AMDS) == ""


def test_provenance_flags_the_temporal_caveat():
    # Foreshadows #3: the footer must remind the reader to check the amending
    # act's own application dates (the amendment's timing, not the base act's).
    out = _build_amendment_provenance("Article 6(1a) applies", _AMDS)
    assert "application date" in out.lower()
