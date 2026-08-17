"""Deterministic amendment-pedigree footer (application/_postprocessing.py).

CRSS surfaces a controlling amendment's text into context, but the model repeats
the provenance marker only unreliably. `_build_amendment_provenance` renders the
pedigree deterministically from the AMENDS-edge metadata for every amended
provision the answer actually cites — the traceability compliance teams need.
"""
from application._postprocessing import (
    _amendment_change_summary,
    _amendment_new_wording,
    _amendment_target_in_answer,
    _build_amendment_provenance,
)


# ── 'what changed' extraction from the amending provision's lead-in ──────────

_HEAD = "Article 1 — Amendments to Regulation (EU) 2024/1689 | "


def test_change_summary_insertion():
    a = {"article_text": _HEAD + "in Article 6, the following paragraphs are inserted: ‘1a. …’"}
    assert _amendment_change_summary(a, "Article 6") == "the following paragraphs are inserted"


def test_change_summary_replacement_drops_by_the_following():
    a = {"article_text": _HEAD + "in Article 43, paragraph 3 is replaced by the following: ‘3. …’"}
    assert _amendment_change_summary(a, "Article 43") == "paragraph 3 is replaced"


def test_change_summary_nested_amendment_drops_as_follows():
    a = {"article_text": _HEAD + "Article 3 is amended as follows: point (14) is amended as follows: ‘…’"}
    assert _amendment_change_summary(a, "Article 3") == "point (14) is amended"


def test_change_summary_annex_multi_operation():
    a = {"article_text": _HEAD + "Annex I is amended as follows: in Section A, point 1 is "
         "deleted; in Section B, the following point is added: ‘…’"}
    assert _amendment_change_summary(a, "Annex I") == (
        "in Section A, point 1 is deleted; in Section B, the following point is added")


def test_change_summary_empty_without_text():
    assert _amendment_change_summary({}, "Article 6") == ""


def test_provenance_renders_what_changed_when_text_present():
    amds = [{"_amends_target_ref": "Article 6", "amending_act": "Regulation (EU) 2026/1744",
             "article_text": _HEAD + "in Article 6, the following paragraphs are inserted: ‘1a. …’"}]
    out = _build_amendment_provenance("high-risk under Article 6(1a)", amds)
    # the row carries the operation AND the actual new wording (in quotes)
    assert "**Article 6** — the following paragraphs are inserted: “1a. …” (**Regulation (EU) 2026/1744**)" in out


def test_provenance_row_shows_the_actual_amended_wording():
    # The header promises "the amended wording is what currently applies" — the row
    # must therefore show that wording (e.g. Article 113's deferred dates), not just
    # the operation label.
    amds = [{"_amends_target_ref": "Article 113", "amending_act": "Regulation (EU) 2026/1744",
             "article_text": _HEAD + ("in Article 113, the third paragraph is amended as follows: "
                                      "point (c) is replaced by the following: ‘(c) Chapter III shall "
                                      "apply from: (i) 2 December 2027 as regards Article 6(2) and Annex III; "
                                      "and (ii) 2 August 2028 as regards Article 6(1) and Annex I;’")}]
    out = _build_amendment_provenance("enforcement under Article 113", amds)
    assert "2 December 2027" in out and "2 August 2028" in out          # the substance, not just "replaced"


def test_amendment_new_wording_truncates_at_a_word_boundary():
    long = _HEAD + "the following point is added: ‘" + "word " * 200 + "’"
    w = _amendment_new_wording({"article_text": long})
    assert w.endswith("…") and " word…" not in w[:-1]                    # clean cut, no mid-word

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


def test_provenance_header_names_the_act_with_its_catalog_name():
    # The pedigree header must name the instrument (recognizable short name from
    # the catalog), not hedge with a bare "a later act".
    out = _build_amendment_provenance("Article 6(1a) applies", _AMDS)
    assert "a later act" not in out
    assert "modified by **Regulation (EU) 2026/1744 (Digital Omnibus on AI)**" in out


def test_provenance_header_lists_multiple_acts_and_falls_back_for_unknown():
    amds = [
        {"_amends_target_ref": "Article 6", "amending_act": "Regulation (EU) 2026/1744"},
        {"_amends_target_ref": "Article 9", "amending_act": "Regulation (EU) 2099/0001"},
    ]
    out = _build_amendment_provenance("Article 6 and Article 9 both apply.", amds)
    assert "modified by later acts (" in out
    assert "Regulation (EU) 2026/1744 (Digital Omnibus on AI)" in out   # known → named
    assert "**Regulation (EU) 2099/0001**" in out                       # unknown → bare, no crash
