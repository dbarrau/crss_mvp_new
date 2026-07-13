"""Deterministic quote repair — substitute source text, re-point citations.

Redaction discards what verification already computed: the true source span
(near-verbatim/fabricated paraphrases) and the true provision (misattributed).
Repair converts those into corrected answers; only unrepairable quotes redact.
"""
from application._faithfulness import (
    build_repair_note,
    check_faithfulness,
    repair_and_redact,
)

# Long enough to clear _MIN_QUOTE_LEN (40 chars) comfortably.
_ART9_TEXT = (
    "A risk management system shall be established, implemented, documented "
    "and maintained in relation to high-risk AI systems. "
    "The risk management system shall be understood as a continuous "
    "iterative process planned and run throughout the entire lifecycle of a "
    "high-risk AI system."
)
_ART10_TEXT = (
    "Training, validation and testing data sets shall be subject to data "
    "governance and management practices appropriate for the intended "
    "purpose of the high-risk AI system."
)


def _provisions():
    return [
        {"article_ref": "Article 9", "article_text": _ART9_TEXT},
        {"article_ref": "Article 10", "article_text": _ART10_TEXT},
    ]


def test_misattributed_quote_gets_repointed_not_removed():
    # Real Article 10 text cited as Article 9 → attribution corrected in place.
    answer = (
        "Data duties are strict. Under **Article 9**, "
        "“Training, validation and testing data sets shall be subject to "
        "data governance and management practices appropriate for the "
        "intended purpose of the high-risk AI system.”"
    )
    report = check_faithfulness(answer, _provisions())
    assert report.misattributed_count == 1

    repaired, residual, notes = repair_and_redact(answer, report, _provisions())
    assert residual.misattributed == []
    assert "Training, validation and testing data sets" in repaired
    assert "**Article 10**" in repaired          # re-pointed
    assert len(notes) == 1 and "Article 10" in notes[0]


def test_fabricated_paraphrase_of_cited_provision_is_substituted():
    # Model paraphrases Article 9 from memory inside quote marks.
    answer = (
        "Under **Article 9**, “A risk management system must be set up, "
        "implemented, documented and maintained for high-risk AI systems.”"
    )
    report = check_faithfulness(answer, _provisions())
    assert report.unverified_count == 1

    repaired, residual, notes = repair_and_redact(answer, report, _provisions())
    assert residual.unverified == []
    assert "shall be established, implemented, documented" in repaired
    assert "[…]" not in repaired
    assert notes


def test_truly_ungroundable_quote_still_redacts():
    answer = (
        "Under **Article 9**, “Providers shall notify the Pan-Galactic "
        "Compliance Authority within twelve parsecs of any incident.”"
    )
    report = check_faithfulness(answer, _provisions())
    assert report.unverified_count == 1

    repaired, residual, notes = repair_and_redact(answer, report, _provisions())
    assert residual.unverified and notes == []
    assert "[…]" in repaired
    assert "Pan-Galactic" not in repaired


def test_near_verbatim_quote_aligned_to_exact_source_wording():
    # A dropped word late in the quote ("entire") keeps a long contiguous
    # prefix → classifies near-verbatim → aligned to the exact source text.
    answer = (
        "Per **Article 9**, “The risk management system shall be understood "
        "as a continuous iterative process planned and run throughout the "
        "lifecycle of a high-risk AI system.”"
    )
    report = check_faithfulness(answer, _provisions())
    assert report.near_verbatim_count == 1

    repaired, residual, notes = repair_and_redact(answer, report, _provisions())
    assert residual.near_verbatim == []
    assert "throughout the entire lifecycle" in repaired


def test_repair_note_renders():
    note = build_repair_note(["citation corrected: the quoted text is from Article 10"])
    assert "Auto-verified corrections" in note and "Article 10" in note
    assert build_repair_note([]) is None
