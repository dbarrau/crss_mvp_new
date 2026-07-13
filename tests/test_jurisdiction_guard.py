"""Deterministic jurisdiction guard — foreign statutory citations are removed.

The corpus is EU-only, so a US CFR/USC citation in an answer is unverifiable by
construction and was necessarily typed from training memory. The guard is
narrow: only statutory-citation grammar triggers; a bare mention of "FDA" or
"510(k)" is legitimate (e.g. "FDA clearance confers no EU conformity").
"""
from application._postprocessing import _strip_foreign_law_citations


def test_cfr_citation_line_is_removed_with_warning():
    answer = (
        "You cannot rely on FDA 510(k) clearance under the MDR.\n"
        "- Report adverse events under 21 CFR Part 803 within 30 days.\n"
        "- Conformity assessment follows **Article 52**.\n"
    )
    cleaned, n = _strip_foreign_law_citations(answer)
    assert n == 1
    assert "21 CFR" not in cleaned
    assert "JURISDICTION FLAG" in cleaned
    assert "**Article 52**" in cleaned
    # The legitimate FDA mention survives — only the statutory citation goes.
    assert "FDA 510(k) clearance" in cleaned


def test_usc_and_medwatch_variants_trigger():
    for line in (
        "See 42 U.S.C. § 262 for biologics.",
        "File a MedWatch report with the FDA.",
        "Corrections are governed by CFR Part 806.",
        "The FD&C Act defines adulteration.",
    ):
        _, n = _strip_foreign_law_citations(f"Intro.\n{line}\nOutro.")
        assert n == 1, line


def test_bare_fda_and_510k_mentions_do_not_trigger():
    answer = (
        "FDA 510(k) clearance confers no conformity under the MDR; there is no "
        "mutual recognition. FDA requirements are outside the scope of this "
        "analysis and should be confirmed with US counsel."
    )
    cleaned, n = _strip_foreign_law_citations(answer)
    assert n == 0
    assert cleaned == answer


def test_eu_citations_never_trigger():
    answer = (
        "**Article 52(4)** governs Class IIb devices; see **Annex IX, Chapter I, "
        "point 3.3** and Regulation (EU) 2017/745. GDPR Article 28 also applies."
    )
    cleaned, n = _strip_foreign_law_citations(answer)
    assert n == 0
    assert cleaned == answer


def test_clean_answer_gets_no_warning_block():
    cleaned, n = _strip_foreign_law_citations("A fully EU-scoped answer.")
    assert n == 0
    assert "JURISDICTION FLAG" not in cleaned
