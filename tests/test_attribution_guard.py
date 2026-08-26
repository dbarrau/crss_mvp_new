"""Tests for the guidance-attribution guard (application/_attribution.py)."""
from application._attribution import normalize_guidance_attribution

_TRANSPARENCY = "AI Office Guidelines on Transparency (Art. 50 AI Act)"


def _prov(celex, regulation, article_ref, article_path=""):
    return {
        "celex": celex,
        "regulation": regulation,
        "article_ref": article_ref,
        "article_path": article_path,
    }


def _transparency_provisions():
    """One AI Office doc retrieved, with the 'Exclusions from the scope' section."""
    return [
        _prov(
            "AI_OFFICE_2025_TRANSPARENCY_ART50",
            _TRANSPARENCY,
            "Exclusions from the scope of the AI Act",
            f"{_TRANSPARENCY} / Exclusions from the scope of the AI Act",
        ),
        _prov(
            "AI_OFFICE_2025_TRANSPARENCY_ART50",
            _TRANSPARENCY,
            "Research & Development",
            f"{_TRANSPARENCY} / Exclusions from the scope of the AI Act / Research & Development",
        ),
    ]


def test_fabricated_title_corrected_to_canonical():
    ans = (
        "The AI Office Guidelines on Exclusions from the Scope of the AI Act "
        "confirm that the exemption is tied to development purpose."
    )
    out, changes = normalize_guidance_attribution(ans, _transparency_provisions())
    assert _TRANSPARENCY in out
    assert "Guidelines on Exclusions from the Scope" not in out
    assert out.startswith("The ")  # leading article preserved
    assert any("corrected fabricated" in c for c in changes)


def test_correct_title_left_unchanged():
    ans = f"The {_TRANSPARENCY} clarify the exemption's scope."
    out, changes = normalize_guidance_attribution(ans, _transparency_provisions())
    assert out == ans
    assert not any("corrected fabricated" in c for c in changes)


def test_machinery_leak_stripped():
    ans = "The guidance (referenced in the context) supports this reading."
    out, changes = normalize_guidance_attribution(ans, _transparency_provisions())
    assert "referenced in the context" not in out
    assert "the context" not in out.lower() or "context of" in out.lower()
    assert any("machinery-leak" in c for c in changes)


def test_legitimate_in_the_context_of_preserved():
    # The legit legal phrase must survive — only the machinery form is stripped.
    ans = "In the context of Article 50 AI Act, this exclusion covers outputs."
    out, changes = normalize_guidance_attribution(ans, _transparency_provisions())
    assert out == ans
    assert not changes


def test_multiple_docs_resolved_via_owning_section():
    # Two AI Office docs retrieved, but the fabricated tail is a section of only
    # ONE of them (Transparency) → resolve to that doc precisely.
    provs = [
        _prov("AI_OFFICE_2025_TRANSPARENCY_ART50", _TRANSPARENCY,
              "Exclusions from the scope of the AI Act"),
        _prov("AI_OFFICE_2025_HIGHRISK_ANNEX_III",
              "AI Office Draft Guidelines on High-Risk Classification: Annex III",
              "Point 4(a)"),
    ]
    ans = "The AI Office Guidelines on Exclusions from the Scope of the AI Act confirm this."
    out, changes = normalize_guidance_attribution(ans, provs)
    assert _TRANSPARENCY in out
    assert "Guidelines on Exclusions" not in out
    assert any("corrected fabricated" in c for c in changes)


def test_truly_ambiguous_section_not_rewritten():
    # The same section name appears in BOTH docs → cannot resolve → leave as-is.
    provs = [
        _prov("AI_OFFICE_2025_TRANSPARENCY_ART50", _TRANSPARENCY, "Scope and definitions"),
        _prov("AI_OFFICE_2025_HIGHRISK_ANNEX_III",
              "AI Office Draft Guidelines on High-Risk Classification: Annex III",
              "Scope and definitions"),
    ]
    ans = "The AI Office Guidelines on Scope and Definitions of the AI Act say X."
    out, changes = normalize_guidance_attribution(ans, provs)
    assert out == ans
    assert not any("corrected fabricated" in c for c in changes)


def test_no_guidance_retrieved_is_noop():
    provs = [_prov("32024R1689", "EU AI Act", "Article 2(6)")]
    ans = "Article 2(6) AI Act governs the R&D exemption."
    out, changes = normalize_guidance_attribution(ans, provs)
    assert out == ans
    assert not changes


def test_legit_clause_ending_in_ai_act_not_mangled():
    # A non-title clause that happens to end in "AI Act" must NOT be rewritten,
    # because its tail does not match any retrieved section name.
    ans = (
        "The AI Office Guidelines on Transparency require disclosure obligations "
        "under the AI Act."
    )
    out, changes = normalize_guidance_attribution(ans, _transparency_provisions())
    assert out == ans
    assert not any("corrected fabricated" in c for c in changes)


def test_empty_answer():
    out, changes = normalize_guidance_attribution("", _transparency_provisions())
    assert out == ""
    assert not changes
