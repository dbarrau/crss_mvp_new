"""Audit definition grounding — the auditor's gap-fill must resolve a definition
by its TERM, not by a paragraph number it recalled from memory.

Observed failure: the auditor requested "Article 3(40) AI Act (definition of
'safety component')", but safety component is Article 3(14) — 3(40) is 'biometric
categorisation system'. The number was fed to retrieve_by_refs verbatim, so the
gap-fill injected the wrong definition and the needed one never arrived.

Two complementary fixes:
  (i)  the medical-device-AI qualification backbone force-loads Article 3(14)
       directly, so it is in context and the auditor need not guess a number;
  (ii) the gap-fill resolves definition-wants through the term channel.
"""
from __future__ import annotations

from application._audit import (
    _DEFINITION_WANT_RE,
    _extract_defined_term,
    _gap_retrieve,
)
from application._routing import _build_legal_qualification_targets

AI = "32024R1689"


# ---------------------------------------------------------------------------
# (i) backbone anchors the safety-component definition point directly
# ---------------------------------------------------------------------------

def test_backbone_force_loads_safety_component_point_for_medical_device_ai():
    targets = _build_legal_qualification_targets(
        "Is my LLM-in-a-medical-device high-risk and what are my obligations?",
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("provider", AI)],
    )
    refs = [t.ref for t in targets]
    assert "Article 3(14)" in refs        # the safety-component definition point
    assert "Article 6" in refs and "Annex I" in refs  # the product route it supports


def test_backbone_omits_safety_component_when_ai_act_not_in_scope():
    targets = _build_legal_qualification_targets(
        "What are the lawful bases for processing personal data?",
        mentioned_regs={"General Data Protection Regulation (GDPR) 2016/679"},
        role_specs=[],
    )
    assert "Article 3(14)" not in [t.ref for t in targets]


# ---------------------------------------------------------------------------
# (ii) term extraction from an audit want
# ---------------------------------------------------------------------------

def test_extract_defined_term_from_quote_and_definition_of():
    assert _extract_defined_term(
        "Article 3(40) AI Act (definition of 'safety component')"
    ) == "safety component"
    assert _extract_defined_term(
        "Article 3(40) AI Act (definition of ‘safety component’)"
    ) == "safety component"
    assert _extract_defined_term("Article 3 AI Act (definition of provider)") == "provider"


def test_definition_signal_gates_extraction():
    # A quoted phrase without a definition signal is NOT a definition-want — the
    # want is really the article (Article 5), so it must not be diverted.
    ref = "Article 5 (the prohibition on 'social scoring')"
    assert not _DEFINITION_WANT_RE.search(ref)


# ---------------------------------------------------------------------------
# (ii) gap-fill resolves the definition by term, never the hallucinated number
# ---------------------------------------------------------------------------

class _FakeRetriever:
    def find_by_term(self, term):
        if term.lower() == "safety component":
            return [{"source_provision_id": f"{AI}_art_3_pt_14", "celex": AI}]
        return []

    def retrieve_by_ids(self, ids):
        return [{"article_id": i} for i in ids]

    def retrieve_by_refs(self, refs, celex_filter=None):
        out = []
        for r in refs:
            if "3(40)" in r:      # the wrong node — must never be requested here
                out.append({"article_id": f"{AI}_art_3_pt_40"})
            elif "6(1)" in r:
                out.append({"article_id": f"{AI}_art_6"})
        return out


def test_gap_fill_resolves_definition_by_term_not_number():
    findings = {
        "missing_provision_refs": [
            "Article 3(40) AI Act (definition of 'safety component')",
            "Article 6(1) AI Act (applicability date)",
        ],
        "missing_topics": [],
    }
    got = _gap_retrieve(
        findings, _FakeRetriever(),
        target_celexes={AI}, existing_ids=set(), max_add=8,
    )
    ids = [p["article_id"] for p in got]
    assert f"{AI}_art_3_pt_14" in ids       # correct definition, via the term channel
    assert f"{AI}_art_3_pt_40" not in ids    # the hallucinated number is never retrieved
    assert f"{AI}_art_6" in ids              # an ordinary ref still resolves by number


def test_gap_fill_falls_back_to_number_when_term_unresolvable():
    # A definition-want whose term is not a real DefinedTerm degrades to the
    # number path rather than dropping the request.
    class _Empty(_FakeRetriever):
        def find_by_term(self, term):
            return []
    findings = {
        "missing_provision_refs": ["Article 6(1) AI Act (definition of 'made-up term')"],
        "missing_topics": [],
    }
    got = _gap_retrieve(
        findings, _Empty(), target_celexes={AI}, existing_ids=set(), max_add=8,
    )
    assert [p["article_id"] for p in got] == [f"{AI}_art_6"]
