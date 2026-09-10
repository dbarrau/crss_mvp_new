from domain.legislation_catalog import (
    AI_ACT_CELEX,
)
from application.agent import _detect_defined_terms, _expand_definitions_from_provisions


class _FakeRetriever:
    def __init__(self):
        self._results = {
            "substantial modification": [{
                "term": "substantial modification",
                "definition_type": "formal",
                "celex": AI_ACT_CELEX,
                "regulation": "EU AI Act",
                "definition_text": "formal definition text",
            }],
            "remote biometric identification system": [{
                "term": "remote biometric identification system",
                "definition_type": "formal",
                "celex": AI_ACT_CELEX,
                "regulation": "EU AI Act",
                "definition_text": "other definition text",
            }],
            "ai system": [{
                "term": "AI system",
                "definition_type": "formal",
                "celex": AI_ACT_CELEX,
                "regulation": "EU AI Act",
                "definition_text": "'AI system' means a machine-based system ...",
            }],
        }

    def get_defined_terms_index(self):
        return {
            "substantial modification": "substantial_modification",
            "remote biometric identification system": "remote_biometric_identification_system",
            "ai system": "ai_system",
        }

    def find_by_term(self, term: str):
        return list(self._results.get(term, []))


def test_expand_definitions_from_provisions_adds_formal_term_used_in_context():
    retriever = _FakeRetriever()
    existing = [{"term": "provider"}, {"term": "deployer"}]
    provisions = [{
        "article_text": (
            "Any deployer becomes a provider if they make a substantial "
            "modification to a high-risk AI system."
        ),
        "children": [],
        "matched_leaf_id": None,
    }]

    expanded = _expand_definitions_from_provisions(
        provisions,
        retriever,
        existing,
        target_celexes={AI_ACT_CELEX},
    )

    # 'high-risk AI system' in the provision text also resolves the
    # 'AI system' definition (order: existing terms, then context expansions
    # longest-first, so 'substantial modification' precedes 'AI system').
    assert [d["term"] for d in expanded] == [
        "provider",
        "deployer",
        "substantial modification",
        "AI system",
    ]


def test_expand_definitions_from_provisions_ignores_lower_ranked_noise():
    retriever = _FakeRetriever()
    provisions = [
        {
            "article_text": "Article 25 refers to a substantial modification.",
            "children": [],
            "matched_leaf_id": None,
        },
        {
            "article_text": "Recital 84 repeats substantial modification.",
            "children": [],
            "matched_leaf_id": None,
        },
        {
            "article_text": "Article 43 also mentions substantial modification.",
            "children": [],
            "matched_leaf_id": None,
        },
        ] + [
            {
                "article_text": f"Dummy padding {i}.",
                "children": [],
                "matched_leaf_id": None,
            } for i in range(10)
        ] + [
    ]

    expanded = _expand_definitions_from_provisions(
        provisions,
        retriever,
        existing=[],
        target_celexes={AI_ACT_CELEX},
    )

    terms = [d["term"] for d in expanded]
    assert "substantial modification" in terms
    assert "remote biometric identification system" not in terms


def test_expand_definitions_matches_plural_term_in_context():
    """Provisions use 'AI systems' (plural); the index key is singular.

    Regression for the silent-fallback bug where Article 3(1) 'AI system' was
    never expanded because ``\\bai system\\b`` did not match 'AI systems',
    forcing the LLM to backfill the definition from training memory.
    """
    retriever = _FakeRetriever()
    provisions = [{
        "article_text": (
            "High-risk AI systems shall be designed and developed to achieve an "
            "appropriate level of accuracy, robustness and cybersecurity."
        ),
        "children": [],
        "matched_leaf_id": None,
    }]

    expanded = _expand_definitions_from_provisions(
        provisions,
        retriever,
        existing=[],
        target_celexes={AI_ACT_CELEX},
    )

    assert "AI system" in [d["term"] for d in expanded]


def test_detect_defined_terms_matches_plural_in_question():
    """A question phrased with the plural ('AI systems') still resolves the
    singular index key ('ai system')."""
    retriever = _FakeRetriever()
    matched = _detect_defined_terms(
        "What obligations apply to high-risk AI systems under the AI Act?",
        retriever,
    )
    assert "AI system" in [d["term"] for d in matched]


# ── cross-regulation defined-term detection (spelling + breadth) ─────────────
from domain.legislation_catalog import MDR_CELEX, IVDR_CELEX, GDPR_CELEX


class _MultiRegRepRetriever:
    """'authorised representative' is defined in MDR/IVDR/AI Act; the GDPR defines
    the shorter 'representative'. Mirrors the real graph that surfaced the bug."""

    _AR = "authorised representative"

    def get_defined_terms_index(self):
        return {self._AR: "authorised_representative", "representative": "representative"}

    def find_by_term(self, term):
        if term == self._AR:
            return [
                {"term": self._AR, "celex": MDR_CELEX, "regulation": "MDR 2017/745",
                 "definition_text": "AR (MDR)"},
                {"term": self._AR, "celex": IVDR_CELEX, "regulation": "IVDR 2017/746",
                 "definition_text": "AR (IVDR)"},
                {"term": self._AR, "celex": AI_ACT_CELEX, "regulation": "EU AI Act",
                 "definition_text": "AR (AI Act)"},
            ]
        if term == "representative":
            return [{"term": "representative", "celex": GDPR_CELEX,
                     "regulation": "General Data Protection Regulation (GDPR) 2016/679",
                     "definition_text": "representative (GDPR)"}]
        return []


def test_us_spelling_matches_british_defined_term():
    # "authorized" (US) must hit the British "authorised representative" key.
    matched = _detect_defined_terms(
        "Explain to me what is an authorized representative", _MultiRegRepRetriever()
    )
    celexes = {d["celex"] for d in matched}
    assert {MDR_CELEX, IVDR_CELEX, AI_ACT_CELEX} <= celexes


def test_no_reg_named_surfaces_every_regulations_definition():
    # A bare "what is X?" for a cross-regulation term must return one definition
    # per regulation, not one arbitrary pick (the old results[:1] tunnel).
    matched = _detect_defined_terms(
        "what is an authorized representative", _MultiRegRepRetriever()
    )
    # one 'authorised representative' per MDR/IVDR/AI Act, plus GDPR 'representative'
    assert len([d for d in matched if d["term"] == "authorised representative"]) == 3
    assert any(d["celex"] == GDPR_CELEX for d in matched)


def test_named_regulation_still_narrows_to_one():
    # When the question names a regulation, the old single-definition scoping holds.
    matched = _detect_defined_terms(
        "what is the authorised representative under the MDR", _MultiRegRepRetriever()
    )
    ar = [d for d in matched if d["term"] == "authorised representative"]
    assert len(ar) == 1 and ar[0]["celex"] == MDR_CELEX
