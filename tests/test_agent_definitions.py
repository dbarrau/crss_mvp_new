from application.agent import _detect_defined_terms, _expand_definitions_from_provisions


class _FakeRetriever:
    def __init__(self):
        self._results = {
            "substantial modification": [{
                "term": "substantial modification",
                "definition_type": "formal",
                "celex": "32024R1689",
                "regulation": "EU AI Act",
                "definition_text": "formal definition text",
            }],
            "remote biometric identification system": [{
                "term": "remote biometric identification system",
                "definition_type": "formal",
                "celex": "32024R1689",
                "regulation": "EU AI Act",
                "definition_text": "other definition text",
            }],
            "ai system": [{
                "term": "AI system",
                "definition_type": "formal",
                "celex": "32024R1689",
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
        target_celexes={"32024R1689"},
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
        target_celexes={"32024R1689"},
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
        target_celexes={"32024R1689"},
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
