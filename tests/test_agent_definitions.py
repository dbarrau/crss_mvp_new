from application.agent import _expand_definitions_from_provisions


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
        }

    def get_defined_terms_index(self):
        return {
            "substantial modification": "substantial_modification",
            "remote biometric identification system": "remote_biometric_identification_system",
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

    assert [d["term"] for d in expanded] == [
        "provider",
        "deployer",
        "substantial modification",
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
        {
            "article_text": "Article 50 mentions remote biometric identification system.",
            "children": [],
            "matched_leaf_id": None,
        },
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
