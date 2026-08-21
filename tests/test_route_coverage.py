"""Route-coverage matrix — the cheap, deterministic assessment instrument for
"does CRSS pick the right retrieval strategy for each in-scope question type?".

Almost every "odd answer" a user reports is a ROUTING failure (the wrong
retrieval strategy for the question's intent), not a generation failure — and
routing is a pure function, so it can be pinned exhaustively for free, with no
LLM and no Neo4j. This matrix asserts the expected route for one exemplar of
every in-scope archetype; a routing regression (or a newly-added archetype)
shows up here as a red test instead of a user-reported surprise.

It drives the REAL end-to-end classifier (application.scenario.detect_scenario),
because the interesting bugs live in how a raw question becomes explicit_refs /
is_definition_question, not in _select_question_route in isolation. The only
retriever dependency in that path is defined-term surfacing; the fake below
returns an empty term index (the same result as a cold graph), so every exemplar
names its regulation explicitly and routing stays independent of live services.

Known gaps are recorded as xfail(strict=True): they document a real routing
defect and will flip to a hard failure the moment it is fixed — the signal to
delete the marker and promote the case to a normal assertion.
"""
import pytest

from application.scenario import detect_scenario


class _NoDefinedTerms:
    """Retriever stand-in that surfaces no defined terms — keeps route selection
    dependent only on the question's surface features (no Neo4j needed)."""

    def get_defined_terms_index(self):
        return {}

    def find_by_term(self, term):
        return []


def _route(question: str) -> str:
    return detect_scenario(question, _NoDefinedTerms(), 12).route.id


# (expected_route_id, question) — one exemplar per in-scope archetype.
_MATRIX = [
    ("provision_lookup",         "Show me Article 6 of the AI Act"),
    ("definition_lookup",        "What is a 'high-risk AI system' under the AI Act?"),
    ("role_obligations",         "What are the obligations of a provider under the AI Act?"),
    ("cross_regulation",         "How do the AI Act and MDR interact for medical device software?"),
    ("classification_chain",     "What obligations flow from classifying an AI system as high-risk under the AI Act?"),
    ("community_summary_search", "Give me a comprehensive overview of the AI Act"),
    ("legal_qualification",      "Is our in-house hospital AI tool a high-risk medical device under the AI Act and MDR?"),
    ("general_compliance",       "Tell me about the transparency obligations in the AI Act"),

    # Reverse cross-reference: the asker wants the provisions that CITE the annex,
    # not the annex itself (the target ref is captured separately and fed to the
    # reverse-citation lookup).
    ("reverse_reference", "Which articles reference Annex III of the AI Act?"),
    ("reverse_reference", "Which provisions cite Annex VIII of the MDR?"),

    # Broad "learn about X" overview: the "what are the main <concept>" phrasing
    # must reach a corpus-level overview, not be swallowed as a defined term.
    ("community_summary_search", "What are the main aspects of the AI Act I should know?"),
]


@pytest.mark.parametrize("expected, question", _MATRIX)
def test_question_routes_to_expected_archetype(expected, question):
    assert _route(question) == expected
