import pytest

from application.agent import (
    _detect_question_roles,
    _build_legal_qualification_targets,
    _build_route_answer_guidance,
    _build_uncertainty_banner,
    _build_user_message,
    _postprocess_answer,
    _evaluate_route_sufficiency,
    _run_corrective_retrieval_pass,
    _retrieve_route_provisions,
    _select_question_route,
    _has_inhouse_developer_signal,
    _has_multistage_question,
    _validate_legal_backbone,
)


class _FakeRetriever:
    def __init__(self, *, direct=None, role=None, hybrid=None):
        self.direct = self._normalize_sequences(direct)
        self.role = self._normalize_sequences(role)
        self.hybrid = self._normalize_sequences(hybrid)
        self.calls: list[str] = []

    @staticmethod
    def _normalize_sequences(value):
        if value is None:
            return [[]]
        if value and isinstance(value, list) and isinstance(value[0], list):
            return [list(item) for item in value]
        return [list(value)]

    @staticmethod
    def _take(sequence_store):
        if len(sequence_store) > 1:
            return sequence_store.pop(0)
        return list(sequence_store[0])

    def retrieve_by_refs(self, refs, celex_filter=None):
        self.calls.append("direct")
        return self._take(self.direct)

    def retrieve_by_roles(self, role_specs, k=8):
        self.calls.append("roles")
        return self._take(self.role)

    def encode_as_passage(self, text):
        self.calls.append("encode")
        return f"encoded:{text}"

    def retrieve(self, question, k=20, target_celexes=None, query_vec=None):
        self.calls.append("retrieve")
        return self._take(self.hybrid)


def test_select_question_route_prefers_direct_provision_lookup():
    route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )

    assert route.id == "provision_lookup"


def test_select_question_route_uses_definition_lookup():
    route = _select_question_route(
        "What is a provider under the AI Act?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=True,
    )

    assert route.id == "definition_lookup"


def test_select_question_route_uses_role_obligations():
    route = _select_question_route(
        "What obligations does the provider have under the AI Act?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act"},
        role_specs=[("provider", "32024R1689")],
        is_definition_question=False,
    )

    assert route.id == "role_obligations"


def test_select_question_route_prefers_cross_regulation():
    route = _select_question_route(
        "How do the MDR and the AI Act interact for provider obligations?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("provider", "32024R1689")],
        is_definition_question=False,
    )

    assert route.id == "cross_regulation"


def test_select_question_route_uses_legal_qualification_for_medical_ai_status_question():
    route = _select_question_route(
        (
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act or a "
            "manufacturer under the MDR?"
        ),
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )

    assert route.id == "legal_qualification"


def test_detect_question_roles_inflects_provider_and_manufacturer_from_conduct():
    roles = _detect_question_roles(
        (
            "A university hospital develops and puts into service its own "
            "AI pathology system for internal use."
        ),
        target_celexes={"32024R1689", "32017R0745"},
    )

    assert ("provider", "32024R1689") in roles
    assert ("manufacturer", "32017R0745") in roles
    assert ("deployer", "32024R1689") in roles


def test_retrieve_route_provisions_skips_hyde_for_direct_lookup():
    retriever = _FakeRetriever(
        direct=[{"article_id": "art-26", "children": []}],
    )
    route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )

    result = _retrieve_route_provisions(
        "What does Article 26 of the AI Act require?",
        retriever,
        client=object(),
        k=8,
        route=route,
        target_celexes={"32024R1689"},
        explicit_refs=["Article 26"],
        role_specs=[],
        has_definitions=False,
        hyde_builder=lambda *_args, **_kwargs: pytest.fail("HyDE should not run"),
    )

    assert retriever.calls == ["direct"]
    assert [p["article_id"] for p in result["provisions"]] == ["art-26"]


def test_retrieve_route_provisions_skips_hyde_for_role_lookup():
    retriever = _FakeRetriever(
        role=[{"article_id": "art-provider"}],
    )
    route = _select_question_route(
        "What obligations does the provider have under the AI Act?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act"},
        role_specs=[("provider", "32024R1689")],
        is_definition_question=False,
    )

    result = _retrieve_route_provisions(
        "What obligations does the provider have under the AI Act?",
        retriever,
        client=object(),
        k=8,
        route=route,
        target_celexes={"32024R1689"},
        explicit_refs=[],
        role_specs=[("provider", "32024R1689")],
        has_definitions=False,
        hyde_builder=lambda *_args, **_kwargs: pytest.fail("HyDE should not run"),
    )

    assert retriever.calls == ["roles"]
    assert [p["article_id"] for p in result["provisions"]] == ["art-provider"]


def test_retrieve_route_provisions_cross_regulation_combines_all_paths():
    retriever = _FakeRetriever(
        direct=[{"article_id": "art-43", "children": []}],
        role=[{"article_id": "art-provider"}],
        hybrid=[{"article_id": "art-hyde"}],
    )
    route = _select_question_route(
        "How do Article 43 of the AI Act and MDR provider duties interact?",
        explicit_refs=["Article 43"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("provider", "32024R1689")],
        is_definition_question=False,
    )

    result = _retrieve_route_provisions(
        "How do Article 43 of the AI Act and MDR provider duties interact?",
        retriever,
        client=object(),
        k=8,
        route=route,
        target_celexes={"32024R1689", "32017R0745"},
        explicit_refs=["Article 43"],
        role_specs=[("provider", "32024R1689")],
        has_definitions=False,
        hyde_builder=lambda *_args, **_kwargs: "synthetic hyde text",
    )

    assert retriever.calls == ["direct", "roles", "encode", "retrieve"]
    assert result["hyde_text"] == "synthetic hyde text"
    assert [p["article_id"] for p in result["provisions"]] == [
        "art-43",
        "art-provider",
        "art-hyde",
    ]


def test_retrieve_route_provisions_legal_qualification_forces_backbone_refs():
    retriever = _FakeRetriever(
        direct=[
            [{"article_id": "mdr-art-5", "article_ref": "Article 5", "celex": "32017R0745", "children": []}],
            [{"article_id": "ai-art-3", "article_ref": "Article 3", "celex": "32024R1689"}],
            [{"article_id": "mdr-art-2", "article_ref": "Article 2", "celex": "32017R0745"}],
            [{"article_id": "mdr-art-5-dup", "article_ref": "Article 5", "celex": "32017R0745"}],
            [{"article_id": "ai-art-6", "article_ref": "Article 6", "celex": "32024R1689"}],
            [{"article_id": "ai-annex-i", "article_ref": "Annex I", "celex": "32024R1689"}],
            [{"article_id": "ai-art-43", "article_ref": "Article 43", "celex": "32024R1689"}],
            [{"article_id": "ai-art-25", "article_ref": "Article 25", "celex": "32024R1689"}],
            [{"article_id": "mdcg-2025-6", "article_ref": "MDCG 2025-6", "celex": "MDCG_2025_6"}],
        ],
        role=[{"article_id": "art-role", "article_ref": "Article 29", "celex": "32024R1689"}],
        hybrid=[{"article_id": "art-hyde", "article_ref": "Article 10", "celex": "32024R1689"}],
    )
    route = _select_question_route(
        (
            "At what stage does a hospital using an in-house AI pathology "
            "system under MDR Article 5(5) become a provider under the AI Act "
            "or a manufacturer under the MDR?"
        ),
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )

    result = _retrieve_route_provisions(
        (
            "At what stage does a hospital using an in-house AI pathology "
            "system under MDR Article 5(5) become a provider under the AI Act "
            "or a manufacturer under the MDR?"
        ),
        retriever,
        client=object(),
        k=8,
        route=route,
        target_celexes={"32024R1689", "32017R0745"},
        explicit_refs=["Article 5"],
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        has_definitions=False,
        hyde_builder=lambda *_args, **_kwargs: "synthetic hyde text",
    )

    assert retriever.calls == [
        "direct",
        "direct",
        "direct",
        "direct",
        "direct",
        "direct",
        "direct",
        "direct",
        "roles",
        "encode",
        "retrieve",
    ]
    assert [target.ref for target in result["legal_qualification_targets"]] == [
        "Article 3",
        "Article 2",
        "Article 5",
        "Article 6",
        "Annex I",
        "Article 43",
        "Article 25",
        "MDCG 2025-6",
    ]
    assert [p["article_id"] for p in result["provisions"][:9]] == [
        "mdr-art-5",
        "ai-art-3",
        "mdr-art-2",
        "mdr-art-5-dup",
        "ai-art-6",
        "ai-annex-i",
        "ai-art-43",
        "ai-art-25",
        "art-role",
    ]


def test_build_legal_qualification_targets_skips_article_25_for_already_provider_question():
    targets = _build_legal_qualification_targets(
        (
            "Is a university hospital that develops and uses its own in-house "
            "AI pathology system already a provider under the AI Act and a "
            "manufacturer under the MDR?"
        ),
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[
            ("provider", "32024R1689"),
            ("deployer", "32024R1689"),
            ("manufacturer", "32017R0745"),
        ],
    )

    assert [target.ref for target in targets] == [
        "Article 3",
        "Article 2",
        "Article 5",
        "Article 6",
        "Annex I",
        "Article 43",
        "MDCG 2025-6",
    ]


def test_build_legal_qualification_targets_only_adds_annex_iii_when_explicitly_supported():
    default_targets = _build_legal_qualification_targets(
        (
            "Does an in-house AI pathology system under MDR Article 5(5) become "
            "high-risk under the AI Act?"
        ),
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
    )
    annex_iii_targets = _build_legal_qualification_targets(
        (
            "For a biometric identification system used by a hospital, should the "
            "AI Act analysis proceed through Annex III or the MDR-linked route?"
        ),
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
    )

    assert "Annex III" not in [target.ref for target in default_targets]
    assert [target.ref for target in annex_iii_targets] == [
        "Article 3",
        "Article 2",
        "Article 5",
        "Article 6",
        "Annex I",
        "Annex III",
        "Article 43",
        "MDCG 2025-6",
    ]


def test_evaluate_route_sufficiency_flags_missing_explicit_ref():
    route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )

    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question="What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        target_celexes={"32024R1689"},
        role_specs=[],
        provisions=[],
        definitions=[],
        direct_provisions=[],
        role_provisions=[],
        legal_qualification_targets=[],
    )

    assert sufficiency["ok"] is False
    assert sufficiency["missing_refs"] == ["Article 26"]


def test_corrective_retrieval_pass_recovers_missing_explicit_ref():
    retriever = _FakeRetriever(
        direct=[{"article_id": "art-26", "article_ref": "Article 26", "celex": "32024R1689", "children": []}],
    )
    route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )
    provisions = []
    direct_provisions = []
    role_provisions = []
    definitions = []

    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question="What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        target_celexes={"32024R1689"},
        role_specs=[],
        provisions=provisions,
        definitions=definitions,
        direct_provisions=direct_provisions,
        role_provisions=role_provisions,
        legal_qualification_targets=[],
    )
    recovery = _run_corrective_retrieval_pass(
        "What does Article 26 of the AI Act require?",
        retriever,
        client=object(),
        k=8,
        route=route,
        target_celexes={"32024R1689"},
        explicit_refs=["Article 26"],
        role_specs=[],
        provisions=provisions,
        direct_provisions=direct_provisions,
        role_provisions=role_provisions,
        definitions=definitions,
        sufficiency=sufficiency,
        hyde_text=None,
        legal_qualification_targets=[],
        hyde_builder=lambda *_args, **_kwargs: pytest.fail("HyDE should not run"),
    )

    assert recovery["sufficiency"]["ok"] is True
    assert recovery["actions"] == ["recovered 1 explicit ref target(s)"]
    assert [p["article_ref"] for p in provisions] == ["Article 26"]


def test_evaluate_route_sufficiency_flags_missing_cross_regulation_coverage():
    route = _select_question_route(
        "How do the MDR and the AI Act interact for provider obligations?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("provider", "32024R1689")],
        is_definition_question=False,
    )

    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question="How do the MDR and the AI Act interact for provider obligations?",
        explicit_refs=[],
        target_celexes={"32024R1689", "32017R0745"},
        role_specs=[("provider", "32024R1689")],
        provisions=[{"article_id": "art-provider", "celex": "32024R1689", "matched_role": "provider"}],
        definitions=[],
        direct_provisions=[],
        role_provisions=[{"article_id": "art-provider", "celex": "32024R1689"}],
        legal_qualification_targets=[],
    )

    assert sufficiency["ok"] is False
    assert sufficiency["missing_celexes"] == ["32017R0745"]


def test_evaluate_route_sufficiency_flags_single_community_concentration():
    route = _select_question_route(
        "How do AI Act and MDR deployer obligations compare?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689")],
        is_definition_question=False,
    )

    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question="How do AI Act and MDR deployer obligations compare?",
        explicit_refs=[],
        target_celexes={"32024R1689", "32017R0745"},
        role_specs=[("deployer", "32024R1689")],
        provisions=[
            {
                "article_id": "ai-art-26",
                "celex": "32024R1689",
                "community_id": "community::deployer",
                "matched_role": "deployer",
            },
            {
                "article_id": "mdr-art-10",
                "celex": "32017R0745",
                "community_id": "community::deployer",
                "matched_role": "deployer",
            },
        ],
        definitions=[],
        direct_provisions=[],
        role_provisions=[{"article_id": "ai-art-26", "celex": "32024R1689"}],
        legal_qualification_targets=[],
    )

    assert sufficiency["ok"] is False
    assert sufficiency["context_communities"] == ["community::deployer"]
    assert any(
        check["name"] == "community_diversity" and check["passed"] is False
        for check in sufficiency["checks"]
    )


def test_evaluate_route_sufficiency_passes_with_multi_community_coverage():
    route = _select_question_route(
        "How do AI Act and MDR deployer obligations compare?",
        explicit_refs=[],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689")],
        is_definition_question=False,
    )

    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question="How do AI Act and MDR deployer obligations compare?",
        explicit_refs=[],
        target_celexes={"32024R1689", "32017R0745"},
        role_specs=[("deployer", "32024R1689")],
        provisions=[
            {
                "article_id": "ai-art-26",
                "celex": "32024R1689",
                "community_id": "community::deployer",
                "matched_role": "deployer",
            },
            {
                "article_id": "mdr-art-10",
                "celex": "32017R0745",
                "community_id": "community::manufacturer",
                "matched_role": "deployer",
            },
        ],
        definitions=[],
        direct_provisions=[],
        role_provisions=[{"article_id": "ai-art-26", "celex": "32024R1689"}],
        legal_qualification_targets=[],
    )

    assert sufficiency["ok"] is True
    assert sufficiency["context_communities"] == [
        "community::deployer",
        "community::manufacturer",
    ]
    assert any(
        check["name"] == "community_diversity" and check["passed"] is True
        for check in sufficiency["checks"]
    )


def test_evaluate_route_sufficiency_flags_missing_qualification_backbone():
    question = (
        "When does a hospital using an in-house AI medical device under "
        "MDR Article 5(5) become a provider under the AI Act or a "
        "manufacturer under the MDR?"
    )
    route = _select_question_route(
        question,
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )
    qualification_targets = _build_legal_qualification_targets(
        question,
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
    )

    sufficiency = _evaluate_route_sufficiency(
        route=route,
        question=question,
        explicit_refs=["Article 5"],
        target_celexes={"32024R1689", "32017R0745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        provisions=[
            {"article_id": "mdr-art-5", "article_ref": "Article 5", "celex": "32017R0745"},
            {"article_id": "ai-art-3", "article_ref": "Article 3", "celex": "32024R1689", "matched_role": "deployer"},
        ],
        definitions=[],
        direct_provisions=[{"article_id": "mdr-art-5", "article_ref": "Article 5", "celex": "32017R0745"}],
        role_provisions=[{"article_id": "ai-art-3", "article_ref": "Article 3", "celex": "32024R1689"}],
        legal_qualification_targets=qualification_targets,
    )

    assert sufficiency["ok"] is False
    assert [item["ref"] for item in sufficiency["missing_qualification_targets"]] == [
        "Article 2",
        "Article 6",
        "Annex I",
        "Article 43",
        "Article 25",
        "MDCG 2025-6",
    ]


def test_build_route_answer_guidance_requires_uncertainty_for_qualification_route():
    route = _select_question_route(
        (
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act or a "
            "manufacturer under the MDR?"
        ),
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )

    guidance = _build_route_answer_guidance(
        route,
        question=(
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act or a "
            "manufacturer under the MDR?"
        ),
        sufficiency={"ok": False},
    )

    assert guidance is not None
    assert "MANDATORY LEGAL RULES" in guidance
    assert "case-specific" in guidance
    assert "Explicitly stated in retrieved text" in guidance
    assert "Resolve initial actor status before any transition analysis" in guidance
    assert "treat Article 6(1) plus Annex I as the default high-risk route" in guidance
    assert "Article 6(1) plus Annex I" in guidance
    assert "Article 3 provider-definition analysis before Article 25" in guidance
    assert "avoid a definitive bottom-line conclusion" in guidance


def test_build_user_message_injects_route_guidance_only_for_qualification_route():
    qualification_route = _select_question_route(
        (
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act or a "
            "manufacturer under the MDR?"
        ),
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )
    general_route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )

    qualification_message = _build_user_message(
        question="Q1",
        context="CTX",
        route=qualification_route,
        sufficiency={"ok": True},
    )
    general_message = _build_user_message(
        question="Q2",
        context="CTX",
        route=general_route,
        sufficiency={"ok": True},
    )

    assert "MANDATORY LEGAL RULES" in qualification_message
    assert "ANSWER DISCIPLINE FOR THIS QUESTION:" in qualification_message
    assert "REGULATORY CONTEXT:\nCTX" in qualification_message
    assert qualification_message.endswith("QUESTION: Q1")
    assert "MANDATORY LEGAL RULES" not in general_message
    assert general_message == "REGULATORY CONTEXT:\nCTX\n\nQUESTION: Q2"


def test_build_uncertainty_banner_varies_with_sufficiency():
    route = _select_question_route(
        (
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act or a "
            "manufacturer under the MDR?"
        ),
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )

    partial_banner = _build_uncertainty_banner(route, sufficiency={"ok": False})
    complete_banner = _build_uncertainty_banner(route, sufficiency={"ok": True})

    assert "partial retrieval support" in partial_banner
    assert "not as an automatic status determination" in complete_banner


def test_postprocess_answer_adds_banner_and_softens_categorical_phrasing():
    route = _select_question_route(
        (
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act or a "
            "manufacturer under the MDR?"
        ),
        explicit_refs=["Article 5"],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689"), ("manufacturer", "32017R0745")],
        is_definition_question=False,
    )

    processed = _postprocess_answer(
        "It only when shared externally constitutes a transfer that triggers full obligations.",
        route,
        question=(
            "When does a hospital using an in-house AI medical device under "
            "MDR Article 5(5) become a provider under the AI Act?"
        ),
        sufficiency={"ok": False},
    )

    assert processed.startswith("> ASSESSMENT STATUS — Provisional legal qualification assessment")
    assert "most clearly when" in processed
    assert "is likely to constitute" in processed
    assert "is likely to trigger" in processed


def test_postprocess_answer_leaves_general_route_unchanged():
    route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )

    processed = _postprocess_answer(
        "Article 26 constitutes a relevant provision.",
        route,
        question="What does Article 26 of the AI Act require?",
        sufficiency={"ok": True},
    )

    assert processed == "Article 26 constitutes a relevant provision."


# ══ New helpers (Phase A, D, E) ═══════════════════════════════════════════════════════════

def test_has_inhouse_developer_signal_true_for_hospital_developer():
    assert _has_inhouse_developer_signal(
        "A university hospital develops and trains its own in-house AI pathology system."
    ) is True
    assert _has_inhouse_developer_signal(
        "The hospital builds an AI model internally and puts it into service."
    ) is True
    assert _has_inhouse_developer_signal(
        "The institution designed and implemented an in-house AI diagnostic tool."
    ) is True


def test_has_inhouse_developer_signal_false_for_vendor_user():
    # No development verb — purely deploying someone else's system.
    assert _has_inhouse_developer_signal(
        "A hospital deploys an AI system purchased from a vendor."
    ) is False
    # No institutional context.
    assert _has_inhouse_developer_signal(
        "A company develops a general-purpose AI system."
    ) is False
    # Using verb but no development context.
    assert _has_inhouse_developer_signal(
        "At what stage does a hospital using an in-house AI pathology system become a provider?"
    ) is False


def test_build_legal_qualification_targets_includes_mdcg_2025_6_for_ai_act_mdr_overlap():
    targets = _build_legal_qualification_targets(
        "Does a hospital AI system qualify as high-risk under both MDR and the AI Act?",
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[],
    )
    refs = [t.ref for t in targets]
    assert "MDCG 2025-6" in refs
    # IVDR + AI Act overlap also triggers it.
    targets_ivdr = _build_legal_qualification_targets(
        "How does the AI Act apply to an IVDR-regulated AI diagnostic?",
        mentioned_regs={"EU AI Act", "IVDR 2017/746"},
        role_specs=[],
    )
    assert "MDCG 2025-6" in [t.ref for t in targets_ivdr]
    # AI Act alone does NOT add MDCG 2025-6.
    targets_ai_only = _build_legal_qualification_targets(
        "A hospital uses an AI system; when does it become a provider under the AI Act?",
        mentioned_regs={"EU AI Act"},
        role_specs=[("deployer", "32024R1689")],
    )
    assert "MDCG 2025-6" not in [t.ref for t in targets_ai_only]


def test_has_multistage_question_detection():
    # Benchmark question has many temporal markers.
    assert _has_multistage_question(
        "The hospital initially uses the system internally. After two years, "
        "it subsequently shares model weights with a second hospital."
    ) is True
    # Single temporal marker — not multistage.
    assert _has_multistage_question(
        "Is the hospital initially a deployer under the AI Act?"
    ) is False


def test_validate_legal_backbone_flags_deployer_misclassification():
    route = _select_question_route(
        "A university hospital develops its own in-house AI system.",
        explicit_refs=[],
        mentioned_regs={"EU AI Act", "MDR 2017/745"},
        role_specs=[("deployer", "32024R1689")],
        is_definition_question=False,
    )
    question = "A university hospital develops and trains its own in-house AI pathology system."

    # Answer that incorrectly classifies the developer as initially a deployer.
    warnings = _validate_legal_backbone(
        "The hospital is initially a deployer under the AI Act because it uses the system internally.",
        question,
        route,
    )
    assert len(warnings) >= 1
    assert any("BACKBONE FLAG" in w for w in warnings)

    # Correct answer — no warnings.
    warnings_ok = _validate_legal_backbone(
        "The hospital is a provider from inception under Article 3(3) AI Act.",
        question,
        route,
    )
    assert warnings_ok == []


def test_validate_legal_backbone_silent_for_general_route():
    route = _select_question_route(
        "What does Article 26 of the AI Act require?",
        explicit_refs=["Article 26"],
        mentioned_regs={"EU AI Act"},
        role_specs=[],
        is_definition_question=False,
    )
    # Even if answer has deployer language, validator doesn't fire for non-qualification routes.
    warnings = _validate_legal_backbone(
        "The hospital is initially a deployer.",
        "What does Article 26 require?",
        route,
    )
    assert warnings == []
