from canonicalization.role_linker import (
    _augment_with_derived_roles,
    _build_actor_roles,
    _build_equivalent_edges,
    _build_includes_edges,
    _select_actor_terms,
)
from domain.ontology.actor_roles import detect_role_specs


def test_detect_role_specs_resolves_hospital_to_deployer_and_users():
    specs = detect_role_specs(
        "What must a hospital verify before putting a high-risk AI medical device into service?"
    )

    assert specs == [
        ("deployer", "32024R1689"),
        ("user", "32017R0745"),
        ("user", "32017R0746"),
    ]


def test_detect_role_specs_honors_celex_filter():
    specs = detect_role_specs(
        "What must a hospital verify before putting a high-risk AI medical device into service?",
        target_celexes={"32024R1689"},
    )

    assert specs == [("deployer", "32024R1689")]


def test_select_actor_terms_promotes_composite_operator_definition():
    defined_terms = [
        {
            "defined_term_id": "dt_provider",
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": "32024R1689",
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_deployer",
            "term": "deployer",
            "category": "actor",
            "term_normalized": "deployer",
            "celex": "32024R1689",
            "definition_text": "'deployer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_importer",
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": "32024R1689",
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": "32024R1689",
            "definition_text": "'operator' means a provider, deployer or importer;",
        },
    ]

    selected = _select_actor_terms(defined_terms)

    assert {row["term_normalized"] for row in selected} == {
        "provider",
        "deployer",
        "importer",
        "operator",
    }


def test_select_actor_terms_promotes_exact_legal_user_role():
    defined_terms = [
        {
            "defined_term_id": "dt_user_mdr",
            "term": "user",
            "category": "other",
            "term_normalized": "user",
            "celex": "32017R0745",
            "definition_text": "'user' means any healthcare professional or lay person who uses a device;",
        },
        {
            "defined_term_id": "dt_user_ivdr",
            "term": "user",
            "category": "other",
            "term_normalized": "user",
            "celex": "32017R0746",
            "definition_text": "'user' means any healthcare professional or lay person who uses a device;",
        },
    ]

    selected = _select_actor_terms(defined_terms)

    assert {(row["celex"], row["term_normalized"]) for row in selected} == {
        ("32017R0745", "user"),
        ("32017R0746", "user"),
    }


def test_build_includes_edges_expands_composite_operator_role():
    actor_terms = [
        {
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": "32024R1689",
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "term": "deployer",
            "category": "actor",
            "term_normalized": "deployer",
            "celex": "32024R1689",
            "definition_text": "'deployer' means a natural or legal person;",
        },
        {
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": "32024R1689",
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": "32024R1689",
            "definition_text": "'operator' means a provider, deployer or importer;",
        },
    ]

    edges = _build_includes_edges(actor_terms)

    assert {edge["child_role_id"] for edge in edges if edge["parent_role_id"] == "32024R1689::role::operator"} == {
        "32024R1689::role::provider",
        "32024R1689::role::deployer",
        "32024R1689::role::importer",
    }


def test_select_actor_terms_promotes_nested_composite_roles():
    defined_terms = [
        {
            "defined_term_id": "dt_manufacturer",
            "term": "manufacturer",
            "category": "actor",
            "term_normalized": "manufacturer",
            "celex": "32024R1689",
            "definition_text": "'manufacturer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_provider",
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": "32024R1689",
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_deployer",
            "term": "deployer",
            "category": "actor",
            "term_normalized": "deployer",
            "celex": "32024R1689",
            "definition_text": "'deployer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_authrep",
            "term": "authorised representative",
            "category": "actor",
            "term_normalized": "authorised_representative",
            "celex": "32024R1689",
            "definition_text": "'authorised representative' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_importer",
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": "32024R1689",
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_distributor",
            "term": "distributor",
            "category": "actor",
            "term_normalized": "distributor",
            "celex": "32024R1689",
            "definition_text": "'distributor' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_product_manufacturer",
            "term": "product manufacturer",
            "category": "other",
            "term_normalized": "product_manufacturer",
            "celex": "32024R1689",
            "definition_text": "'product manufacturer' means a manufacturer of a product;",
        },
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": "32024R1689",
            "definition_text": "'operator' means a provider, product manufacturer, deployer, authorised representative, importer or distributor;",
        },
    ]

    selected = _select_actor_terms(defined_terms)

    assert {row["term_normalized"] for row in selected} >= {"operator"}


def test_augment_with_derived_roles_adds_helper_roles_for_present_celexes():
    actor_terms = [
        {
            "defined_term_id": "dt_provider",
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": "32024R1689",
            "regulation": "EU AI Act",
            "source_provision_id": "32024R1689_art_3_pt_3",
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": "32024R1689",
            "regulation": "EU AI Act",
            "source_provision_id": "32024R1689_art_3_pt_8",
            "definition_text": "'operator' means a provider, product manufacturer, deployer, authorised representative, importer or distributor;",
        },
        {
            "defined_term_id": "dt_economic_operator",
            "term": "economic operator",
            "category": "other",
            "term_normalized": "economic_operator",
            "celex": "32017R0745",
            "regulation": "MDR",
            "source_provision_id": "32017R0745_art_2_pt_35",
            "definition_text": "'economic operator' means a manufacturer, an authorised representative, an importer, a distributor or the person referred to in Article 22(1) and 22(3);",
        },
    ]

    augmented = _augment_with_derived_roles(actor_terms)

    assert {row["term_normalized"] for row in augmented} >= {
        "product_manufacturer",
        "article_22_person",
    }


def test_build_actor_roles_marks_derived_helpers_with_source_type():
    actor_terms = _augment_with_derived_roles([
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": "32024R1689",
            "regulation": "EU AI Act",
            "source_provision_id": "32024R1689_art_3_pt_8",
            "definition_text": "'operator' means a provider, product manufacturer, deployer, authorised representative, importer or distributor;",
        },
    ])

    roles = _build_actor_roles(actor_terms)
    role_by_term = {row["term_normalized"]: row for row in roles}

    assert role_by_term["operator"]["source_type"] == "defined_term"
    assert role_by_term["product_manufacturer"]["source_type"] == "derived_role"


def test_build_includes_edges_completes_mdr_economic_operator_with_article_22_person():
    actor_terms = [
        {
            "term": "manufacturer",
            "category": "actor",
            "term_normalized": "manufacturer",
            "celex": "32017R0745",
            "source_provision_id": "32017R0745_art_2_pt_31",
            "definition_text": "'manufacturer' means a natural or legal person;",
        },
        {
            "term": "authorised representative",
            "category": "actor",
            "term_normalized": "authorised_representative",
            "celex": "32017R0745",
            "source_provision_id": "32017R0745_art_2_pt_32",
            "definition_text": "'authorised representative' means a natural or legal person;",
        },
        {
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": "32017R0745",
            "source_provision_id": "32017R0745_art_2_pt_33",
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "term": "distributor",
            "category": "actor",
            "term_normalized": "distributor",
            "celex": "32017R0745",
            "source_provision_id": "32017R0745_art_2_pt_34",
            "definition_text": "'distributor' means a natural or legal person;",
        },
        {
            "term": "economic operator",
            "category": "other",
            "term_normalized": "economic_operator",
            "celex": "32017R0745",
            "source_provision_id": "32017R0745_art_2_pt_35",
            "definition_text": "'economic operator' means a manufacturer, an authorised representative, an importer, a distributor or the person referred to in Article 22(1) and 22(3);",
        },
        {
            "term": "Article 22 person",
            "category": "derived",
            "term_normalized": "article_22_person",
            "celex": "32017R0745",
            "source_provision_id": None,
            "definition_text": "",
            "source_type": "derived_role",
            "basis_note": "MDR economic operator definition references the person referred to in Article 22(1) and 22(3)",
        },
    ]

    edges = _build_includes_edges(actor_terms)

    assert {
        edge["child_role_id"] for edge in edges if edge["parent_role_id"] == "32017R0745::role::economic_operator"
    } == {
        "32017R0745::role::manufacturer",
        "32017R0745::role::authorised_representative",
        "32017R0745::role::importer",
        "32017R0745::role::distributor",
        "32017R0745::role::article_22_person",
    }


def test_build_equivalent_edges_are_classified_as_retrieval_analogies():
    edges = _build_equivalent_edges()

    deployer_to_user = next(
        edge for edge in edges
        if edge["left_role_id"] == "32024R1689::role::deployer"
        and edge["right_role_id"] == "32017R0745::role::user"
    )

    assert deployer_to_user["mapping_kind"] == "retrieval_analogy"
    assert deployer_to_user["scope"] == "healthcare_deployment"
    assert deployer_to_user["confidence"] == "curated"

    reverse_edge = next(
        edge for edge in edges
        if edge["left_role_id"] == "32017R0745::role::user"
        and edge["right_role_id"] == "32024R1689::role::deployer"
    )

    assert reverse_edge["mapping_kind"] == "retrieval_analogy"
