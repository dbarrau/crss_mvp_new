from domain.legislation_catalog import (
    MDR_CELEX,
    AI_ACT_CELEX,
    IVDR_CELEX,
    GDPR_CELEX,
    CIR_CELEX,
)
from canonicalization.role_linker import (
    _augment_with_derived_roles,
    _augment_with_standalone_roles,
    _build_actor_roles,
    _build_equivalent_edges,
    _build_includes_edges,
    _build_obligation_edges,
    _build_role_regex,
    _select_actor_terms,
    _title_is_role_named,
)
from domain.ontology.actor_roles import detect_role_specs


# Article 11 of the MDR/IVDR is titled exactly "Authorised representative" and
# opens with a conditional ("Where the manufacturer ... designates a sole
# authorised representative") that carries no modal verb, so _detect_modality
# misses it and the role's core duty article went unlinked. A title that *is*
# the role name supplies the obligation modality.
_AR_ART_11_TEXT = (
    "Where the manufacturer of a device is not established in a Member State, "
    "the device may only be placed on the Union market if the manufacturer "
    "designates a sole authorised representative."
)


def test_title_is_role_named_fires_only_on_bare_role_title():
    ar_regex = _build_role_regex("authorised representative")
    importer_regex = _build_role_regex("importer")
    manufacturer_regex = _build_role_regex("manufacturer")
    assert _title_is_role_named("Authorised representative", ar_regex)
    assert _title_is_role_named("The authorised representative", ar_regex)
    # Titles already covered by the "obligation" keyword path must not fire here.
    assert not _title_is_role_named("Obligations of importers", importer_regex)
    assert not _title_is_role_named(
        "General obligations of manufacturers", manufacturer_regex
    )
    assert not _title_is_role_named("", ar_regex)


def test_build_obligation_edges_links_role_named_title_without_modal():
    actor_terms = [
        {
            "term_normalized": "authorised_representative",
            "celex": MDR_CELEX,
            "term": "authorised representative",
        },
        {
            "term_normalized": "manufacturer",
            "celex": MDR_CELEX,
            "term": "manufacturer",
        },
    ]
    provisions = [{
        "id": f"{MDR_CELEX}_art_11",
        "celex": MDR_CELEX,
        "title": "Authorised representative",
        "text": _AR_ART_11_TEXT,
    }]

    edges = _build_obligation_edges(actor_terms, provisions)
    linked_roles = {e["role_id"] for e in edges}

    # The article's named role is linked …
    assert f"{MDR_CELEX}::role::authorised_representative" in linked_roles
    # … but a different role merely mentioned in the sentence is not.
    assert f"{MDR_CELEX}::role::manufacturer" not in linked_roles
    assert all(e["modality"] == "obligation" for e in edges)


def test_detect_role_specs_resolves_hospital_to_deployer_and_users():
    specs = detect_role_specs(
        "What must a hospital verify before putting a high-risk AI medical device into service?"
    )

    assert specs == [
        ("deployer", AI_ACT_CELEX),
        ("user", MDR_CELEX),
        ("user", IVDR_CELEX),
    ]


def test_detect_role_specs_honors_celex_filter():
    specs = detect_role_specs(
        "What must a hospital verify before putting a high-risk AI medical device into service?",
        target_celexes={AI_ACT_CELEX},
    )

    assert specs == [("deployer", AI_ACT_CELEX)]


def test_detect_role_specs_routes_gdpr_controller_and_processor():
    # GDPR controller/processor ActorRole nodes materialize automatically (their
    # Article 4 definitions contain "natural or legal person" -> category=actor);
    # the gap addressed here is purely query-time routing to them.
    assert detect_role_specs(
        "What obligations does a data controller have under GDPR?"
    ) == [("controller", GDPR_CELEX)]
    assert detect_role_specs("processor responsibilities") == [
        ("processor", GDPR_CELEX)
    ]
    assert detect_role_specs("duties of the supervisory authority") == [
        ("supervisory_authority", GDPR_CELEX)
    ]


def test_detect_role_specs_gdpr_honors_celex_filter():
    # GDPR roles must not leak when the question is scoped to another regulation.
    assert detect_role_specs("controller duties", target_celexes={MDR_CELEX}) == []


def test_select_actor_terms_promotes_composite_operator_definition():
    defined_terms = [
        {
            "defined_term_id": "dt_provider",
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": AI_ACT_CELEX,
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_deployer",
            "term": "deployer",
            "category": "actor",
            "term_normalized": "deployer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'deployer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_importer",
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": AI_ACT_CELEX,
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
            "celex": MDR_CELEX,
            "definition_text": "'user' means any healthcare professional or lay person who uses a device;",
        },
        {
            "defined_term_id": "dt_user_ivdr",
            "term": "user",
            "category": "other",
            "term_normalized": "user",
            "celex": IVDR_CELEX,
            "definition_text": "'user' means any healthcare professional or lay person who uses a device;",
        },
    ]

    selected = _select_actor_terms(defined_terms)

    assert {(row["celex"], row["term_normalized"]) for row in selected} == {
        (MDR_CELEX, "user"),
        (IVDR_CELEX, "user"),
    }


def test_build_includes_edges_expands_composite_operator_role():
    actor_terms = [
        {
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": AI_ACT_CELEX,
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "term": "deployer",
            "category": "actor",
            "term_normalized": "deployer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'deployer' means a natural or legal person;",
        },
        {
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": AI_ACT_CELEX,
            "definition_text": "'operator' means a provider, deployer or importer;",
        },
    ]

    edges = _build_includes_edges(actor_terms)

    assert {edge["child_role_id"] for edge in edges if edge["parent_role_id"] == f"{AI_ACT_CELEX}::role::operator"} == {
        f"{AI_ACT_CELEX}::role::provider",
        f"{AI_ACT_CELEX}::role::deployer",
        f"{AI_ACT_CELEX}::role::importer",
    }


def test_select_actor_terms_promotes_nested_composite_roles():
    defined_terms = [
        {
            "defined_term_id": "dt_manufacturer",
            "term": "manufacturer",
            "category": "actor",
            "term_normalized": "manufacturer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'manufacturer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_provider",
            "term": "provider",
            "category": "actor",
            "term_normalized": "provider",
            "celex": AI_ACT_CELEX,
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_deployer",
            "term": "deployer",
            "category": "actor",
            "term_normalized": "deployer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'deployer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_authrep",
            "term": "authorised representative",
            "category": "actor",
            "term_normalized": "authorised_representative",
            "celex": AI_ACT_CELEX,
            "definition_text": "'authorised representative' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_importer",
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_distributor",
            "term": "distributor",
            "category": "actor",
            "term_normalized": "distributor",
            "celex": AI_ACT_CELEX,
            "definition_text": "'distributor' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_product_manufacturer",
            "term": "product manufacturer",
            "category": "other",
            "term_normalized": "product_manufacturer",
            "celex": AI_ACT_CELEX,
            "definition_text": "'product manufacturer' means a manufacturer of a product;",
        },
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": AI_ACT_CELEX,
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
            "celex": AI_ACT_CELEX,
            "regulation": "EU AI Act",
            "source_provision_id": f"{AI_ACT_CELEX}_art_3_pt_3",
            "definition_text": "'provider' means a natural or legal person;",
        },
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": AI_ACT_CELEX,
            "regulation": "EU AI Act",
            "source_provision_id": f"{AI_ACT_CELEX}_art_3_pt_8",
            "definition_text": "'operator' means a provider, product manufacturer, deployer, authorised representative, importer or distributor;",
        },
        {
            "defined_term_id": "dt_economic_operator",
            "term": "economic operator",
            "category": "other",
            "term_normalized": "economic_operator",
            "celex": MDR_CELEX,
            "regulation": "MDR",
            "source_provision_id": f"{MDR_CELEX}_art_2_pt_35",
            "definition_text": "'economic operator' means a manufacturer, an authorised representative, an importer, a distributor or the person referred to in Article 22(1) and 22(3);",
        },
    ]

    augmented = _augment_with_derived_roles(actor_terms)

    assert {row["term_normalized"] for row in augmented} >= {
        "product_manufacturer",
        "article_22_person",
    }


def test_augment_with_standalone_roles_covers_definition_less_regulations():
    """CIR 2026/977 defines no terms of its own (definitions inherited from
    MDR/IVDR), so DefinedTerm-driven selection yields nothing for it — the
    standalone specs must materialize its roles regardless."""
    augmented = _augment_with_standalone_roles([])

    cir_rows = {row["term_normalized"]: row for row in augmented if row["celex"] == CIR_CELEX}
    assert {"manufacturer", "notified_body"} <= set(cir_rows)
    assert cir_rows["notified_body"]["source_type"] == "standalone_curated"
    assert cir_rows["notified_body"]["defined_term_id"] is None


def test_standalone_roles_do_not_duplicate_existing_terms():
    existing = [{
        "defined_term_id": "dt_x",
        "term": "manufacturer",
        "category": "actor",
        "term_normalized": "manufacturer",
        "celex": CIR_CELEX,
        "regulation": None,
        "source_provision_id": None,
        "definition_text": "",
    }]
    augmented = _augment_with_standalone_roles(existing)
    manufacturer_rows = [
        row for row in augmented
        if row["celex"] == CIR_CELEX and row["term_normalized"] == "manufacturer"
    ]
    assert len(manufacturer_rows) == 1
    assert manufacturer_rows[0]["defined_term_id"] == "dt_x"


def test_standalone_role_obligation_edges_link_cir_notified_body_duties():
    """The standard obligation heuristic must link CIR provisions addressed
    to the standalone notified-body role ('The notified body shall…') — also
    when the flattened article body opens with a paragraph marker ('1. '),
    which used to make the first-sentence check see only the literal '1.'."""
    actor_terms = _augment_with_standalone_roles([])
    provisions = [
        {
            "id": f"{CIR_CELEX}_art_2",
            "celex": CIR_CELEX,
            "kind": "article",
            "title": "Timelines",
            "text": "1. The notified body shall complete the conformity assessment within the timelines set out in the Annex.",
            "display_ref": "Article 2",
        },
        {
            "id": f"{CIR_CELEX}_art_9",
            "celex": CIR_CELEX,
            "kind": "article",
            "title": "Entry into force and application",
            "text": "This Regulation shall enter into force on the twentieth day following that of its publication.",
            "display_ref": "Article 9",
        },
    ]
    edges = _build_obligation_edges(actor_terms, provisions)
    linked = {(e["provision_id"], e["role_id"]) for e in edges}
    assert (f"{CIR_CELEX}_art_2", f"{CIR_CELEX}::role::notified_body") in linked
    assert not any(pid == f"{CIR_CELEX}_art_9" for pid, _ in linked)


def test_build_actor_roles_marks_derived_helpers_with_source_type():
    actor_terms = _augment_with_derived_roles([
        {
            "defined_term_id": "dt_operator",
            "term": "operator",
            "category": "other",
            "term_normalized": "operator",
            "celex": AI_ACT_CELEX,
            "regulation": "EU AI Act",
            "source_provision_id": f"{AI_ACT_CELEX}_art_3_pt_8",
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
            "celex": MDR_CELEX,
            "source_provision_id": f"{MDR_CELEX}_art_2_pt_31",
            "definition_text": "'manufacturer' means a natural or legal person;",
        },
        {
            "term": "authorised representative",
            "category": "actor",
            "term_normalized": "authorised_representative",
            "celex": MDR_CELEX,
            "source_provision_id": f"{MDR_CELEX}_art_2_pt_32",
            "definition_text": "'authorised representative' means a natural or legal person;",
        },
        {
            "term": "importer",
            "category": "actor",
            "term_normalized": "importer",
            "celex": MDR_CELEX,
            "source_provision_id": f"{MDR_CELEX}_art_2_pt_33",
            "definition_text": "'importer' means a natural or legal person;",
        },
        {
            "term": "distributor",
            "category": "actor",
            "term_normalized": "distributor",
            "celex": MDR_CELEX,
            "source_provision_id": f"{MDR_CELEX}_art_2_pt_34",
            "definition_text": "'distributor' means a natural or legal person;",
        },
        {
            "term": "economic operator",
            "category": "other",
            "term_normalized": "economic_operator",
            "celex": MDR_CELEX,
            "source_provision_id": f"{MDR_CELEX}_art_2_pt_35",
            "definition_text": "'economic operator' means a manufacturer, an authorised representative, an importer, a distributor or the person referred to in Article 22(1) and 22(3);",
        },
        {
            "term": "Article 22 person",
            "category": "derived",
            "term_normalized": "article_22_person",
            "celex": MDR_CELEX,
            "source_provision_id": None,
            "definition_text": "",
            "source_type": "derived_role",
            "basis_note": "MDR economic operator definition references the person referred to in Article 22(1) and 22(3)",
        },
    ]

    edges = _build_includes_edges(actor_terms)

    assert {
        edge["child_role_id"] for edge in edges if edge["parent_role_id"] == f"{MDR_CELEX}::role::economic_operator"
    } == {
        f"{MDR_CELEX}::role::manufacturer",
        f"{MDR_CELEX}::role::authorised_representative",
        f"{MDR_CELEX}::role::importer",
        f"{MDR_CELEX}::role::distributor",
        f"{MDR_CELEX}::role::article_22_person",
    }


def test_build_equivalent_edges_are_classified_as_retrieval_analogies():
    edges = _build_equivalent_edges()

    deployer_to_user = next(
        edge for edge in edges
        if edge["left_role_id"] == f"{AI_ACT_CELEX}::role::deployer"
        and edge["right_role_id"] == f"{MDR_CELEX}::role::user"
    )

    assert deployer_to_user["mapping_kind"] == "retrieval_analogy"
    assert deployer_to_user["scope"] == "healthcare_deployment"
    assert deployer_to_user["confidence"] == "curated"

    reverse_edge = next(
        edge for edge in edges
        if edge["left_role_id"] == f"{MDR_CELEX}::role::user"
        and edge["right_role_id"] == f"{AI_ACT_CELEX}::role::deployer"
    )

    assert reverse_edge["mapping_kind"] == "retrieval_analogy"
