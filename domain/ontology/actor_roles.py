"""Static knowledge for actor-role awareness in legislation retrieval.

This module keeps non-textual role knowledge separate from the graph
materialization logic:

- real-world entity synonyms (e.g. hospital -> deployer/user)
- curated cross-regulation retrieval alignments
- role-term normalization helpers
"""
from __future__ import annotations

import re


def normalize_role_term(term: str) -> str:
    """Normalize a role term to the DefinedTerm / ActorRole key format."""
    return re.sub(r"\s+", "_", term.strip().lower())


ROLE_MAPPING_KIND_RETRIEVAL_ANALOGY = "retrieval_analogy"
ROLE_SOURCE_TYPE_DEFINED_TERM = "defined_term"
ROLE_SOURCE_TYPE_DERIVED = "derived_role"


# Exact legal roles that should be promoted into ActorRole even when the
# parser-side category heuristic does not classify them as `actor`.
EXACT_LEGAL_ROLE_SPECS: dict[tuple[str, str], dict[str, str]] = {
    ("32017R0745", "user"): {
        "term": "user",
        "basis_note": "MDR Article 2 definition of user",
        "source_type": ROLE_SOURCE_TYPE_DEFINED_TERM,
    },
    ("32017R0746", "user"): {
        "term": "user",
        "basis_note": "IVDR Article 2 definition of user",
        "source_type": ROLE_SOURCE_TYPE_DEFINED_TERM,
    },
}


# Real-world entities and direct role mentions that should resolve to one or
# more regulation-specific actor roles at query time.
ENTITY_SYNONYMS: dict[str, list[tuple[str, str]]] = {
    # Operational entities
    "hospital": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "clinic": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "health institution": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "healthcare institution": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "healthcare provider": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    # Direct role terms
    "deployer": [("deployer", "32024R1689")],
    "provider": [("provider", "32024R1689")],
    "operator": [("operator", "32024R1689")],
    "product manufacturer": [("product manufacturer", "32024R1689")],
    "manufacturer": [
        ("manufacturer", "32024R1689"),
        ("manufacturer", "32017R0745"),
        ("manufacturer", "32017R0746"),
    ],
    "authorised representative": [
        ("authorised representative", "32024R1689"),
        ("authorised representative", "32017R0745"),
        ("authorised representative", "32017R0746"),
    ],
    "authorized representative": [
        ("authorised representative", "32024R1689"),
        ("authorised representative", "32017R0745"),
        ("authorised representative", "32017R0746"),
    ],
    "importer": [
        ("importer", "32024R1689"),
        ("importer", "32017R0745"),
        ("importer", "32017R0746"),
    ],
    "distributor": [
        ("distributor", "32024R1689"),
        ("distributor", "32017R0745"),
        ("distributor", "32017R0746"),
    ],
    "user": [("user", "32017R0745"), ("user", "32017R0746")],
}


# Derived graph helper roles that improve structural completeness but are not
# standalone formal definition points in the parsed legislation.
DERIVED_ROLE_SPECS: dict[tuple[str, str], dict[str, str]] = {
    ("32024R1689", "product_manufacturer"): {
        "term": "product manufacturer",
        "source_type": ROLE_SOURCE_TYPE_DERIVED,
        "basis_note": "AI Act operator component and Article 25(3) provider allocation",
    },
    ("32017R0745", "article_22_person"): {
        "term": "Article 22 person",
        "source_type": ROLE_SOURCE_TYPE_DERIVED,
        "basis_note": "MDR economic operator definition references the person referred to in Article 22(1) and 22(3)",
    },
}


# Curated role alignments used only for retrieval broadening. They should not
# be interpreted as strict legal identity in every context.
CROSS_REG_EQUIVALENCES: list[tuple[tuple[str, str], tuple[str, str], dict[str, str]]] = [
    (
        ("deployer", "32024R1689"),
        ("user", "32017R0745"),
        {
            "basis_note": "Healthcare deployment context",
            "mapping_kind": ROLE_MAPPING_KIND_RETRIEVAL_ANALOGY,
            "scope": "healthcare_deployment",
            "confidence": "curated",
        },
    ),
    (
        ("deployer", "32024R1689"),
        ("user", "32017R0746"),
        {
            "basis_note": "Healthcare deployment context",
            "mapping_kind": ROLE_MAPPING_KIND_RETRIEVAL_ANALOGY,
            "scope": "healthcare_deployment",
            "confidence": "curated",
        },
    ),
    (
        ("provider", "32024R1689"),
        ("manufacturer", "32017R0745"),
        {
            "basis_note": "Medical AI product supply chain",
            "mapping_kind": ROLE_MAPPING_KIND_RETRIEVAL_ANALOGY,
            "scope": "medical_device_supply_chain",
            "confidence": "curated",
        },
    ),
    (
        ("provider", "32024R1689"),
        ("manufacturer", "32017R0746"),
        {
            "basis_note": "Medical AI product supply chain",
            "mapping_kind": ROLE_MAPPING_KIND_RETRIEVAL_ANALOGY,
            "scope": "medical_device_supply_chain",
            "confidence": "curated",
        },
    ),
]


# Curated composite-role structure used when formal definitions describe one
# role in terms of other roles. This complements the textual heuristic and is
# especially useful where the legal wording is not reducible to a pure list.
COMPOSITE_ROLE_COMPONENTS: dict[tuple[str, str], list[str]] = {
    ("32024R1689", "operator"): [
        "provider",
        "product_manufacturer",
        "deployer",
        "authorised_representative",
        "importer",
        "distributor",
    ],
    ("32017R0745", "economic_operator"): [
        "manufacturer",
        "authorised_representative",
        "importer",
        "distributor",
        "article_22_person",
    ],
    ("32017R0746", "economic_operator"): [
        "manufacturer",
        "authorised_representative",
        "importer",
        "distributor",
    ],
}


COMPOSITE_ROLE_BASIS: dict[tuple[str, str], str] = {
    ("32024R1689", "operator"): "AI Act Article 3 definition of operator",
    ("32017R0745", "economic_operator"): "MDR Article 2 definition of economic operator",
    ("32017R0746", "economic_operator"): "IVDR Article 2 definition of economic operator",
}


def detect_role_specs(
    question: str,
    *,
    target_celexes: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(term_normalized, celex), ...]`` detected in *question*.

    Matching is deterministic, word-boundary based, and longest-first.
    """
    detected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    q_lower = question.lower()

    for phrase, specs in sorted(ENTITY_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        if not re.search(r"\b" + re.escape(phrase) + r"\b", q_lower):
            continue
        for term, celex in specs:
            if target_celexes and celex not in target_celexes:
                continue
            pair = (normalize_role_term(term), celex)
            if pair not in seen:
                seen.add(pair)
                detected.append(pair)

    return detected
