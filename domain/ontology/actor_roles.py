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
# Same role term carried into an implementing act whose definitions are
# inherited from the basic act — legal identity, not mere analogy.
ROLE_MAPPING_KIND_IMPLEMENTING_ACT = "implementing_act_identity"
ROLE_SOURCE_TYPE_DEFINED_TERM = "defined_term"
ROLE_SOURCE_TYPE_DERIVED = "derived_role"
# Curated role with no DefinedTerm node of its own in that regulation
# (e.g. an implementing regulation that inherits the basic act's definitions).
ROLE_SOURCE_TYPE_STANDALONE = "standalone_curated"


# ---------------------------------------------------------------------------
# Canonical actor-role registry — the single source of truth
# ---------------------------------------------------------------------------
# Maps each canonical actor-role term to the regulations (CELEX) in which it is
# a recognized role. This is the authority the other actor lexicons are checked
# against (see ``tests/test_actor_lexicon_consistency.py``) so they cannot
# silently drift apart — the failure mode that left GDPR ``supervisory
# authority`` routable but undetectable by the obligation rule:
#
#   * ``ENTITY_SYNONYMS`` (below) routes real-world phrases to roles; every
#     target (role, celex) MUST appear here.
#   * ``provision_roles._ACTOR_SUBJECTS`` is the obligation-rule subject list;
#     every canonical role here MUST be matched by it (it may carry extra
#     non-role subjects such as "Member State" / "person").
#
# NOTE: ``defined_terms.ACTOR_SIGNALS`` is a *different axis* — definition-body
# signal phrases ("natural or legal person") used to classify a freshly parsed
# DefinedTerm as an actor. It is intentionally not reconciled with this set.
CANONICAL_ACTOR_ROLES: dict[str, frozenset[str]] = {
    "provider": frozenset({"32024R1689"}),
    "deployer": frozenset({"32024R1689"}),
    "operator": frozenset({"32024R1689"}),
    "product manufacturer": frozenset({"32024R1689"}),
    "manufacturer": frozenset({"32024R1689", "32017R0745", "32017R0746", "32026R0977"}),
    "authorised representative": frozenset({"32024R1689", "32017R0745", "32017R0746"}),
    "importer": frozenset({"32024R1689", "32017R0745", "32017R0746"}),
    "distributor": frozenset({"32024R1689", "32017R0745", "32017R0746"}),
    "user": frozenset({"32017R0745", "32017R0746"}),
    # notified body: defined in MDR/IVDR/AI-Act (DefinedTerm category 'body',
    # promoted via EXACT_LEGAL_ROLE_SPECS) and an obligation-bearer in CIR
    # 2026/977 (standalone; the CIR inherits the MDR/IVDR definitions).
    "notified body": frozenset({"32017R0745", "32017R0746", "32024R1689", "32026R0977"}),
    "controller": frozenset({"32016R0679"}),
    "processor": frozenset({"32016R0679"}),
    "supervisory authority": frozenset({"32016R0679"}),
}


def is_known_actor_role(term: str, celex: str) -> bool:
    """Return whether *term* is a registered actor role in *celex*."""
    return celex in CANONICAL_ACTOR_ROLES.get(term.strip().lower(), frozenset())


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
    # Notified bodies are classified category='body' by the parser heuristic,
    # so they were never promoted — leaving the ENTITY_SYNONYMS "notified
    # body" mapping dead and their substantial obligation sets (MDR/IVDR
    # Annex VII, MDR Arts 44-46, AI Act Arts 31-34) unlinked.
    ("32017R0745", "notified_body"): {
        "term": "notified body",
        "basis_note": "MDR Article 2(42) definition of notified body",
        "source_type": ROLE_SOURCE_TYPE_DEFINED_TERM,
    },
    ("32017R0746", "notified_body"): {
        "term": "notified body",
        "basis_note": "IVDR Article 2(34) definition of notified body",
        "source_type": ROLE_SOURCE_TYPE_DEFINED_TERM,
    },
    ("32024R1689", "notified_body"): {
        "term": "notified body",
        "basis_note": "AI Act Article 3(22) definition of notified body",
        "source_type": ROLE_SOURCE_TYPE_DEFINED_TERM,
    },
}


# Roles that exist in a regulation without any DefinedTerm node of their own —
# implementing acts inherit the basic act's definitions, so the parser finds
# no "'X' means…" point to extract. The role_linker materializes these as
# ActorRole nodes directly; the standard obligation heuristic then links the
# regulation's actor-addressed provisions ("The notified body shall…") as for
# any other role. Without this, an implementing regulation has zero ActorRole
# nodes and the role-obligation channel is silently empty for it.
STANDALONE_ROLE_SPECS: dict[tuple[str, str], dict[str, str]] = {
    ("32026R0977", "manufacturer"): {
        "term": "manufacturer",
        "source_type": ROLE_SOURCE_TYPE_STANDALONE,
        "basis_note": (
            "CIR 2026/977 implements MDR/IVDR notified-body (re)certification "
            "procedures; 'manufacturer' is inherited from MDR Article 2(30) / "
            "IVDR Article 2(23)."
        ),
    },
    ("32026R0977", "notified_body"): {
        "term": "notified body",
        "source_type": ROLE_SOURCE_TYPE_STANDALONE,
        "basis_note": (
            "CIR 2026/977 addresses its procedural obligations (quotations, "
            "timelines, re-certification decisions) primarily to notified "
            "bodies; the term is inherited from MDR Article 2(42) / IVDR "
            "Article 2(34)."
        ),
    },
}


# Real-world entities and direct role mentions that should resolve to one or
# more regulation-specific actor roles at query time.
ENTITY_SYNONYMS: dict[str, list[tuple[str, str]]] = {
    # Operational entities
    "hospital": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "hospitals": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "clinic": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "clinics": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "health institution": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "health institutions": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "healthcare institution": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "healthcare institutions": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "healthcare provider": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    "healthcare providers": [("deployer", "32024R1689"), ("user", "32017R0745"), ("user", "32017R0746")],
    # Direct role terms
    "deployer": [("deployer", "32024R1689")],
    "deployers": [("deployer", "32024R1689")],
    "provider": [("provider", "32024R1689")],
    "providers": [("provider", "32024R1689")],
    "operator": [("operator", "32024R1689")],
    "operators": [("operator", "32024R1689")],
    "product manufacturer": [("product manufacturer", "32024R1689")],
    "product manufacturers": [("product manufacturer", "32024R1689")],
    "manufacturer": [
        ("manufacturer", "32024R1689"),
        ("manufacturer", "32017R0745"),
        ("manufacturer", "32017R0746"),
        ("manufacturer", "32026R0977"),
    ],
    "manufacturers": [
        ("manufacturer", "32024R1689"),
        ("manufacturer", "32017R0745"),
        ("manufacturer", "32017R0746"),
        ("manufacturer", "32026R0977"),
    ],
    "authorised representative": [
        ("authorised representative", "32024R1689"),
        ("authorised representative", "32017R0745"),
        ("authorised representative", "32017R0746"),
    ],
    "authorised representatives": [
        ("authorised representative", "32024R1689"),
        ("authorised representative", "32017R0745"),
        ("authorised representative", "32017R0746"),
    ],
    "authorized representative": [
        ("authorised representative", "32024R1689"),
        ("authorised representative", "32017R0745"),
        ("authorised representative", "32017R0746"),
    ],
    "authorized representatives": [
        ("authorised representative", "32024R1689"),
        ("authorised representative", "32017R0745"),
        ("authorised representative", "32017R0746"),
    ],
    "importer": [
        ("importer", "32024R1689"),
        ("importer", "32017R0745"),
        ("importer", "32017R0746"),
    ],
    "importers": [
        ("importer", "32024R1689"),
        ("importer", "32017R0745"),
        ("importer", "32017R0746"),
    ],
    "distributor": [
        ("distributor", "32024R1689"),
        ("distributor", "32017R0745"),
        ("distributor", "32017R0746"),
    ],
    "distributors": [
        ("distributor", "32024R1689"),
        ("distributor", "32017R0745"),
        ("distributor", "32017R0746"),
    ],
    "user": [("user", "32017R0745"), ("user", "32017R0746")],
    "users": [("user", "32017R0745"), ("user", "32017R0746")],
    "notified body": [
        ("notified body", "32017R0745"),
        ("notified body", "32017R0746"),
        ("notified body", "32024R1689"),
        ("notified body", "32026R0977"),
    ],
    "notified bodies": [
        ("notified body", "32017R0745"),
        ("notified body", "32017R0746"),
        ("notified body", "32024R1689"),
        ("notified body", "32026R0977"),
    ],
    # GDPR actors. These ActorRole nodes are materialized automatically (their
    # Article 4 definitions contain "natural or legal person" -> category=actor),
    # so the only gap was query-time routing to them. "data controller" /
    # "data processor" are listed (longest-first match wins) for the common
    # real-world phrasings. NB: "data subject" is intentionally absent — it has
    # no ActorRole node (defined parenthetically in Art 4(1), not as a
    # standalone "'data subject' means" point), so a synonym would be a dead
    # mapping until extraction is extended.
    "controller": [("controller", "32016R0679")],
    "controllers": [("controller", "32016R0679")],
    "data controller": [("controller", "32016R0679")],
    "data controllers": [("controller", "32016R0679")],
    "processor": [("processor", "32016R0679")],
    "processors": [("processor", "32016R0679")],
    "data processor": [("processor", "32016R0679")],
    "data processors": [("processor", "32016R0679")],
    "supervisory authority": [("supervisory authority", "32016R0679")],
    "supervisory authorities": [("supervisory authority", "32016R0679")],
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
    # CIR 2026/977 inherits the MDR/IVDR definitions, so its roles are the
    # same legal persons — a question seeded on the MDR/IVDR role should also
    # surface the implementing regulation's procedural obligations.
    (
        ("manufacturer", "32026R0977"),
        ("manufacturer", "32017R0745"),
        {
            "basis_note": "CIR 2026/977 implements MDR; identical manufacturer role",
            "mapping_kind": ROLE_MAPPING_KIND_IMPLEMENTING_ACT,
            "scope": "notified_body_certification",
            "confidence": "curated",
        },
    ),
    (
        ("manufacturer", "32026R0977"),
        ("manufacturer", "32017R0746"),
        {
            "basis_note": "CIR 2026/977 implements IVDR; identical manufacturer role",
            "mapping_kind": ROLE_MAPPING_KIND_IMPLEMENTING_ACT,
            "scope": "notified_body_certification",
            "confidence": "curated",
        },
    ),
    (
        ("notified body", "32026R0977"),
        ("notified body", "32017R0745"),
        {
            "basis_note": "CIR 2026/977 implements MDR; identical notified body role",
            "mapping_kind": ROLE_MAPPING_KIND_IMPLEMENTING_ACT,
            "scope": "notified_body_certification",
            "confidence": "curated",
        },
    ),
    (
        ("notified body", "32026R0977"),
        ("notified body", "32017R0746"),
        {
            "basis_note": "CIR 2026/977 implements IVDR; identical notified body role",
            "mapping_kind": ROLE_MAPPING_KIND_IMPLEMENTING_ACT,
            "scope": "notified_body_certification",
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


_ENTITY_CONTEXT_PHRASES: tuple[str, ...] = (
    "hospital",
    "clinic",
    "health institution",
    "healthcare institution",
    "healthcare provider",
)

_PROVIDER_ACTION_RE = re.compile(
    r"\b(develop(?:s|ed|ing)?|build(?:s|ing)?|train(?:s|ed|ing)?|"
    r"put(?:s|ting)?\s+(?:an?\s+)?(?:ai\s+system\s+)?into\s+service|"
    r"place(?:s|d|ing)?\s+(?:an?\s+)?(?:ai\s+system\s+)?on\s+the\s+market)\b",
    re.I,
)

_MANUFACTURER_ACTION_RE = re.compile(
    r"\b(develop(?:s|ed|ing)?|manufactur(?:e|es|ed|ing)|produce(?:s|d|ing)|design(?:s|ed|ing)?)\b",
    re.I,
)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether *phrase* appears in *text* as a whole phrase."""
    return bool(re.search(r"\b" + re.escape(phrase) + r"\b", text))


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

    def add(term: str, celex: str) -> None:
        if target_celexes and celex not in target_celexes:
            return
        pair = (normalize_role_term(term), celex)
        if pair not in seen:
            seen.add(pair)
            detected.append(pair)

    for phrase, specs in sorted(ENTITY_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        if not re.search(r"\b" + re.escape(phrase) + r"\b", q_lower):
            continue
        for term, celex in specs:
            add(term, celex)

    has_entity_context = any(_contains_phrase(q_lower, phrase) for phrase in _ENTITY_CONTEXT_PHRASES)
    if has_entity_context and _PROVIDER_ACTION_RE.search(q_lower):
        add("provider", "32024R1689")
    if has_entity_context and _MANUFACTURER_ACTION_RE.search(q_lower):
        add("manufacturer", "32017R0745")
        add("manufacturer", "32017R0746")

    return detected
