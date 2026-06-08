"""Legal reasoning chains — dependency relationships between EU regulatory provisions.

This module encodes the structural legal logic of EU AI Act, MDR, and IVDR
as explicit dependency edges.  These relationships are NOT derivable from
semantic similarity or co-citation proximity (the basis for Louvain community
structure); they are the explicit "if A then B" chains embedded in the
statutory text.

Edge types (maps to Neo4j relationship types):
    TRIGGERS_OBLIGATION_CLUSTER
        Meeting the conditions of the source article activates all articles
        in the target cluster as binding obligations.
        Example: classifying as a high-risk AI system under Article 6
        triggers the obligation cluster Articles 9–17, 43, 49, 72.

    IS_PREREQUISITE_FOR
        The source article must be satisfied / evaluated before the target
        article can be applied.
        Example: Article 3(63) (GPAI model definition) is a prerequisite
        for Article 53 (GPAI baseline obligations).

    REQUIRES_PRIOR_CHECK
        The source article's classification gate requires first verifying
        the target provision (typically an Annex or a definition article).
        Example: Article 6(1) requires checking Annex I first.

    DEROGATES_FROM
        The source article carves out an exception to the target article's
        general rule.
        Example: Article 6(3) derogates from Article 6(2) for certain
        low-risk Annex III systems.

Data structure:
    Each entry in _LEGAL_REASONING_EDGES is a dict with:
        celex       CELEX of the regulation (both source and target share it
                    unless cross_celex=True)
        type        One of the four edge type strings above
        source_ref  display_ref of the source provision
        target_refs List of display_ref strings for target provisions
        rationale   Short human-readable explanation (used in audit traces)

Usage:
    from domain.ontology.legal_reasoning_chains import get_edges_for_celex
    edges = get_edges_for_celex("32024R1689")
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegalReasoningEdge:
    """A single directed legal reasoning edge between two provisions."""

    celex: str
    type: str          # TRIGGERS_OBLIGATION_CLUSTER | IS_PREREQUISITE_FOR |
    #                    REQUIRES_PRIOR_CHECK | DEROGATES_FROM
    source_ref: str    # display_ref of the source provision
    target_refs: tuple[str, ...]  # display_refs of target provisions
    rationale: str = ""
    cross_celex: str | None = None  # target celex if different from source celex


# ---------------------------------------------------------------------------
# EU AI Act (32024R1689)
# ---------------------------------------------------------------------------

_AI_ACT_EDGES: list[LegalReasoningEdge] = [

    # ── Definition prerequisites ─────────────────────────────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 3",
        target_refs=("Article 6", "Article 51", "Article 53", "Article 16",
                     "Article 26", "Article 23", "Article 24"),
        rationale=(
            "Article 3 provides the formal definitions that determine actor "
            "status (provider, deployer, etc.) and product classification "
            "(AI system, GPAI model) — these must be evaluated before any "
            "classification or obligation article can be applied."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 3",
        target_refs=("Article 5",),
        rationale=(
            "Prohibited practices (Article 5) require evaluating whether "
            "the system is an 'AI system' under Article 3(1) and whether "
            "specific prohibited techniques (subliminal, social scoring, "
            "biometric categorisation) are met."
        ),
    ),

    # ── High-Risk AI classification gate ────────────────────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 6",
        target_refs=("Annex I", "Annex III"),
        rationale=(
            "Article 6(1) classification requires verifying Annex I "
            "(Union harmonisation legislation whose products may contain "
            "safety-component AI). Article 6(2) classification requires "
            "verifying Annex III (standalone high-risk use cases)."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="DEROGATES_FROM",
        source_ref="Article 6",
        target_refs=("Annex III",),
        rationale=(
            "Article 6(3) creates an exception to the Article 6(2)/Annex III "
            "route for systems performing narrow procedural tasks or preparatory "
            "tasks without material influence on outcomes — these remain outside "
            "the high-risk regime despite appearing in Annex III. "
            "Exception to the exception: profiling of natural persons under "
            "Article 6(3)(b) always remains high-risk."
        ),
    ),

    # ── High-Risk AI obligation cluster ─────────────────────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 6",
        target_refs=(
            "Article 9",   # Risk management system
            "Article 10",  # Data and data governance
            "Article 11",  # Technical documentation
            "Article 12",  # Record-keeping / logging
            "Article 13",  # Transparency and information to deployers
            "Article 14",  # Human oversight
            "Article 15",  # Accuracy, robustness, cybersecurity
            "Article 16",  # Provider obligations master list
            "Article 43",  # Conformity assessment
            "Article 47",  # EU declaration of conformity
            "Article 48",  # CE marking
            "Article 49",  # Registration in EU database
            "Article 72",  # Post-market monitoring
            "Article 73",  # Reporting of serious incidents
        ),
        rationale=(
            "Classification as a high-risk AI system under Article 6 activates "
            "the full Title III Chapter 2 obligation cluster for providers, "
            "plus post-market surveillance obligations in Title IX."
        ),
    ),

    # ── Provider master list cross-references ───────────────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 16",
        target_refs=(
            "Article 9", "Article 10", "Article 11", "Article 12",
            "Article 13", "Article 14", "Article 15", "Article 17",
            "Article 43", "Article 47", "Article 49", "Article 72",
        ),
        rationale=(
            "Article 16 is the master obligation list for high-risk AI "
            "providers. It enumerates all sub-obligations and cross-references "
            "each specific article. Evaluating Article 16 first ensures no "
            "obligation sub-cluster is missed."
        ),
    ),

    # ── QMS obligation (often missed, triggered by Article 16(g)) ───────
    LegalReasoningEdge(
        celex="32024R1689",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 16",
        target_refs=("Article 17",),
        rationale=(
            "Article 16(g) explicitly requires providers to implement a quality "
            "management system under Article 17. This is commonly overlooked "
            "because Article 17 is not in the Articles 9–15 block."
        ),
    ),

    # ── GPAI classification — two-tier structure ─────────────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 3",
        target_refs=("Article 53",),
        rationale=(
            "Article 3(63) defines 'general-purpose AI model'. Meeting this "
            "definition alone — regardless of systemic risk — triggers the "
            "Article 53 baseline obligations for all GPAI model providers."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 53",
        target_refs=(
            "Article 53",  # Baseline GPAI obligations (documentation, copyright policy)
            "Article 54",  # Codes of practice
        ),
        rationale=(
            "All GPAI model providers (meeting Article 3(63)) must comply with "
            "Article 53 baseline obligations. This is independent of and prior "
            "to any systemic risk assessment."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 51",
        target_refs=("Article 55",),
        rationale=(
            "Article 51 classification as a GPAI model with systemic risk "
            "is a prerequisite for Article 55 additional obligations. "
            "The 10^25 FLOP presumption threshold and Annex XIII criteria "
            "determine whether Article 51 is met."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 51",
        target_refs=("Annex XIII",),
        rationale=(
            "Article 51(1)(b) requires the Commission to assess against "
            "Annex XIII criteria (number of parameters, data quality/size, "
            "capabilities, societal impact) when determining systemic risk "
            "equivalence."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 51",
        target_refs=(
            "Article 55",  # Systemic risk obligations
            "Article 56",  # Authorised representative for GPAI with systemic risk
        ),
        rationale=(
            "Classification under Article 51 as a GPAI model with systemic "
            "risk triggers Article 55 additional obligations (model evaluation, "
            "adversarial testing, incident reporting) on top of the Article 53 "
            "baseline."
        ),
    ),

    # ── Deployer obligation cluster ──────────────────────────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 26",
        target_refs=(
            "Article 26",  # Deployer master list
        ),
        rationale=(
            "Article 26 is the master obligation list for high-risk AI deployers, "
            "covering human oversight implementation, instructions compliance, "
            "logging, fundamental rights impact assessment, and registration."
        ),
    ),

    # ── Prohibited practices gate (always evaluated before obligations) ──
    LegalReasoningEdge(
        celex="32024R1689",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 5",
        target_refs=("Article 6",),
        rationale=(
            "Article 5 (prohibited AI practices) must be evaluated before "
            "Article 6 classification. A system that falls under Article 5 "
            "is prohibited outright and cannot be placed on the market — "
            "the Article 6 classification exercise is moot."
        ),
    ),
]


# ---------------------------------------------------------------------------
# MDR 2017/745 (32017R0745)
# ---------------------------------------------------------------------------

_MDR_EDGES: list[LegalReasoningEdge] = [

    LegalReasoningEdge(
        celex="32017R0745",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 2",
        target_refs=("Article 10", "Article 52", "Article 61"),
        rationale=(
            "Article 2 MDR definitions (device, manufacturer, intended purpose, "
            "serious incident, etc.) are prerequisites for determining whether "
            "MDR obligations apply to a given product or actor."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0745",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 52",
        target_refs=("Annex VIII",),
        rationale=(
            "Article 52 (conformity assessment) requires applying the "
            "classification rules in Annex VIII to determine device class "
            "(I, IIa, IIb, III) before the applicable conformity assessment "
            "route can be determined."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0745",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 10",
        target_refs=(
            "Article 10",  # Manufacturer obligations master list
            "Article 11",  # Authorised representative
            "Article 13",  # General obligations importers
            "Article 14",  # General obligations distributors
        ),
        rationale=(
            "Article 10 is the manufacturer obligation master list under MDR. "
            "It covers QMS, post-market surveillance, vigilance, registration, "
            "conformity assessment, UDI labelling, and more."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0745",
        type="DEROGATES_FROM",
        source_ref="Article 5",
        target_refs=("Article 52",),
        rationale=(
            "Article 5(5) (in-house manufacture exemption) derogates from the "
            "standard conformity assessment route in Article 52 for health "
            "institutions manufacturing devices for internal use."
        ),
    ),
]


# ---------------------------------------------------------------------------
# IVDR 2017/746 (32017R0746)
# ---------------------------------------------------------------------------

_IVDR_EDGES: list[LegalReasoningEdge] = [

    LegalReasoningEdge(
        celex="32017R0746",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 2",
        target_refs=("Article 10", "Article 48"),
        rationale=(
            "Article 2 IVDR definitions are prerequisites for determining "
            "whether IVDR obligations apply."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0746",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 48",
        target_refs=("Annex VIII",),
        rationale=(
            "Article 48 (conformity assessment) requires applying the "
            "IVDR Annex VIII classification rules (Class A–D) before the "
            "applicable conformity assessment route is determined."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0746",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 10",
        target_refs=(
            "Article 10",  # Manufacturer obligations master list
            "Article 11",  # Authorised representative
            "Article 13",  # General obligations importers
            "Article 14",  # General obligations distributors
        ),
        rationale=(
            "Article 10 is the manufacturer obligation master list under IVDR."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0746",
        type="DEROGATES_FROM",
        source_ref="Article 5",
        target_refs=("Article 48",),
        rationale=(
            "Article 5(5) IVDR (in-house manufacture exemption) derogates from "
            "the standard conformity assessment route for health institutions."
        ),
    ),
]


# ---------------------------------------------------------------------------
# GDPR 2016/679 (32016R0679)
# ---------------------------------------------------------------------------

_GDPR_EDGES: list[LegalReasoningEdge] = [

    # ── Definition prerequisites ─────────────────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 4",
        target_refs=(
            "Article 5",   # Principles apply only to 'processing' of 'personal data'
            "Article 6",   # Lawfulness depends on who is 'controller' / what is 'processing'
            "Article 9",   # Special categories defined via 'genetic data', 'biometric data', etc.
            "Article 17",  # Right to erasure depends on lawful basis and 'controller' status
            "Article 20",  # Portability right depends on 'controller' and 'processing' definitions
            "Article 24",  # Controller responsibility requires identifying who is 'controller'
            "Article 26",  # Joint controllers require identifying multiple 'controllers'
            "Article 28",  # Processor contracts require identifying who is 'processor'
            "Article 35",  # DPIA scope depends on 'processing' and 'personal data'
        ),
        rationale=(
            "Article 4 GDPR definitions (personal data, processing, controller, "
            "processor, data subject, consent, health data, genetic data, biometric "
            "data, etc.) are formal prerequisites for all downstream obligation and "
            "rights articles. Without determining whether data is 'personal data' and "
            "whether an actor is a 'controller' or 'processor', no GDPR obligation "
            "can be correctly scoped."
        ),
    ),

    # ── Principles → Accountability chain ───────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 5",
        target_refs=("Article 24",),
        rationale=(
            "Article 5(2) enshrines the accountability principle: the controller "
            "shall be responsible for, and be able to demonstrate compliance with, "
            "Article 5(1). Article 24 operationalises this by requiring technical "
            "and organisational measures proportionate to the processing risks. "
            "Evaluating Article 5 first establishes which principles the "
            "Article 24 measures must demonstrate compliance with."
        ),
    ),

    # ── Lawfulness gate ──────────────────────────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 6",
        target_refs=(
            "Article 13",  # Must disclose lawful basis in information notice
            "Article 14",  # Must disclose lawful basis when data not collected from subject
            "Article 17",  # Right to erasure depends on lawful basis (e.g. consent withdrawal)
            "Article 20",  # Portability only applies to consent and contract bases
            "Article 21",  # Right to object only applies to public interest / legitimate interest bases
        ),
        rationale=(
            "Article 6(1) lawful basis must be established before the controller's "
            "transparency obligations (Articles 13/14) and data subject rights "
            "(Articles 17, 20, 21) can be correctly assessed. The applicable rights "
            "and obligations vary by lawful basis: portability (Article 20) only "
            "applies to consent or contract bases; the right to object (Article 21) "
            "only applies to public interest or legitimate interest bases."
        ),
    ),

    # ── Special categories gate (Article 9) ─────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 9",
        target_refs=(
            "Article 6",   # Art 9 derogation required IN ADDITION TO Art 6 lawful basis
            "Article 35",  # Large-scale special category processing triggers mandatory DPIA
            "Article 37",  # Large-scale special category processing triggers mandatory DPO
            "Article 89",  # Research exemptions for special categories reference Article 89
        ),
        rationale=(
            "Article 9(1) establishes a blanket prohibition on processing special "
            "categories (health data, genetic data, biometric data used for unique "
            "identification, racial/ethnic origin, etc.). Article 9(2) provides "
            "exhaustive derogations. Crucially, a derogation under Article 9(2) "
            "does NOT replace the Article 6 lawful basis requirement — both must "
            "be satisfied simultaneously. For medical device manufacturers, "
            "Article 9(2)(h) (healthcare/treatment) and Article 9(2)(i) (public "
            "health) are the most relevant derogations."
        ),
    ),
    LegalReasoningEdge(
        celex="32016R0679",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 9",
        target_refs=("Article 4",),
        rationale=(
            "Determining whether Article 9's prohibition applies requires checking "
            "Article 4 definitions: 'health data' (Article 4(15)), 'genetic data' "
            "(Article 4(13)), and 'biometric data' (Article 4(14)) define the "
            "special categories most relevant to medical device and AI system "
            "manufacturers."
        ),
    ),

    # ── Controller obligation cluster ────────────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 24",
        target_refs=(
            "Article 25",  # Data protection by design and by default
            "Article 28",  # Processor contracts (where processors are engaged)
            "Article 29",  # Processing under authority of controller or processor
            "Article 30",  # Records of processing activities
            "Article 32",  # Security of processing
            "Article 33",  # Notification of breach to supervisory authority (72-hour)
            "Article 34",  # Communication of breach to data subjects
            "Article 35",  # Data protection impact assessment
            "Article 37",  # Designation of DPO
        ),
        rationale=(
            "Article 24 is the controller accountability master provision. "
            "Identifying an actor as a 'controller' triggers the full Chapter IV "
            "obligation cluster: privacy-by-design (Article 25), processor "
            "contracts (Article 28), records of processing (Article 30), "
            "security (Article 32), breach notification (Articles 33/34), "
            "DPIA (Article 35), and DPO designation (Article 37)."
        ),
    ),

    # ── DPIA gate → Prior Consultation ───────────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 35",
        target_refs=("Article 36",),
        rationale=(
            "Where the DPIA indicates that the processing would result in a high "
            "residual risk despite the controller's mitigation measures, Article 36 "
            "mandates prior consultation with the competent supervisory authority "
            "before commencing processing. For medical devices processing health "
            "data at scale, or AI-enabled devices using biometric data, a mandatory "
            "DPIA under Article 35(3)(b) is almost always triggered."
        ),
    ),
    LegalReasoningEdge(
        celex="32016R0679",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 35",
        target_refs=("Article 9",),
        rationale=(
            "Article 35(3)(b) mandates a DPIA specifically when processing is "
            "carried out on a large scale involving special categories under "
            "Article 9(1). Verifying whether the processing involves health data, "
            "genetic data, or biometric data is therefore a required prior check "
            "before determining whether Article 35(3)(b) triggers a mandatory DPIA."
        ),
    ),

    # ── DPO designation triggers ──────────────────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 37",
        target_refs=(
            "Article 38",  # Position of DPO
            "Article 39",  # Tasks of DPO
        ),
        rationale=(
            "Article 37(1)(c) requires designation of a DPO when the core "
            "activities of the controller or processor involve large-scale "
            "processing of special categories under Article 9. Medical device "
            "manufacturers and digital health platforms frequently meet this "
            "threshold. Designation triggers Articles 38 (independence, resources) "
            "and 39 (mandatory tasks: monitoring compliance, cooperating with SA, "
            "acting as contact point for data subjects)."
        ),
    ),

    # ── Transparency and data subject rights cluster ─────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="TRIGGERS_OBLIGATION_CLUSTER",
        source_ref="Article 12",
        target_refs=(
            "Article 13",  # Information when data collected from the data subject
            "Article 14",  # Information when data not collected from the data subject
            "Article 15",  # Right of access
            "Article 16",  # Right to rectification
            "Article 17",  # Right to erasure ('right to be forgotten')
            "Article 18",  # Right to restriction of processing
            "Article 19",  # Notification obligation re: rectification/erasure/restriction
            "Article 20",  # Right to data portability
            "Article 21",  # Right to object
            "Article 22",  # Automated individual decision-making including profiling
        ),
        rationale=(
            "Article 12 establishes the general framework for transparent "
            "communication and the modalities for exercising data subject rights "
            "(form, timing, fee policy, identity verification). It applies to all "
            "information and communications under Articles 13-22. Evaluating "
            "Article 12 first ensures the correct procedural framework is applied "
            "whenever any of the Articles 13-22 obligations or rights are triggered."
        ),
    ),

    # ── Automated decision-making (AI relevance) ──────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="IS_PREREQUISITE_FOR",
        source_ref="Article 22",
        target_refs=("Article 9",),
        rationale=(
            "Article 22(4) prohibits automated decisions based solely on "
            "processing of special categories of data (Article 9) unless "
            "explicit consent or substantial public interest grounds exist. "
            "AI-enabled medical devices performing automated diagnostic decisions "
            "based on health data must therefore evaluate both Article 22 and "
            "Article 9 together. This is directly relevant to AI Act high-risk "
            "AI systems in the medical device domain."
        ),
    ),

    # ── Research / healthcare derogation ─────────────────────────────────
    LegalReasoningEdge(
        celex="32016R0679",
        type="DEROGATES_FROM",
        source_ref="Article 89",
        target_refs=("Article 5", "Article 9"),
        rationale=(
            "Article 89 provides the conditions under which Member State law may "
            "restrict data subject rights and derogate from certain Article 5 "
            "principles (purpose limitation, storage limitation) for archiving, "
            "scientific/historical research, and statistical purposes. This is the "
            "basis for Article 9(2)(j) special category derogation for medical "
            "research. Clinical investigations under MDR Article 62 and performance "
            "studies under IVDR Article 58 commonly rely on this pathway."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Cross-regulation chains (AI Act ↔ MDR/IVDR)
# ---------------------------------------------------------------------------

_CROSS_REG_EDGES: list[LegalReasoningEdge] = [

    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 43",
        target_refs=("Article 52",),
        cross_celex="32017R0745",
        rationale=(
            "Article 43(3) AI Act: when a high-risk AI system is a safety "
            "component of a medical device, the MDR/IVDR conformity assessment "
            "under Article 52 MDR applies instead of the AI Act's own "
            "conformity assessment procedure. MDR Article 52 is therefore "
            "a prerequisite check before applying AI Act Article 43(3)."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 43",
        target_refs=("Article 48",),
        cross_celex="32017R0746",
        rationale=(
            "Same as above but for IVDR devices: AI Act Article 43(3) "
            "defers to IVDR Article 48 conformity assessment when the "
            "AI system is a safety component of an in vitro diagnostic device."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Cross-regulation chains involving GDPR
# ---------------------------------------------------------------------------

_GDPR_CROSS_REG_EDGES: list[LegalReasoningEdge] = [

    # ── GDPR ↔ MDR: post-market surveillance and clinical investigations ──
    LegalReasoningEdge(
        celex="32017R0745",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 10",
        target_refs=("Article 9",),
        cross_celex="32016R0679",
        rationale=(
            "MDR Article 10(9) requires manufacturers to establish a post-market "
            "surveillance (PMS) system collecting patient safety and clinical "
            "performance data. Where PMS data constitutes health data relating to "
            "identified or identifiable patients, GDPR Article 9 applies. A lawful "
            "basis under GDPR Article 9(2) — commonly Article 9(2)(h) for healthcare "
            "purposes or Article 9(2)(i) for public health — must be established "
            "before the MDR PMS system can lawfully collect patient-level data."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0745",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 61",
        target_refs=("Article 35",),
        cross_celex="32016R0679",
        rationale=(
            "MDR Article 61 (clinical evaluation) and MDR Article 62 (clinical "
            "investigations) involve systematic collection of patient health data. "
            "Such large-scale processing of special category data triggers a "
            "mandatory DPIA under GDPR Article 35(3)(b). The DPIA must be "
            "completed before the clinical investigation commences and should "
            "address the specific risks arising from processing sensitive health "
            "data in a research context."
        ),
    ),

    # ── GDPR ↔ IVDR: performance studies ────────────────────────────────
    LegalReasoningEdge(
        celex="32017R0746",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 10",
        target_refs=("Article 9",),
        cross_celex="32016R0679",
        rationale=(
            "IVDR Article 10(9) mirrors MDR Article 10(9): IVDR manufacturers must "
            "establish PMS systems collecting patient and sample-level data. "
            "Processing of diagnostic results linked to identified patients "
            "constitutes health data under GDPR Article 4(15), triggering Article 9's "
            "special category prohibition and requiring a lawful derogation under "
            "Article 9(2)."
        ),
    ),
    LegalReasoningEdge(
        celex="32017R0746",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 58",
        target_refs=("Article 35",),
        cross_celex="32016R0679",
        rationale=(
            "IVDR Article 58 (performance studies) involves collection of patient "
            "samples and health data for IVD validation. This constitutes large-scale "
            "processing of special categories under GDPR, triggering a mandatory DPIA "
            "under Article 35(3)(b) and potentially prior consultation under "
            "Article 36 if residual risk remains high."
        ),
    ),

    # ── GDPR ↔ AI Act: DPIA / risk management overlap ───────────────────
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 9",
        target_refs=("Article 35",),
        cross_celex="32016R0679",
        rationale=(
            "AI Act Article 9 (risk management system for high-risk AI) and GDPR "
            "Article 35 (DPIA) are parallel but distinct obligations that frequently "
            "overlap for AI systems processing personal data. MDCG 2025-6 explicitly "
            "addresses this interplay for medical device AI. The AI Act risk "
            "management system does not substitute for a GDPR DPIA — both must be "
            "conducted. However, AI Act risk management documentation can inform "
            "and partially satisfy the DPIA's risk assessment requirements."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 10",
        target_refs=("Article 9",),
        cross_celex="32016R0679",
        rationale=(
            "AI Act Article 10(5) permits processing of special categories of "
            "personal data (including health data and biometric data) for bias "
            "monitoring, detection, and correction in high-risk AI systems, to the "
            "extent strictly necessary. This permission under AI Act does NOT create "
            "a GDPR lawful basis — the controller must still establish a derogation "
            "under GDPR Article 9(2), most likely Article 9(2)(g) (substantial "
            "public interest) or Article 9(2)(j) (research purposes subject to "
            "Article 89 safeguards)."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 6",
        target_refs=("Article 9",),
        cross_celex="32016R0679",
        rationale=(
            "AI Act Annex III items 1 and 6 include biometric identification and "
            "categorisation systems as high-risk AI. Biometric data used for unique "
            "identification of natural persons is a special category under GDPR "
            "Article 9(1). Deploying a high-risk AI system in this domain therefore "
            "requires a GDPR Article 9(2) derogation in addition to the AI Act "
            "conformity assessment under Article 43."
        ),
    ),
    LegalReasoningEdge(
        celex="32024R1689",
        type="REQUIRES_PRIOR_CHECK",
        source_ref="Article 26",
        target_refs=("Article 28",),
        cross_celex="32016R0679",
        rationale=(
            "AI Act Article 26 imposes obligations on deployers of high-risk AI "
            "systems. Where the deployer processes personal data using the AI system "
            "on behalf of a controller, the deployer may also qualify as a "
            "'processor' under GDPR Article 4(8), requiring a data processing "
            "agreement under GDPR Article 28. The AI Act deployer role and the GDPR "
            "processor role are independent but may coincide — both sets of "
            "obligations apply simultaneously when they do."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Aggregated access
# ---------------------------------------------------------------------------

_ALL_EDGES: list[LegalReasoningEdge] = (
    _AI_ACT_EDGES + _MDR_EDGES + _IVDR_EDGES + _GDPR_EDGES
    + _CROSS_REG_EDGES + _GDPR_CROSS_REG_EDGES
)

# Lookup: source_ref → list of edges (within a celex)
_EDGE_INDEX: dict[tuple[str, str], list[LegalReasoningEdge]] = {}
for _edge in _ALL_EDGES:
    key = (_edge.celex, _edge.source_ref)
    _EDGE_INDEX.setdefault(key, []).append(_edge)


def get_edges_for_celex(celex: str) -> list[LegalReasoningEdge]:
    """Return all legal reasoning edges for a given regulation."""
    return [e for e in _ALL_EDGES if e.celex == celex]


def get_edges_from(source_ref: str, celex: str) -> list[LegalReasoningEdge]:
    """Return all outgoing edges from a specific provision."""
    return _EDGE_INDEX.get((celex, source_ref), [])


def get_obligation_chain(
    source_ref: str,
    celex: str,
    *,
    edge_types: set[str] | None = None,
    max_depth: int = 2,
) -> list[tuple[str, str, str]]:
    """Return ``[(target_ref, celex, edge_type), ...]`` reachable from source.

    Performs a breadth-first traversal of the legal reasoning graph starting
    from *source_ref*.  Useful for chain retrieval: given a classification
    article, returns the full set of provisions that must be retrieved to
    give the LLM a structurally complete context.

    Parameters
    ----------
    source_ref:
        display_ref to start from (e.g. ``"Article 6"``).
    celex:
        Regulation CELEX (e.g. ``"32024R1689"``).
    edge_types:
        Optional filter on edge types.  Defaults to all types.
    max_depth:
        Maximum BFS depth.  Keep at 2 to avoid pulling the entire regulation.
    """
    allowed = edge_types or {
        "TRIGGERS_OBLIGATION_CLUSTER",
        "IS_PREREQUISITE_FOR",
        "REQUIRES_PRIOR_CHECK",
        "DEROGATES_FROM",
    }
    visited: set[tuple[str, str]] = set()
    result: list[tuple[str, str, str]] = []
    queue: list[tuple[str, str, int]] = [(source_ref, celex, 0)]

    while queue:
        ref, cx, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for edge in _EDGE_INDEX.get((cx, ref), []):
            if edge.type not in allowed:
                continue
            target_cx = edge.cross_celex or cx
            for target_ref in edge.target_refs:
                key = (target_ref, target_cx)
                if key not in visited:
                    visited.add(key)
                    result.append((target_ref, target_cx, edge.type))
                    queue.append((target_ref, target_cx, depth + 1))

    return result
