"""Question routing — deterministic classification of user queries.

Analyses a question's surface features (mentioned regulations, actor roles,
explicit provision refs, keyword signals) to select one of a small set of
retrieval routes.  No LLM calls; no retriever I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from domain.ontology.actor_roles import detect_role_specs as _detect_role_specs

# ---------------------------------------------------------------------------
# Route and target dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QuestionRoute:
    """Deterministic routing decision for the question-answer pipeline."""

    id: str
    label: str
    rationale: str


@dataclass(frozen=True)
class _ProvisionLookupTarget:
    """A provision reference that should be retrieved with a specific scope."""

    ref: str
    celexes: frozenset[str] | None = None


# ---------------------------------------------------------------------------
# Definition-question patterns
# ---------------------------------------------------------------------------

# Patterns that signal the user is asking for the meaning/definition of a
# concept.  Each pattern must contain a named group ``concept`` that
# captures the subject term.
_DEFINITION_Q_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bwhat\s+is\s+(?:an?\s+)?(?:the\s+)?(?:concept\s+of\s+)?(?P<concept>[^?.,]+)", re.I),
    re.compile(r"\bwhat\s+are\s+(?:the\s+)?(?P<concept>[^?.,]+)", re.I),
    re.compile(r"\bdefin(?:e|ition\s+of)\s+(?P<concept>[^?.,]+)", re.I),
    re.compile(r"\bwhat\s+does\s+(?P<concept>[^?.,]+?)\s+mean\b", re.I),
    re.compile(r"\bhow\s+is\s+(?P<concept>[^?.,]+?)\s+defined\b", re.I),
    re.compile(r"\bwhat\s+is\s+meant\s+by\s+(?P<concept>[^?.,]+)", re.I),
]

# ---------------------------------------------------------------------------
# Route-selection signal regexes
# ---------------------------------------------------------------------------

_OBLIGATION_Q_RE = re.compile(
    r"\b(shall|must|obligation|obligations|required|requirement|"
    r"responsib(?:le|ility)|duty|duties|has\s+to|have\s+to|"
    r"need(?:s)?\s+to)\b",
    re.I,
)

_CROSS_REG_Q_RE = re.compile(
    r"\b(interact|interaction|overlap|overlaps|relationship|"
    r"relate|combined|compare|compared|versus|vs\.?|both|across)\b",
    re.I,
)

_QUALIFICATION_Q_RE = re.compile(
    r"\b(high-risk|classif(?:y|ication)|qualif(?:y|ication)|"
    r"transition|become|status|substantial\s+modification|"
    r"provider|deployer|manufacturer|user|exemption|article\s+5\(5\)|"
    r"continuous\s+learning|retrain(?:ing)?)\b",
    re.I,
)

_ROLE_TRANSITION_Q_RE = re.compile(
    r"\b(become|transition|transitions|convert|converted|at\s+what\s+stage|"
    r"when\s+does|cease|ceases|lose|loses|losing)\b",
    re.I,
)

_MODIFICATION_Q_RE = re.compile(
    r"\b(substantial\s+modification|modify|modifies|modified|modification|"
    r"continuous\s+learning|retrain(?:ing)?)\b",
    re.I,
)

# Signals that the user is asking about the classification criteria or
# obligation chain that flows FROM a classification gate article.
# Triggers chain retrieval through TRIGGERS_OBLIGATION_CLUSTER edges.
_CHAIN_CLASSIFICATION_Q_RE = re.compile(
    r"\b(classif(?:y|ication|ied)|trigger(?:s|ed|ing)?|oblig(?:ation|ations)\s+"
    r"(?:that\s+)?(?:flow|arise|result|apply)\b|"
    r"what\s+(?:obligations|requirements|duties)\s+(?:does|do|will)\s+"
    r"(?:an?|the)?\s*classification|"
    r"obligations?\s+for\s+(?:an?\s+)?high.risk|"
    r"high.risk\s+(?:ai\s+)?(?:system\s+)?(?:obligations?|requirements?|duties?)|"
    r"what\s+(?:must|shall)\s+(?:an?\s+)?(?:provider|deployer|manufacturer)\s+"
    r"(?:do|comply))\b",
    re.I,
)

# Signals that the user wants broad, corpus-level coverage rather than a
# specific provision.  Triggers community-summary-first retrieval.
_COMMUNITY_SUMMARY_Q_RE = re.compile(
    r"\b(all|every|comprehensive|complete|overview|survey|full\s+list|"
    r"across\s+(?:all|both|the)|summarise|summarize|enumerate|list\s+all|"
    r"what\s+are\s+all|which\s+are\s+all)\b",
    re.I,
)

_MEDICAL_DEVICE_AI_Q_RE = re.compile(
    r"\b(medical\s+device|device|hospital|health\s+institution|"
    r"healthcare\s+institution|in-house|pathology|surgery|"
    r"tumou?r|patient|spectroscopy)\b",
    re.I,
)

_ANNEX_III_Q_RE = re.compile(
    r"\b(annex\s+iii|biometric|emotion\s+recognition|emotion\s+inference|"
    r"employment|worker|recruitment|creditworthiness|credit\s+score|"
    r"education|student|exam|law\s+enforcement|migration|asylum|border\s+control)\b",
    re.I,
)

# Patterns for detecting entities that DEVELOPED (not merely deployed) the AI system.
_INHOUSE_DEVELOPER_RE = re.compile(
    r"\b(develops?|developed|trains?|trained|builds?|built|creates?|created|"
    r"designs?|designed|implement(?:s|ed)?|construct(?:s|ed)?)\b",
    re.IGNORECASE,
)

_INHOUSE_CONTEXT_RE = re.compile(
    r"\b(in-house|in house|internally|own(?:\s+system)?|hospital|institution|"
    r"university|health\s+institution|healthcare\s+institution)\b",
    re.IGNORECASE,
)

_TEMPORAL_MARKER_RE = re.compile(
    r"\b(initially|after|later|two\s+years|subsequently|then|stage|"
    r"at\s+which\s+point|at\s+which\s+stage|first|second|third)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Signal detector functions
# ---------------------------------------------------------------------------


def _is_definition_question(question: str) -> tuple[bool, str | None]:
    """Detect if *question* asks for the definition/meaning of a concept.

    Returns ``(True, concept_text)`` when matched, ``(False, None)`` otherwise.
    The *concept_text* is the raw extracted subject, lowercased and stripped.
    """
    # Guard against broad obligation/listing prompts (e.g. "What are all
    # obligations of providers under the AI Act?").  The generic
    # "what are <concept>" definition pattern would otherwise misclassify these
    # as definition_lookup, bypassing role/community retrieval routes.
    if _OBLIGATION_Q_RE.search(question) or _COMMUNITY_SUMMARY_Q_RE.search(question):
        return False, None

    for pat in _DEFINITION_Q_PATTERNS:
        m = pat.search(question)
        if m:
            concept = m.group("concept").strip().rstrip("?").strip()
            # Drop trailing regulation references — the user already says
            # "according to MDR" elsewhere, we just want the concept.
            concept = re.sub(
                r"\s+(?:according\s+to|pursuant\s+to|under|in|of|per)\s+"
                r"(?:the\s+)?(?:MDR|IVDR|AI\s*Act|EU|Regulation|2017|2024).*$",
                "", concept, flags=re.I,
            ).strip()
            if concept:
                return True, concept.lower()
    return False, None


def _has_obligation_focus(question: str) -> bool:
    """Return whether the question is asking about duties or obligations."""
    return bool(_OBLIGATION_Q_RE.search(question))


def _has_cross_reg_focus(question: str) -> bool:
    """Return whether the question asks about interplay across frameworks."""
    return bool(_CROSS_REG_Q_RE.search(question))


def _has_qualification_focus(question: str) -> bool:
    """Return whether the question asks about legal status or qualification."""
    return bool(_QUALIFICATION_Q_RE.search(question))


def _has_role_transition_focus(question: str) -> bool:
    """Return whether the question asks when one legal role changes into another."""
    return bool(_ROLE_TRANSITION_Q_RE.search(question))


def _has_modification_focus(question: str) -> bool:
    """Return whether the question turns on modification or retraining triggers."""
    return bool(_MODIFICATION_Q_RE.search(question))


def _is_classification_chain_question(
    question: str,
    *,
    mentioned_regs: set[str],
    explicit_refs: list[str],
) -> bool:
    """Return True when the question asks about classification criteria and
    their downstream obligation chain.

    Triggers the ``classification_chain`` route, which uses
    :meth:`GraphRetriever.retrieve_by_chain` to follow
    ``TRIGGERS_OBLIGATION_CLUSTER`` edges from a classification gate
    (e.g. Article 6 AI Act) instead of relying on semantic similarity.

    Only fires when:
    - classification/chain language is present, AND
    - at least one regulation is in scope, AND
    - the question does NOT name a specific leaf provision (those go to
      provision_lookup so we do not over-retrieve).
    """
    if not mentioned_regs:
        return False
    if explicit_refs:
        # Specific provision named → provision_lookup is more precise
        return False
    return bool(_CHAIN_CLASSIFICATION_Q_RE.search(question))


def _is_community_summary_question(
    question: str,
    *,
    mentioned_regs: set[str],
    role_specs: list[tuple[str, str]],
) -> bool:
    """Return True when the question asks for broad/corpus-level coverage.

    Triggers the ``community_summary_search`` route, which searches community
    summaries first (300 vectors) before descending to member provisions.
    Only fires when:
    - corpus-coverage language is present, AND
    - at least one regulation or role is in scope (concrete enough to route), AND
    - no specific provision reference was extracted (those go to provision_lookup).
    """
    return (
        bool(_COMMUNITY_SUMMARY_Q_RE.search(question))
        and bool(mentioned_regs or role_specs)
    )


def _has_inhouse_developer_signal(question: str) -> bool:
    """Return True when the question describes an entity that DEVELOPED the AI system.

    Distinguishes developer-providers (Article 3(3) AI Act) from pure deployers
    (Article 3(4) AI Act) by detecting development verbs combined with institutional
    or in-house context.  When True, the entity's initial status is governed by
    Article 3(3), not Article 3(4), and Article 25 is inapplicable for the
    initial-status determination.
    """
    return bool(
        _INHOUSE_DEVELOPER_RE.search(question)
        and _INHOUSE_CONTEXT_RE.search(question)
    )


def _has_multistage_question(question: str) -> bool:
    """Return True when the question describes multiple events or stages over time.

    Requires at least two distinct temporal markers to distinguish a genuinely
    phased scenario (initial state → event 1 → event 2) from a simple question
    that incidentally uses time language.
    """
    markers = list(_TEMPORAL_MARKER_RE.finditer(question))
    return len(markers) >= 2


def _is_medical_device_ai_overlap(
    question: str,
    mentioned_regs: set[str],
) -> bool:
    """Return whether the question combines AI Act with MDR/IVDR product context."""
    has_ai_act = "EU AI Act" in mentioned_regs
    has_product_reg = bool({"MDR 2017/745", "IVDR 2017/746"} & mentioned_regs)
    return has_ai_act and has_product_reg and bool(_MEDICAL_DEVICE_AI_Q_RE.search(question))


def _uses_legal_qualification_route(
    question: str,
    *,
    mentioned_regs: set[str],
    role_specs: list[tuple[str, str]],
) -> bool:
    """Return whether the question needs qualification-first retrieval.

    Fires for medical-device + AI Act questions whenever the user asks about
    qualification, names an actor role, OR asks an obligation-framed question.
    The obligation-focus disjunct captures "what are our obligations" framings
    that semantically equal a classification request but lack explicit
    classification keywords — these questions still need the forced Article 6
    + Annex I backbone and the curated cross-regulation targets, otherwise
    they fall through to ``cross_regulation`` which has no route-specific
    discipline.
    """
    if not _is_medical_device_ai_overlap(question, mentioned_regs):
        return False
    return (
        _has_qualification_focus(question)
        or bool(role_specs)
        or _has_obligation_focus(question)
    )


def _question_mentions_any(question: str, terms: set[str]) -> bool:
    """Return whether any term appears as a whole word in the question."""
    q_lower = question.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", q_lower) for term in terms)


def _needs_actor_status_analysis(
    question: str,
    *,
    role_specs: list[tuple[str, str]],
) -> bool:
    """Return whether the question must resolve initial actor status first."""
    role_terms = {term for term, _celex in role_specs}
    return (
        bool(role_terms & {"provider", "deployer", "manufacturer", "user", "operator"})
        or _has_role_transition_focus(question)
        or _question_mentions_any(
            question,
            {
                "provider",
                "deployer",
                "manufacturer",
                "user",
                "operator",
                "put into service",
                "placing on the market",
                "placed on the market",
            },
        )
    )


def _needs_annex_iii_analysis(question: str) -> bool:
    """Return whether the question explicitly points to an Annex III use-case route."""
    return bool(_ANNEX_III_Q_RE.search(question))


# ---------------------------------------------------------------------------
# Qualification target builder
# ---------------------------------------------------------------------------


def _build_legal_qualification_targets(
    question: str,
    *,
    mentioned_regs: set[str],
    role_specs: list[tuple[str, str]],
) -> list[_ProvisionLookupTarget]:
    """Return curated provision targets for qualification-heavy questions."""
    targets: list[_ProvisionLookupTarget] = []
    seen: set[tuple[str, tuple[str, ...] | None]] = set()

    def add(ref: str, celexes: set[str] | None) -> None:
        key = (ref, tuple(sorted(celexes)) if celexes else None)
        if key in seen:
            return
        seen.add(key)
        targets.append(
            _ProvisionLookupTarget(
                ref=ref,
                celexes=frozenset(celexes) if celexes else None,
            )
        )

    ai_celex = {"32024R1689"} if "EU AI Act" in mentioned_regs else None
    mdr_celex = {"32017R0745"} if "MDR 2017/745" in mentioned_regs else None
    ivdr_celex = {"32017R0746"} if "IVDR 2017/746" in mentioned_regs else None
    gdpr_celex = (
        {"32016R0679"}
        if "General Data Protection Regulation (GDPR) 2016/679" in mentioned_regs
        else None
    )
    role_terms = {term for term, _celex in role_specs}
    manufacturer_status_terms = {"manufacturer", "user"}
    needs_actor_status = _needs_actor_status_analysis(question, role_specs=role_specs)
    is_developer = _has_inhouse_developer_signal(question)
    # Article 25 applies to third-party deployers who received the system from an
    # external provider. It does NOT govern the initial status of the original
    # developer. Exclude it from context when the developer signal is present so
    # the LLM cannot misuse it as the primary provider-conversion mechanism.
    needs_article_25 = (
        (_has_role_transition_focus(question) or _has_modification_focus(question))
        and not is_developer
    )
    needs_annex_iii = _needs_annex_iii_analysis(question)

    if ai_celex:
        if needs_actor_status:
            add("Article 3", ai_celex)

    if mdr_celex:
        if needs_actor_status or role_terms & manufacturer_status_terms or _question_mentions_any(
            question, manufacturer_status_terms | {"hospital", "in-house"}
        ):
            add("Article 2", mdr_celex)
        if _question_mentions_any(question, {"exemption", "hospital", "in-house"}):
            add("Article 5", mdr_celex)

    if ivdr_celex:
        if needs_actor_status or role_terms & manufacturer_status_terms or _question_mentions_any(
            question, manufacturer_status_terms | {"hospital", "in-house"}
        ):
            add("Article 2", ivdr_celex)
        if _question_mentions_any(question, {"exemption", "hospital", "in-house"}):
            add("Article 5", ivdr_celex)

    if ai_celex:
        add("Article 6", ai_celex)
        add("Annex I", ai_celex)
        if needs_annex_iii:
            add("Annex III", ai_celex)
        if {"MDR 2017/745", "IVDR 2017/746"} & mentioned_regs:
            add("Article 43", ai_celex)
        if needs_article_25:
            add("Article 25", ai_celex)

    # Force MDCG 2025-6 whenever both AI Act and MDR/IVDR are in scope.
    # This is the primary guidance on their interplay and has 81 CITES edges
    # in the graph. It must appear in context regardless of what HyDE retrieves.
    if ai_celex and (mdr_celex or ivdr_celex):
        add("MDCG 2025-6", {"MDCG_2025_6"})

    # GDPR backbone: when GDPR is in scope (especially alongside AI Act +
    # MDR/IVDR for medical AI questions), force-inject the decisive articles.
    # Without this, the LLM may misidentify the DPIA trigger (35(3)(a) vs (b))
    # or conflate the Article 9(2) derogation with the Article 6(1) lawful
    # basis. Article 4 grounds the definitions of personal data, health data,
    # genetic data, and biometric data — prerequisite for any Article 9
    # special-category analysis.
    if gdpr_celex:
        add("Article 4", gdpr_celex)
        add("Article 6", gdpr_celex)
        add("Article 9", gdpr_celex)
        add("Article 35", gdpr_celex)
        # DPIA-explicit questions also need Article 36 (prior consultation
        # with the supervisory authority where residual risk remains high).
        if _question_mentions_any(
            question,
            {
                "dpia",
                "data protection impact assessment",
                "prior consultation",
            },
        ):
            add("Article 36", gdpr_celex)
        # Controller accountability cluster: triggered when the question
        # explicitly references controller/processor status or breach handling.
        if _question_mentions_any(
            question,
            {
                "controller",
                "processor",
                "data controller",
                "data processor",
                "breach",
                "data breach",
            },
        ):
            add("Article 24", gdpr_celex)
            add("Article 28", gdpr_celex)

    return targets


# ---------------------------------------------------------------------------
# Route selector + role detection
# ---------------------------------------------------------------------------


def _detect_question_roles(
    question: str,
    *,
    target_celexes: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve role-bearing entities mentioned in the question.

    Returns ``[(term_normalized, celex), ...]`` suitable for the retriever's
    role-aware lookup path.
    """
    return _detect_role_specs(question, target_celexes=target_celexes)


def _select_question_route(
    question: str,
    *,
    explicit_refs: list[str],
    mentioned_regs: set[str],
    role_specs: list[tuple[str, str]],
    is_definition_question: bool,
) -> _QuestionRoute:
    """Classify the question into a bounded retrieval route."""
    if _uses_legal_qualification_route(
        question,
        mentioned_regs=mentioned_regs,
        role_specs=role_specs,
    ):
        return _QuestionRoute(
            id="legal_qualification",
            label="Legal qualification",
            rationale="medical-device AI qualification requires a forced Article 6 / Annex I backbone",
        )

    if len(mentioned_regs) >= 2 and (
        explicit_refs or role_specs or _has_cross_reg_focus(question)
    ):
        return _QuestionRoute(
            id="cross_regulation",
            label="Cross-regulation relational",
            rationale="multiple regulatory frameworks are in scope",
        )

    if explicit_refs:
        return _QuestionRoute(
            id="provision_lookup",
            label="Structural provision lookup",
            rationale="the question names a specific provision explicitly",
        )

    if is_definition_question:
        return _QuestionRoute(
            id="definition_lookup",
            label="Definition lookup",
            rationale="the question asks for the meaning of a concept",
        )

    if _is_classification_chain_question(
        question,
        mentioned_regs=mentioned_regs,
        explicit_refs=explicit_refs,
    ):
        return _QuestionRoute(
            id="classification_chain",
            label="Classification chain retrieval",
            rationale=(
                "classification/obligation-trigger language detected; "
                "using legal reasoning edges to traverse the obligation cluster"
            ),
        )

    if _is_community_summary_question(
        question,
        mentioned_regs=mentioned_regs,
        role_specs=role_specs,
    ):
        return _QuestionRoute(
            id="community_summary_search",
            label="Community-summary overview",
            rationale="corpus-coverage language detected; searching community summaries first",
        )

    if role_specs and _has_obligation_focus(question):
        return _QuestionRoute(
            id="role_obligations",
            label="Obligation by actor role",
            rationale="an actor role is present with obligation-oriented language",
        )

    if len(mentioned_regs) >= 2:
        return _QuestionRoute(
            id="cross_regulation",
            label="Cross-regulation relational",
            rationale="multiple regulatory frameworks are in scope",
        )

    return _QuestionRoute(
        id="general_compliance",
        label="General compliance",
        rationale="no narrower deterministic route dominates the query",
    )
