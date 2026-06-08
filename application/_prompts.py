"""LLM prompt construction — system prompt, route guidance, and user message.

Contains the static ``SYSTEM_PROMPT`` constant and the functions that assemble
the per-request user message and route-specific answer-discipline instructions.
"""
from __future__ import annotations

from typing import Any

from application._routing import (
    _QuestionRoute,
    _has_inhouse_developer_signal,
    _has_multistage_question,
    _has_role_transition_focus,
)

# ---------------------------------------------------------------------------
# System prompt (static, shared across all requests)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a European regulatory compliance expert specializing in \
MDR 2017/745, IVDR 2017/746, the EU AI Act (Regulation 2024/1689), \
and the General Data Protection Regulation (GDPR, Regulation 2016/679), \
as well as MDCG guidance documents that supplement the medical device regulations.

You answer questions based on the REGULATORY CONTEXT provided below.

TEXTUAL GROUNDING RULES:
- Your factual basis (paragraph numbers, quoted text, cross-references) MUST come exclusively from the provided REGULATORY CONTEXT.
- NEVER supply regulatory details (paragraph numbers, definitions, subparagraph ordinals, recital numbers, annex rule numbers) from your training memory — these are version-sensitive.
- If the context lacks a specific detail needed to complete your reasoning, state that the context is insufficient rather than guessing.
- Use ONLY EU regulatory terminology.

REGULATORY REASONING & LEGAL INFERENCE (mandatory for qualification and overlaps):
- You MUST use your expert understanding of European law to draw logical inferences, bridge multi-step definitions, and resolve cross-regulatory overlaps (e.g., AI Act + MDR).
- It is expected and required that you evaluate prerequisite legal definitions before applying obligations to a specific product or role.
- While your facts must be grounded in the context, your synthesis and deduction describing how those facts interact is your responsibility.
- State clearly what the text says vs how you legally interpret its application to the user's question.

BALANCED ANALYSIS (required):
- When the question asks whether something qualifies as a regulatory category \
(e.g. "does this constitute a significant change?"), you MUST present BOTH the \
arguments FOR and AGAINST based on the provisions in context.
- When guidance documents state that assessments must be done "case-by-case", \
explicitly flag this and do NOT give a categorical yes/no conclusion. Instead, \
state which outcome is more likely and under what conditions the opposite could \
apply.
- If the context contains lists of both qualifying and non-qualifying examples \
(e.g. Chart C significant vs non-significant software changes), cite BOTH lists \
and explain which examples are closest to the scenario in question.
- Clearly distinguish "likely" from "definitive" conclusions.

CROSS-REGULATION AWARENESS (required):
- When the context includes provisions from multiple regulations or frameworks \
(e.g. MDR and AI Act, or a regulation and MDCG guidance), you MUST address how \
they interact. Identify overlapping obligations, complementary requirements, or \
potential conflicts.
- When the question explicitly mentions concepts from multiple regulatory \
domains (e.g. "High-Risk AI" + "Class IIa medical device"), address each \
applicable framework even if one is less prominent in the question.
- Distinguish binding regulation from non-binding guidance: provisions tagged \
[LEGISLATION] are binding law; provisions tagged [GUIDANCE] are interpretive \
aids, not law. They carry persuasive but not legal authority.

LEGAL HIERARCHY RULES (critical):

- Binding EU Regulations (e.g., MDR 2017/745, IVDR 2017/746, EU AI Act 2024/1689, GDPR 2016/679)
  take precedence over all guidance documents.

- MDCG guidance documents are NON-BINDING. They:
  - interpret regulatory provisions,
  - provide examples and decision frameworks,
  - but do NOT create legal obligations.

- Where guidance appears to suggest a categorical outcome, but the Regulation
  requires a case-by-case assessment, you MUST:
  - defer to the Regulation, and
  - present the guidance as supportive, not determinative.

- In case of overlap between regulations (e.g., MDR and AI Act):
  - Apply the procedural rule explicitly stated in the Regulation text
    (e.g., Article 43(3) AI Act → MDR conformity assessment applies),
  - Then integrate additional requirements from the other regulation.

- NEVER treat guidance examples (e.g., "Chart C") as automatic legal conclusions.
  They are indicators, not binding classifications.
FORMAL DEFINITIONS vs. REGULATORY CONCEPTS (critical):
- A term is "defined" in EU law ONLY if it appears in the regulation's \
definitions article (Article 2 for MDR/IVDR, Article 3 for the AI Act) using \
the canonical form: \u2018term\u2019 means \u2026
- If the LEGAL DEFINITIONS section above contains a formal definition for a \
concept, cite it as a legal definition.
- If NO formal definition is provided for a concept, do NOT say it is \
"defined" in any provision. Instead, identify the provisions that establish \
criteria, requirements, or conditions for that concept and describe them as \
such \u2014 not as definitions.
- Example: \u201cequivalence\u201d is not defined in Article 2 of the MDR. Annex XIV Part A, \
Point 3 establishes CRITERIA for demonstrating equivalence \u2014 these are \
assessment requirements, not a definition.
- A NOTE in the context may explicitly flag that a concept lacks a formal \
definition. Respect that note.

CONCEPT \u2192 REGULATORY USE LINKAGE (required):
- When describing criteria, requirements, or conditions from a specific \
provision (e.g. Annex XIV equivalence criteria), you MUST also identify the \
substantive Article(s) that invoke or rely on those criteria (e.g. Article 61 \
requires clinical evaluation and references Annex XIV for equivalence \
assessment). This gives the reader the full regulatory picture.
- If the invoking Article appears in the Cross-references section of the \
context, cite it directly. If it does not appear but you know from the \
regulatory structure that a linkage exists, state it as an inference: \
"By regulatory structure, [Article X] invokes these criteria for [purpose]."
- Always connect Annex criteria back to their parent obligation in the \
enacting terms; never present Annex content in isolation.

Format your answer with:
1. A direct answer based on the provided context and sound regulatory reasoning
2. At least one VERBATIM quote (in quotation marks) per key provision cited — \
the exact words from the REGULATORY CONTEXT, not a paraphrase. Label each with \
the provision reference shown in the context header (e.g. [1] Article 2)
3. Cross-references that appear explicitly in the context
"""

# ---------------------------------------------------------------------------
# Route-specific answer guidance
# ---------------------------------------------------------------------------


def _build_route_answer_guidance(
    route: _QuestionRoute,
    *,
    question: str,
    sufficiency: dict[str, Any],
) -> str | None:
    """Return route-specific answer discipline for the final LLM prompt."""

    # ── Obligation-breadth questions (community summary + role obligations) ──
    # When a backbone master article was force-retrieved, instruct the LLM to
    # use it as a structural skeleton rather than free-forming the answer.
    if route.id in {"community_summary_search", "role_obligations"} and sufficiency.get("has_backbone"):
        backbone_label = sufficiency.get("backbone_label", "the obligations master list article")
        lines: list[str] = []
        lines.append("ANSWER DISCIPLINE — OBLIGATION COMPLETENESS:")
        lines.append(
            f"The REGULATORY CONTEXT begins with [{backbone_label}] marked as the "
            "OBLIGATIONS MASTER LIST. This is the authoritative statutory checklist for "
            "the actor in question — it was written specifically to enumerate all of that "
            "actor's obligations in one place."
        )
        lines.append(
            "Structure your answer by addressing each item on that master list in order. "
            "For each item, state the obligation and cite the specific article/paragraph "
            "it cross-references. If an item falls outside the question's scope, note it "
            "in one sentence and skip the detail."
        )
        lines.append(
            "After covering the master list, add any obligations from other parts of "
            "the regulation not captured in the master list (e.g. GPAI-specific tiers "
            "under Articles 51-56, prohibited-practice gates under Article 5). Label "
            "this section 'Additional obligations beyond the master list'."
        )
        lines.append(
            "Use precise article-level citations. Do NOT reorganise or omit items "
            "from the master list without explicit acknowledgment."
        )
        if not sufficiency.get("ok", True):
            lines.append(
                "Retrieval sufficiency is partial. Flag remaining uncertainty and "
                "avoid a definitive bottom-line conclusion."
            )
        return "\n".join(lines)

    if route.id == "classification_chain":
        lines: list[str] = []
        lines.append("ANSWER DISCIPLINE — AI ACT CLASSIFICATION SEQUENCE:")
        lines.append(
            "MANDATORY: Apply the classification analysis in this exact order. "
            "Do NOT skip or reorder these steps."
        )
        lines.append(
            "STEP 0 — PROHIBITED PRACTICES GATE (Article 5): "
            "Before any classification analysis, check whether the AI system falls "
            "within a prohibited practice under Article 5 (e.g. subliminal manipulation, "
            "social scoring, real-time remote biometric identification in public spaces). "
            "If it does, the system is PROHIBITED regardless of high-risk classification. "
            "The Article 6 and Article 51 analyses are legally moot for prohibited systems. "
            "Always surface this gate even if the question does not explicitly mention Article 5."
        )
        lines.append(
            "STEP 1 — IS THE SYSTEM AN AI SYSTEM? (Article 3(1)): "
            "Confirm the product meets the Article 3(1) definition of 'AI system' before "
            "proceeding to classification. If the context does not confirm this, flag it."
        )
        lines.append(
            "STEP 2 — GPAI MODEL CHECK (Article 3(63)): "
            "Determine whether the system is a 'general-purpose AI model' under Article 3(63). "
            "CRITICAL: Meeting the Article 3(63) definition ALONE — regardless of systemic risk — "
            "triggers the Article 53 BASELINE obligations (technical documentation, copyright "
            "policy, transparency to downstream providers). Do NOT conflate this with the "
            "systemic-risk assessment. A GPAI model that does NOT meet the Article 51 "
            "systemic-risk threshold still carries full Article 53 obligations."
        )
        lines.append(
            "STEP 3 — GPAI SYSTEMIC RISK (Article 51 — TWO DISTINCT ROUTES): "
            "If the system is a GPAI model (Step 2), assess systemic risk via BOTH routes:\n"
            "  Route A (provider-triggered): Article 51(1)(a) + Article 51(2). "
            "If training computation exceeds 10²⁵ FLOPs, the model is PRESUMED to have "
            "high impact capabilities. This is a rebuttable presumption: the provider may "
            "argue against it. The provider must notify the AI Office before placing the "
            "model on the market (Article 52(1)(c)).\n"
            "  Route B (Commission-triggered): Article 51(1)(b) + Annex XIII. "
            "The Commission may, ex officio or following a scientific panel alert, issue a "
            "decision that a model below the 10²⁵ FLOPs threshold has equivalent impact. "
            "Annex XIII criteria (parameters, data, modalities, capabilities) govern this "
            "Commission assessment — Annex XIII does NOT govern the Route A self-assessment.\n"
            "State which route applies (or both if relevant) and what the legal effect is "
            "(Article 55 additional obligations: adversarial testing, incident reporting, "
            "AI Office cooperation)."
        )
        lines.append(
            "STEP 4 — HIGH-RISK CLASSIFICATION (Article 6): "
            "Apply the two high-risk routes in order:\n"
            "  Route I (Article 6(1) + Annex I): AI system is a safety component or "
            "product under Union harmonisation legislation listed in Annex I AND requires "
            "third-party conformity assessment. Both conditions must be satisfied. "
            "No derogation applies to this route.\n"
            "  Route II (Article 6(2) + Annex III): AI system falls within one of the "
            "8 Annex III use-case categories. Subject to derogation under Article 6(3) "
            "for narrow procedural/preparatory tasks — EXCEPT for profiling of natural "
            "persons, which is always high-risk regardless of Article 6(3) conditions.\n"
            "Always check Route I before Route II. Do not invoke Annex III if the Annex I "
            "route is more specific to the facts."
        )
        lines.append(
            "STEP 5 — LEGAL EFFECTS: "
            "After classification, state what obligations the classification triggers: "
            "Article 53 baseline (all GPAI), Article 55 additional (systemic-risk GPAI), "
            "Articles 9-17 + 43 + 47-49 + 72 cluster (high-risk AI), "
            "Article 5 prohibition (prohibited practices). "
            "These effects are the legal consequence of the classification — state them "
            "as conclusions, not as the starting point for analysis."
        )
        lines.append(
            "CALIBRATION: Distinguish explicit law from inference. "
            "Where the classification depends on factual assessment (intended purpose, "
            "whether third-party conformity assessment is required, FLOPs count), "
            "flag what the classifier must verify and what cannot be concluded without it. "
            "Use 'likely', 'presumed', or 'subject to verification' as appropriate."
        )
        if not sufficiency.get("ok", True):
            lines.append(
                "Retrieval sufficiency is partial. Explicitly flag any provision "
                "missing from context and do not draw definitive conclusions from "
                "incomplete grounding."
            )
        return "\n".join(lines)

    if route.id != "legal_qualification":
        return None

    is_developer = _has_inhouse_developer_signal(question)
    is_multistage = _has_multistage_question(question)

    lines: list[str] = []

    # ── MANDATORY LEGAL RULES ────────────────────────────────────────────────
    # These rules fire before any answer-discipline instructions so the LLM
    # cannot override them by reasoning differently.
    lines.append("MANDATORY LEGAL RULES — READ BEFORE ANSWERING:")
    lines.append(
        "RULE 1 — AI Act provider status (Article 3(3)): "
        "'Puts into service' in Article 3(3) AI Act includes deploying a system "
        "for use within the developer's own organisation. External distribution is "
        "NOT required to qualify as a provider. "
        "An entity that develops an AI system AND first deploys it internally is a "
        "PROVIDER from inception, not a deployer. "
        "Deployer status (Article 3(4)) applies ONLY when the entity did NOT develop "
        "the system — it received or licensed it from a third-party provider."
    )
    lines.append(
        "RULE 2 — Article 25 scope: "
        "Article 25 AI Act applies ONLY to distributors, importers, deployers, or "
        "other third parties who received the system from an external provider and "
        "then substantially modified it or put their name on it. "
        "Article 25 CANNOT determine the initial provider status of the original "
        "developer. For self-developers, the governing provision is Article 3(3) alone."
    )

    if is_developer:
        lines.append(
            "RULE 3 — THIS QUESTION DESCRIBES A DEVELOPER: "
            "The entity described developed the AI system (in-house or internally). "
            "Under Article 3(3), development + internal deployment = provider status "
            "from inception. BEGIN the AI Act analysis from provider status. "
            "Do NOT frame the entity as initially a deployer. "
            "Article 25 is NOT applicable to determine this entity's initial status "
            "— do not use it as the primary conversion mechanism."
        )

    # ── DECOMPOSITION FOR MULTI-STAGE QUESTIONS ──────────────────────────────
    if is_multistage and _has_role_transition_focus(question):
        lines.append(
            "MANDATORY DECOMPOSITION: This question describes multiple events across "
            "time. Answer each stage separately and in order before synthesizing:"
        )
        lines.append(
            "  Stage 1: Initial legal status under EACH applicable regulation "
            "(before any events occur)."
        )
        lines.append(
            "  Stage 2: Effect of the first event (e.g., continuous learning, "
            "initial deployment) on each status."
        )
        lines.append(
            "  Stage 3: Effect of the second event (e.g., transfer to another "
            "entity, collaboration) on each status."
        )
        lines.append(
            "  Synthesis table and bottom-line conclusions come AFTER all stages, "
            "not before."
        )

    # ── ANSWER DISCIPLINE ────────────────────────────────────────────────────
    lines.append("ANSWER DISCIPLINE FOR THIS QUESTION:")
    lines.append(
        "You must structure your reasoning chronologically, resolving prerequisites "
        "before obligations. Use the following logical flow:"
    )
    lines.append(
        "- Format the answer using these sections in this order: "
        "(1) Explicitly stated in retrieved text, "
        "(2) Inference and likely outcome, "
        "(3) Remaining uncertainty and what would change the conclusion."
    )
    lines.append(
        "- Resolve initial actor status before any transition analysis: "
        "test whether the entity already meets the Article 3 provider definition "
        "and the MDR Article 2 manufacturer concept before applying exemption-loss "
        "consequences."
    )
    lines.append(
        "- For medical-device AI questions, treat Article 6(1) plus Annex I as the "
        "default high-risk route. Do not invoke Annex III unless the question or "
        "retrieved context clearly supports a specific Annex III pathway."
    )
    lines.append("1. Definitions First: Evaluate whether the product/actor meets the formal definition (e.g., 'active device', 'substantial modification', 'provider').")
    lines.append("2. Qualification Criteria: Assess specific criteria in Annexes or classification rules.")
    lines.append("3. Obligation Triggers: Explain what provisions apply IF the qualification is met (e.g., Conformity Assessment procedures).")
    lines.append("4. Conclusion & Uncertainty: Summarize the likely outcome. Use calibrated language such as 'likely', 'depends on', or 'requires case-by-case assessment' rather than categorical yes/no if there is any ambiguity.")
    lines.append("- Treat qualification and status-transition questions as case-specific unless the retrieved context makes the conclusion categorical.")
    lines.append("- Establish the legal route in order: Article 6(1) plus Annex I product-route analysis before Article 6(2) or Annex III reasoning; Article 3 provider-definition analysis before Article 25 transition analysis when both are in play.")
    lines.append("- Do not jump straight to an obligation without verifying the definitional prerequisite first.")
    lines.append("- State clearly what the retrieved text explicitly states versus what you logically infer from it.")
    lines.append("- If definitions or criteria are missing, flag that the prerequisite cannot be definitively established.")

    if not sufficiency.get("ok", True):
        lines.append(
            "- Retrieval sufficiency is partial. You must explicitly flag the remaining uncertainty and avoid a definitive bottom-line conclusion."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------


def _build_user_message(
    *,
    question: str,
    context: str,
    route: _QuestionRoute,
    sufficiency: dict[str, Any],
    mentioned_regs: set[str] | None = None,
) -> str:
    """Build the final user message sent to the answer-generation model."""
    route_guidance = _build_route_answer_guidance(route, question=question, sufficiency=sufficiency)
    parts: list[str] = []
    if route_guidance:
        parts.append(route_guidance)
    # Single-regulation scope constraint — prevents cross-regulation citation contamination.
    # When exactly one regulation is in scope, the LLM receives an explicit prohibition
    # on importing citations from other regulations.  For multi-regulation questions the
    # CROSS-REGULATION AWARENESS instruction in the system prompt remains fully active.
    if mentioned_regs and len(mentioned_regs) == 1:
        reg_name = next(iter(mentioned_regs))
        parts.append(
            f"REGULATORY SCOPE CONSTRAINT:\n"
            f"This question is scoped EXCLUSIVELY to {reg_name}. "
            f"Every provision citation in your answer MUST come from {reg_name}. "
            f"If a concept (e.g. 'reasonably foreseeable misuse', 'intended purpose', "
            f"'substantial modification') also appears in another regulation, cite ONLY "
            f"the {reg_name} version. "
            f"Cross-regulation citations are PROHIBITED unless a provision from another "
            f"regulation appears explicitly in the REGULATORY CONTEXT below, tagged "
            f"[LEGISLATION] with its own regulation name."
        )
    parts.append(f"REGULATORY CONTEXT:\n{context}")
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)
