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
MDR 2017/745, IVDR 2017/746, and the EU AI Act (Regulation 2024/1689), \
as well as MDCG guidance documents that supplement these regulations.

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

- Binding EU Regulations (e.g., MDR 2017/745, IVDR 2017/746, EU AI Act 2024/1689)
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
) -> str:
    """Build the final user message sent to the answer-generation model."""
    route_guidance = _build_route_answer_guidance(route, question=question, sufficiency=sufficiency)
    parts: list[str] = []
    if route_guidance:
        parts.append(route_guidance)
    parts.append(f"REGULATORY CONTEXT:\n{context}")
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)
