"""Detection stage — turn a raw question into a typed :class:`Scenario`.

"Understand the question" is the first stage of the agent spine
(understand → clarify? → plan → retrieve → generate → verify). It is fully
deterministic — no LLM: regex/keyword detectors plus the retriever's in-memory
defined-term index. This module is the **single source of truth** for that
stage. Both consumers drive it:

* ``application.agent.ask_stream`` (the live read path), and
* ``scripts/eval_retrieval.py`` (the deterministic retrieval net),

which previously each re-implemented the detection block line-for-line. Folding
them onto one ``detect_scenario`` removes that duplication and means the net
**gates** any change here.

The function returns a :class:`Detection`: the frozen :class:`Scenario` contract
plus the side outputs later (not-yet-migrated) stages still consume as loose
values — the ``route`` object (for its label/rationale), the fetched
``definitions``, the possibly-bumped retrieval budget ``k``, and the raw
detection locals with their exact original types (``mentioned_regs`` as a set,
``target_celexes`` as ``set | None``). As the plan/retrieve/generate stages move
onto the ``Scenario`` / ``Evidence`` contracts, these extras shrink.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from application._config import (
    _REG_NAME_TO_CELEX,
    _detect_mentioned_regulations,
    _extract_provision_refs,
    _extract_implicit_provision_refs,
    _extract_context_anchor_refs,
)
from application._definitions import _detect_defined_terms
from application._routing import (
    _detect_question_roles,
    _is_definition_question,
    _select_question_route,
)
from application.contracts import Scenario

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Everything the detection stage produces for one question.

    ``scenario`` is the typed contract the spine organises around; the remaining
    fields are the loose locals the downstream stages still read directly, kept
    at their original types so the relocation is byte-for-byte behaviour-neutral
    for both consumers (notably ``target_celexes`` stays ``None`` — not an empty
    set — when no regulation is in scope).
    """

    scenario: Scenario
    definitions: list[dict]
    route: Any
    k: int
    mentioned_regs: set[str]
    keyword_mentioned_regs: set[str]
    target_celexes: set[str] | None
    role_specs: list
    explicit_refs: list[str]
    context_anchor_refs: list[str]
    is_def_q: bool
    concept_text: str


def detect_scenario(question: str, retriever, k: int) -> Detection:
    """Run the full deterministic detection stage for *question*.

    Mirrors exactly the sequence the agent runs: fetch defined terms, detect
    mentioned regulations (widened by definition-implied regs), derive the CELEX
    filter and per-regulation budget bump, detect actor roles, extract explicit +
    implicit provision refs, classify the definition flag, and select the route.
    No LLM calls; no streaming — side-effecting concerns (progress events) stay
    with the caller. Detection-level diagnostics are logged here, where the logic
    lives.
    """
    # --- defined terms (also surfaces implicit regulation links) ---
    definitions = _detect_defined_terms(question, retriever)
    if definitions:
        logger.info(
            "Injecting %d definition(s): %s",
            len(definitions),
            ", ".join(d.get("term", "?") for d in definitions),
        )

    # --- regulation detection + CELEX filter ---
    keyword_mentioned_regs = _detect_mentioned_regulations(question)
    mentioned_regs = set(keyword_mentioned_regs)
    for d in definitions:
        reg = d.get("regulation", "")
        if reg and reg not in mentioned_regs and reg in _REG_NAME_TO_CELEX:
            mentioned_regs.add(reg)
            logger.info(
                "Implicit regulation detected via defined term '%s': %s",
                d.get("term", "?"), reg,
            )

    target_celexes: set[str] | None = None
    if mentioned_regs:
        target_celexes = {
            _REG_NAME_TO_CELEX[r]
            for r in mentioned_regs
            if r in _REG_NAME_TO_CELEX
        }
        if len(mentioned_regs) > 1:
            has_guidance = any(
                _REG_NAME_TO_CELEX.get(r, "").startswith("MDCG_")
                for r in mentioned_regs
            )
            per_reg = 4 if has_guidance else 3
            k = max(k, len(mentioned_regs) * per_reg)

    # --- roles, refs, definition flag, route ---
    role_specs = _detect_question_roles(question, target_celexes=target_celexes)

    explicit_refs = _extract_provision_refs(question)
    for _ref in _extract_implicit_provision_refs(question, target_celexes=target_celexes):
        if _ref not in explicit_refs:
            explicit_refs.append(_ref)

    is_def_q, concept_text = _is_definition_question(question)
    route = _select_question_route(
        question,
        explicit_refs=explicit_refs,
        mentioned_regs=mentioned_regs,
        keyword_mentioned_regs=keyword_mentioned_regs,
        role_specs=role_specs,
        is_definition_question=is_def_q,
    )
    logger.info("Question routed to %s: %s", route.id, route.rationale)

    # Context anchors: decisive provisions force-retrieved for a topic that the
    # dense/lexical channels miss (e.g. MDR Annex XVI for a wellbeing-app
    # qualification, Annex VIII Rule 11 for CDSS). Kept *separate* from
    # explicit_refs — they must enrich retrieval for ANY route without (a)
    # reclassifying a broad question as provision_lookup or (b) being gated out
    # by the orchestrator's route-specific direct-lookup. The retrieval stage
    # merges them into the bag regardless of route.
    context_anchor_refs = _extract_context_anchor_refs(question, target_celexes=target_celexes)

    scenario = Scenario(
        question=question,
        mentioned_regs=frozenset(mentioned_regs),
        target_celexes=frozenset(target_celexes or ()),
        role_specs=tuple(role_specs),
        explicit_refs=tuple(explicit_refs),
        route_id=route.id,
        is_definition_question=is_def_q,
    )

    return Detection(
        scenario=scenario,
        definitions=definitions,
        route=route,
        k=k,
        mentioned_regs=mentioned_regs,
        keyword_mentioned_regs=keyword_mentioned_regs,
        target_celexes=target_celexes,
        role_specs=role_specs,
        explicit_refs=explicit_refs,
        context_anchor_refs=context_anchor_refs,
        is_def_q=is_def_q,
        concept_text=concept_text,
    )