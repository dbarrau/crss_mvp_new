"""Regulatory compliance Q&A agent backed by Neo4j graph retrieval + Mistral.

Provides the :func:`ask` function that:
1. Detects regulatory terms in the question and fetches legal definitions
2. Retrieves relevant provisions from the knowledge graph
3. Assembles structured context with definitions and cross-references
4. Sends to Mistral (EU-hosted) for a grounded answer

Sub-module layout
-----------------
_config.py          Shared constants, regex patterns, regulation mappings
_routing.py         Deterministic question-route classification
_definitions.py     Defined-term detection and context expansion
_retrieval.py       Graph/vector retrieval orchestration and audit trace
_context.py         Context assembly and LLM-prompt formatting
_prompts.py         System prompt and user-message builder
_postprocessing.py  Answer safety formatting, banners, backbone validation
"""
from __future__ import annotations

import logging
import os
from typing import Any

from domain.ontology.defined_terms import DEFINITIONS_ARTICLES as _DEF_ARTICLES

# ── Sub-module re-exports (keep all private symbols importable from here) ──
from application._config import (                        # noqa: F401
    _REG_NAME_TO_CELEX,
    _REG_PATTERNS,
    _MDCG_EXTRA_PATTERNS,
    _PROVISION_REF_RE,
    _INLINE_REF_RE,
    _BODY_LIMIT,
    _MAX_DEFINITIONS,
    _MAX_RELATED_DEFINITIONS,
    _RELATED_DEFINITION_SCAN_LIMIT,
    _detect_mentioned_regulations,
    _extract_provision_refs,
)
from application._routing import (                       # noqa: F401
    _QuestionRoute,
    _ProvisionLookupTarget,
    _is_definition_question,
    _has_obligation_focus,
    _has_cross_reg_focus,
    _has_qualification_focus,
    _has_role_transition_focus,
    _has_modification_focus,
    _has_inhouse_developer_signal,
    _has_multistage_question,
    _is_medical_device_ai_overlap,
    _uses_legal_qualification_route,
    _question_mentions_any,
    _needs_actor_status_analysis,
    _needs_annex_iii_analysis,
    _build_legal_qualification_targets,
    _detect_question_roles,
    _select_question_route,
)
from application._definitions import (                   # noqa: F401
    _detect_defined_terms,
    _expand_definitions_from_provisions,
)
from application._retrieval import (                     # noqa: F401
    _hyde_query,
    _merge_unique_provisions,
    _retrieve_lookup_targets,
    _has_lookup_target_coverage,
    _retrieve_direct_provisions,
    _retrieve_route_provisions,
    _collect_context_celexes,
    _collect_context_refs,
    _has_role_context,
    _evaluate_route_sufficiency,
    _run_corrective_retrieval_pass,
    _build_audit_trace,
)
from application._context import (                       # noqa: F401
    _MAX_POINTER_REFS,
    _GUIDANCE_CELEX_PREFIXES,
    _normalize_ref,
    _extract_inline_refs,
    _format_definitions,
    _format_context,
)
from application._prompts import (                       # noqa: F401
    SYSTEM_PROMPT,
    _build_route_answer_guidance,
    _build_user_message,
)
from application._postprocessing import (                # noqa: F401
    _build_uncertainty_banner,
    _soften_categorical_language,
    _validate_legal_backbone,
    _postprocess_answer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# History / standalone-question rewriting
# ---------------------------------------------------------------------------


def _rewrite_standalone_question(
    question: str,
    history: list[dict[str, str]],
    client,
) -> str:
    """Rewrite a follow-up question as a complete standalone question.

    Called when *history* has at least 2 turns.  Uses a fast, low-temperature
    Mistral call (max 120 tokens) so retrieval always works on a self-contained
    query.  Falls back to the original *question* on any error.
    """
    _MAX_TURNS = 6
    _MAX_CHARS = 3000
    recent = history[-_MAX_TURNS:]
    total_chars = 0
    trimmed: list[dict[str, str]] = []
    for turn in recent:
        content = turn.get("content", "")
        if total_chars + len(content) > _MAX_CHARS:
            break
        total_chars += len(content)
        trimmed.append(turn)
    if not trimmed:
        trimmed = recent[-1:]

    rewrite_messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a query rewriter for a regulatory compliance assistant. "
                "Given a conversation history and a follow-up question, rewrite the "
                "follow-up as a complete, self-contained question that can be "
                "understood without the conversation history. Preserve all regulatory "
                "references (regulation names, article numbers, defined terms). "
                "Output ONLY the rewritten question — no explanation, no preamble."
            ),
        }
    ]
    for turn in trimmed:
        role = turn.get("role", "user")
        if role == "agent":
            role = "assistant"
        rewrite_messages.append({"role": role, "content": turn.get("content", "")})
    rewrite_messages.append({"role": "user", "content": question})

    try:
        response = client.chat.complete(
            model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
            messages=rewrite_messages,
            temperature=0,
            max_tokens=120,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten:
            logger.info("Standalone rewrite: %r → %r", question[:80], rewritten[:80])
            return rewritten
    except Exception as exc:
        logger.warning("Standalone question rewrite failed: %s", exc)
    return question


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ask_stream(question: str, retriever, k: int = 20, history: list[dict[str, str]] | None = None):
    """Streaming version of :func:`ask`.

    Yields JSON-serializable dicts at each pipeline stage so the caller can
    surface progress to the user in real time, followed by token-by-token LLM
    output.

    Event shapes
    ------------
    ``{"type": "step",       "id": <str>, "label": <str>}``
        A named pipeline step completed.  Callers may show or hide these.
    ``{"type": "generating"}``
        Context assembly is done; LLM generation is starting.
    ``{"type": "token",      "content": <str>}``
        One streamed chunk of LLM output text.
    ``{"type": "done",       "answer": <str>}``
        Generation finished.  ``answer`` is the full concatenated response.
    ``{"type": "error",      "message": <str>}``
        An exception was raised.  No further events will follow.
    ``{"type": "audit",      "trace": <dict>}``
        Structured retrieval/audit metadata emitted before generation.
    """
    import time
    from mistralai.client import Mistral

    try:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        # --- 0. Standalone question rewriting (when conversation history present) ---
        retrieval_question = question
        if history and len(history) >= 2:
            retrieval_question = _rewrite_standalone_question(question, history, client)
            if retrieval_question != question:
                yield {
                    "type": "step",
                    "id": "rewrite",
                    "label": (
                        f"Query rewritten for retrieval: \""
                        f"{retrieval_question[:120]}"
                        f"{'…' if len(retrieval_question) > 120 else ''}\""
                    ),
                }

        # --- 1. Fetch legal definitions ---
        definitions = _detect_defined_terms(retrieval_question, retriever)
        if definitions:
            logger.info(
                "Injecting %d definition(s): %s",
                len(definitions),
                ", ".join(d.get("term", "?") for d in definitions),
            )
        term_names = [d.get("term", "?") for d in definitions]
        yield {
            "type": "step",
            "id": "definitions",
            "label": (
                f"Found {len(definitions)} defined term(s): {', '.join(term_names)}"
                if definitions
                else "No defined terms detected in question"
            ),
        }

        # --- 2. Regulation detection + CELEX filter ---
        mentioned_regs = _detect_mentioned_regulations(retrieval_question)
        for d in definitions:
            reg = d.get("regulation", "")
            if reg and reg not in mentioned_regs and reg in _REG_NAME_TO_CELEX:
                mentioned_regs.add(reg)
                logger.info(
                    "Implicit regulation detected via defined term '%s': %s",
                    d.get("term", "?"), reg,
                )

        target_celexes: set[str] | None = None
        if len(mentioned_regs) >= 1:
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

        role_specs = _detect_question_roles(retrieval_question, target_celexes=target_celexes)

        yield {
            "type": "step",
            "id": "regulations",
            "label": (
                f"Targeting {len(mentioned_regs)} regulation(s): "
                + ", ".join(sorted(mentioned_regs))
                if mentioned_regs
                else "No specific regulation detected — searching all"
            ),
        }
        if role_specs:
            yield {
                "type": "step",
                "id": "roles",
                "label": "Detected actor role(s): " + ", ".join(
                    f"{term}@{celex}" for term, celex in role_specs
                ),
            }

        explicit_refs = _extract_provision_refs(retrieval_question)
        is_def_q, concept_text = _is_definition_question(retrieval_question)
        route = _select_question_route(
            retrieval_question,
            explicit_refs=explicit_refs,
            mentioned_regs=mentioned_regs,
            role_specs=role_specs,
            is_definition_question=is_def_q,
        )
        logger.info("Question routed to %s: %s", route.id, route.rationale)
        yield {
            "type": "step",
            "id": "route",
            "label": f"Route: {route.label} — {route.rationale}",
        }

        # --- 3. Route-specific retrieval ---
        retrieval_result = _retrieve_route_provisions(
            retrieval_question,
            retriever,
            client=client,
            k=k,
            route=route,
            target_celexes=target_celexes,
            explicit_refs=explicit_refs,
            role_specs=role_specs,
            has_definitions=bool(definitions),
        )
        direct_provisions = retrieval_result["direct_provisions"]
        legal_qualification_targets = retrieval_result["legal_qualification_targets"]
        role_provisions = retrieval_result["role_provisions"]
        hyde_text = retrieval_result["hyde_text"]
        provisions = retrieval_result["provisions"]

        if explicit_refs and direct_provisions:
            logger.info("Direct lookup: %s → %d provision(s)", explicit_refs, len(direct_provisions))
            yield {
                "type": "step",
                "id": "direct",
                "label": (
                    f"Direct lookup for {explicit_refs}: "
                    f"{len(direct_provisions)} provision(s) found"
                ),
            }

        # --- 4. Route-specific expansions ---
        if hyde_text:
            logger.debug("HyDE text: %s", hyde_text[:120])
            yield {
                "type": "step",
                "id": "hyde",
                "label": f"HyDE query: \"{hyde_text[:100]}{'…' if len(hyde_text) > 100 else ''}\"",
            }

        if role_provisions:
            logger.info(
                "Role retrieval: %d provision(s) for %s.",
                len(role_provisions),
                ", ".join(f"{term}@{celex}" for term, celex in role_specs),
            )
        yield {
            "type": "step",
            "id": "retrieval",
            "label": (
                f"Hybrid retrieval: {len(provisions)} provision(s) retrieved"
                + (f" ({len(role_provisions)} from role-aware path)" if role_provisions else "")
            ),
        }

        definitions = _expand_definitions_from_provisions(
            provisions, retriever, definitions, target_celexes=target_celexes,
        )

        sufficiency = _evaluate_route_sufficiency(
            route=route,
            question=question,
            explicit_refs=explicit_refs,
            target_celexes=target_celexes,
            role_specs=role_specs,
            provisions=provisions,
            definitions=definitions,
            direct_provisions=direct_provisions,
            role_provisions=role_provisions,
            legal_qualification_targets=legal_qualification_targets,
        )
        corrective_actions: list[str] = []
        if not sufficiency["ok"]:
            recovery = _run_corrective_retrieval_pass(
                retrieval_question,
                retriever,
                client=client,
                k=k,
                route=route,
                target_celexes=target_celexes,
                explicit_refs=explicit_refs,
                role_specs=role_specs,
                provisions=provisions,
                direct_provisions=direct_provisions,
                role_provisions=role_provisions,
                definitions=definitions,
                sufficiency=sufficiency,
                hyde_text=hyde_text,
                legal_qualification_targets=legal_qualification_targets,
            )
            corrective_actions = recovery["actions"]
            hyde_text = recovery["hyde_text"]
            sufficiency = recovery["sufficiency"]

        sufficiency_label = (
            "Retrieval sufficiency passed"
            if sufficiency["ok"] and not corrective_actions
            else "Retrieval sufficiency recovered via corrective pass"
            if sufficiency["ok"]
            else "Retrieval sufficiency remains partial"
        )
        sufficiency_detail = "; ".join(
            check["detail"] for check in sufficiency["checks"] if not check["passed"]
        )
        if corrective_actions:
            detail = "; ".join(corrective_actions)
            if sufficiency_detail:
                detail += f"; remaining gaps: {sufficiency_detail}"
        else:
            detail = sufficiency_detail or "all route checks passed"
        yield {
            "type": "step",
            "id": "sufficiency",
            "label": f"{sufficiency_label}: {detail}",
        }

        if not provisions and not definitions:
            yield {"type": "error", "message": (
                "No relevant provisions were found in the knowledge graph. "
                "Please check that embeddings have been generated "
                "(run scripts/embed_provisions.py)."
            )}
            return

        # --- 5. Pointer expansion ---
        inline_refs = _extract_inline_refs(provisions)
        if inline_refs:
            pointer_provisions = retriever.retrieve_by_refs(
                inline_refs, celex_filter=target_celexes,
            )
            if pointer_provisions:
                seen_ids = {p["article_id"] for p in provisions}
                added = 0
                for p in pointer_provisions:
                    if p["article_id"] not in seen_ids:
                        p["_pointer_expansion"] = True
                        provisions.append(p)
                        seen_ids.add(p["article_id"])
                        added += 1
                        if added >= _MAX_POINTER_REFS:
                            break
                if added:
                    logger.info(
                        "Pointer expansion: %s → added %d provision(s) (total %d).",
                        inline_refs[:5], added, len(provisions),
                    )
                    yield {
                        "type": "step",
                        "id": "pointers",
                        "label": (
                            f"Cross-reference expansion: {added} additional "
                            f"provision(s) pulled in via inline references"
                        ),
                    }

        # --- 6. Assemble context ---
        audit_trace = _build_audit_trace(
            question=question,
            route=route,
            mentioned_regs=mentioned_regs,
            target_celexes=target_celexes,
            explicit_refs=explicit_refs,
            role_specs=role_specs,
            definitions=definitions,
            provisions=provisions,
            direct_provisions=direct_provisions,
            role_provisions=role_provisions,
            legal_qualification_targets=legal_qualification_targets,
            hyde_text=hyde_text,
            inline_refs=inline_refs,
            corrective_actions=corrective_actions,
            sufficiency=sufficiency,
        )

        context_parts: list[str] = []
        if definitions:
            context_parts.append(
                "LEGAL DEFINITIONS (from the definitions article):\n"
                + _format_definitions(definitions)
            )

        if is_def_q and concept_text:
            concept_covered = any(
                concept_text in (d.get("term", "").lower())
                or (d.get("term", "").lower() in concept_text)
                for d in definitions
            )
            if not concept_covered:
                def_art_refs: list[str] = []
                for reg_name in (mentioned_regs or _REG_NAME_TO_CELEX.keys()):
                    celex = _REG_NAME_TO_CELEX.get(reg_name, "")
                    art_info = _DEF_ARTICLES.get(celex)
                    if art_info:
                        def_art_refs.append(f"{art_info['display_ref']} ({reg_name})")
                if def_art_refs:
                    note = (
                        f"NOTE: \u2018{concept_text}\u2019 is NOT a formally defined term "
                        f"in {', '.join(def_art_refs)}. "
                        f"The provisions below may describe criteria, requirements, "
                        f"or conditions related to this concept \u2014 they are NOT "
                        f"definitions."
                    )
                    context_parts.append(note)
                    logger.info("Negative-definition signal injected for '%s'.", concept_text)

        if provisions:
            context_parts.append(_format_context(provisions))
        context = "\n\n---\n\n".join(context_parts)

        logger.debug(
            "Context assembled: %d provisions + %d definitions, %d chars",
            len(provisions), len(definitions), len(context),
        )

        yield {
            "type": "step",
            "id": "context",
            "label": (
                f"Context ready: {len(provisions)} provision(s), "
                f"{len(definitions)} definition(s) — sending to LLM"
            ),
        }
        yield {"type": "audit", "trace": audit_trace}

        # --- 7. Stream LLM response ---
        yield {"type": "generating"}

        # Build bounded history turns for the LLM call
        _HISTORY_MAX_TURNS = 6
        _HISTORY_MAX_CHARS = 3000
        _history_messages: list[dict[str, str]] = []
        if history:
            _total_chars = 0
            for _turn in history[-_HISTORY_MAX_TURNS:]:
                _content = _turn.get("content", "")
                if _total_chars + len(_content) > _HISTORY_MAX_CHARS:
                    break
                _total_chars += len(_content)
                _role = _turn.get("role", "user")
                if _role == "agent":
                    _role = "assistant"
                _history_messages.append({"role": _role, "content": _content})

        _messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *_history_messages,
            {
                "role": "user",
                "content": _build_user_message(
                    question=question,
                    context=context,
                    route=route,
                    sufficiency=sufficiency,
                ),
            },
        ]

        _MAX_RETRIES = 4
        full_answer = ""
        for attempt in range(_MAX_RETRIES):
            try:
                full_answer = ""
                with client.chat.stream(
                    model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
                    messages=_messages,
                    temperature=0.1,
                ) as stream:
                    for chunk in stream:
                        delta = chunk.data.choices[0].delta.content
                        if delta:
                            full_answer += delta
                            yield {"type": "token", "content": delta}
                break  # success — exit retry loop
            except Exception as exc:
                if "429" in str(exc) and attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning("Rate-limited (429). Retrying in %ds…", wait)
                    import time
                    time.sleep(wait)
                else:
                    raise

        final_answer = _postprocess_answer(
            full_answer,
            route,
            question=question,
            sufficiency=sufficiency,
        )
        yield {"type": "done", "answer": final_answer, "audit_trace": audit_trace}

    except Exception as exc:
        logger.exception("Error in ask_stream()")
        yield {"type": "error", "message": str(exc)}


def ask_with_trace(question: str, retriever, k: int = 20, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Return both the grounded answer and structured audit trace."""
    audit_trace: dict[str, Any] | None = None
    for event in ask_stream(question, retriever, k=k, history=history):
        if event.get("type") == "audit":
            audit_trace = event.get("trace")
        if event.get("type") == "done":
            return {
                "answer": event["answer"],
                "audit_trace": event.get("audit_trace") or audit_trace,
            }
        if event.get("type") == "error":
            raise RuntimeError(event["message"])
    return {"answer": "No answer generated.", "audit_trace": audit_trace}


def ask(question: str, retriever, k: int = 20, history: list[dict[str, str]] | None = None) -> str:
    """Retrieve context from the graph and generate an answer via Mistral.

    Delegates to :func:`ask_stream` and accumulates the result.  Kept for
    backward compatibility with ``scripts/chat.py`` and any other CLI callers.

    Parameters
    ----------
    question:
        The user's natural-language question.
    retriever:
        A :class:`retrieval.graph_retriever.GraphRetriever` instance.
    k:
        Number of top provisions to retrieve.

    Returns
    -------
    str
        The LLM-generated answer grounded in regulatory text.
    """
    for event in ask_stream(question, retriever, k=k, history=history):
        if event.get("type") == "done":
            return event["answer"]
        if event.get("type") == "error":
            raise RuntimeError(event["message"])
    return "No answer generated."
