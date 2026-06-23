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
from datetime import date
from typing import Any

from domain.ontology.applicability import applicability_note as _applicability_note
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
    _extract_implicit_provision_refs,
)
from application._faithfulness import (                  # noqa: F401
    build_warning_block as _build_faithfulness_warning,
    check_faithfulness as _check_faithfulness,
    faithfulness_mode as _faithfulness_mode,
    out_of_scope_citation_refs as _out_of_scope_citation_refs,
    remove_unverified_quotes as _remove_unverified_quotes,
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
    _collect_context_communities,
    _has_community_diversity,
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
    _trim_provisions_to_budget,
    _community_summary_header,
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
from application._confidence import (                    # noqa: F401
    compute_confidence,
)
from application._audit import (                          # noqa: F401
    _should_audit,
    _audit_answer,
    _audit_model,
    _needs_revision,
    _gap_retrieve,
    _format_findings,
    _build_revision_messages,
    _strip_meta_leak,
    _max_iters,
    _max_gap_refs,
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
        # Cheap reformulation task — the output is a self-contained question fed
        # back into retrieval, never shown to the user — so it runs on a fast
        # model by default (CRSS_REWRITE_MODEL, default mistral-small-latest).
        response = client.chat.complete(
            model=os.environ.get("CRSS_REWRITE_MODEL", "mistral-small-latest"),
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
        keyword_mentioned_regs = _detect_mentioned_regulations(retrieval_question)
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
        for _ref in _extract_implicit_provision_refs(retrieval_question, target_celexes=target_celexes):
            if _ref not in explicit_refs:
                explicit_refs.append(_ref)
        is_def_q, concept_text = _is_definition_question(retrieval_question)
        route = _select_question_route(
            retrieval_question,
            explicit_refs=explicit_refs,
            mentioned_regs=mentioned_regs,
            keyword_mentioned_regs=keyword_mentioned_regs,
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
        )
        direct_provisions = retrieval_result["direct_provisions"]
        legal_qualification_targets = retrieval_result["legal_qualification_targets"]
        role_provisions = retrieval_result["role_provisions"]
        hyde_text = retrieval_result["hyde_text"]
        provisions = retrieval_result["provisions"]
        has_backbone: bool = retrieval_result.get("has_backbone", False)
        backbone_provisions: list[dict] = retrieval_result.get("backbone_provisions") or []
        backbone_label: str | None = retrieval_result.get("backbone_label")

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
        # Inject backbone metadata so prompts and self-check can consume it.
        sufficiency["has_backbone"] = has_backbone
        if backbone_label:
            sufficiency["backbone_label"] = backbone_label
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

        # Temporal applicability: flag obligations that may not yet be in force
        # as of today (e.g. AI Act general/high-risk application starts
        # 2026-08-02). Annotation only — never filters the retrieved provisions.
        _appl_celexes = _collect_context_celexes(provisions, definitions) or set(
            target_celexes or set()
        )
        _appl_note = _applicability_note(_appl_celexes, date.today())
        if _appl_note:
            context_parts.append(_appl_note)

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
            if route.id == "community_summary_search":
                header = _community_summary_header(provisions)
                if header:
                    context_parts.append(header)

            # Backbone block: force-retrieved master list article rendered
            # before the main provisions so it anchors the LLM's structure.
            if has_backbone and backbone_provisions:
                backbone_header = (
                    f"[OBLIGATIONS MASTER LIST — {backbone_label} — "
                    "Authoritative statutory checklist for this actor. "
                    "Address each item in order.]\n"
                )
                context_parts.append(backbone_header + _format_context(backbone_provisions))

            # Bound the prompt size: broad routes can retrieve 40+ provisions
            # (~220 KB / ~56 K tokens), which inflates the large model's
            # time-to-first-token. Keep the highest-priority provisions and drop
            # the low-value tail. The backbone block above is rendered in full.
            _budgeted = _trim_provisions_to_budget(provisions)
            if len(_budgeted) < len(provisions):
                logger.info(
                    "Context budget: kept %d of %d provisions (tail trimmed)",
                    len(_budgeted), len(provisions),
                )
            context_parts.append(_format_context(_budgeted))
        context = "\n\n---\n\n".join(context_parts)

        logger.info(
            "Context assembled: %d provisions + %d definitions, %d chars "
            "(~%d tokens) — sending to %s",
            len(provisions), len(definitions), len(context), len(context) // 4,
            os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
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
                    mentioned_regs=mentioned_regs,
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

        # --- 7a. Bounded-agentic audit + revise loop ---
        # The Auditor verifies the draft's legal backbone and names provisions
        # to retrieve; gaps drive targeted re-retrieval; the Adjudicator
        # regenerates. Bounded by CRSS_AUDIT_MAX_ITERS. See application/_audit.py.
        audited = False
        if _should_audit(route.id) and full_answer:
            _audit_llm = _audit_model()  # cheap structured check (may be smaller)
            _gen_llm = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
            audit_context = context
            existing_ids = {p["article_id"] for p in provisions if p.get("article_id")}
            for _audit_i in range(_max_iters()):
                _audit_step_id = f"audit_{_audit_i + 1}"
                # Emit a 'running' step BEFORE the blocking audit call so the UI
                # shows activity instead of a frozen cursor during the ~15s call.
                yield {
                    "type": "step",
                    "id": _audit_step_id,
                    "label": f"Auditing answer against the legal backbone (pass {_audit_i + 1})…",
                }
                try:
                    findings = _audit_answer(
                        question, audit_context, full_answer, client, model=_audit_llm,
                    )
                except Exception as _exc:
                    logger.warning("Audit pass skipped: %s", _exc)
                    break
                # Update the same step in place with the verdict.
                yield {
                    "type": "step",
                    "id": _audit_step_id,
                    "label": (
                        f"Audit pass {_audit_i + 1}: {findings['verdict']} — "
                        f"{_format_findings(findings)}"
                    ),
                }
                # CRAG-style gate: only the expensive regeneration when the legal
                # backbone is actually broken. Minor issues leave the draft as-is.
                if not _needs_revision(findings):
                    break
                new_provs = _gap_retrieve(
                    findings, retriever,
                    target_celexes=target_celexes,
                    existing_ids=existing_ids,
                    max_add=_max_gap_refs(),
                )
                if not new_provs and not findings["issues"]:
                    break
                if new_provs:
                    for _p in new_provs:
                        existing_ids.add(_p["article_id"])
                    provisions.extend(new_provs)
                    audit_context = (
                        context
                        + "\n\n--- ADDITIONAL PROVISIONS (audit pass) ---\n\n"
                        + _format_context(new_provs)
                    )
                    yield {
                        "type": "step",
                        "id": "audit_retrieval",
                        "label": (
                            f"Audit gap-fill: pulled {len(new_provs)} additional "
                            "provision(s)"
                        ),
                    }
                _revision_user = _build_user_message(
                    question=question,
                    context=audit_context,
                    route=route,
                    sufficiency=sufficiency,
                    mentioned_regs=mentioned_regs,
                )
                _revision_messages = _build_revision_messages(
                    question, audit_context, findings, full_answer,
                    system_prompt=SYSTEM_PROMPT, user_message=_revision_user,
                )
                # Signal the UI to clear the prior draft and stream the revision
                # fresh, so the user sees continuous output instead of a frozen
                # cursor while the answer is regenerated.
                yield {
                    "type": "revising",
                    "label": f"Revising answer with audit findings (pass {_audit_i + 1})…",
                }
                try:
                    _revised = ""
                    with client.chat.stream(
                        model=_gen_llm, messages=_revision_messages, temperature=0.1,
                    ) as _rev_stream:
                        for _chunk in _rev_stream:
                            _delta = _chunk.data.choices[0].delta.content
                            if _delta:
                                _revised += _delta
                                yield {"type": "token", "content": _delta}
                    _revised = _strip_meta_leak(_revised).strip()
                    if _revised:
                        full_answer = _revised
                        audited = True
                except Exception as _exc:
                    logger.warning("Audit revision skipped: %s", _exc)
                    break

        # --- 7b. Backbone self-check (bounded, 1 call) ---
        # When a master-list article was injected, verify the answer covers all
        # its items.  Uses the actual retrieved text — no training-memory risk.
        # Set env CRSS_BACKBONE_SELFCHECK=0 to disable.
        if (
            has_backbone
            and backbone_provisions
            and os.environ.get("CRSS_BACKBONE_SELFCHECK", "1") != "0"
            and full_answer
        ):
            try:
                _backbone_text = _format_context(backbone_provisions[:1])[:2000]
                _check_resp = client.chat.complete(
                    model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
                    messages=[{
                        "role": "user",
                        "content": (
                            "BACKBONE ARTICLE (authoritative master list):\n"
                            f"{_backbone_text}\n\n"
                            "GENERATED ANSWER:\n"
                            f"{full_answer[:3000]}\n\n"
                            "Does the generated answer address each of the obligation "
                            "categories listed in the BACKBONE ARTICLE?\n"
                            "Output exactly COMPLETE if yes.\n"
                            "If no, output only the missing article numbers or "
                            "obligation headings, one per line. Max 30 words total."
                        ),
                    }],
                    temperature=0.0,
                    max_tokens=80,
                )
                _check_text = _check_resp.choices[0].message.content.strip()
                if not _check_text.upper().startswith("COMPLETE"):
                    full_answer += (
                        "\n\n---\n> **⚠ Completeness note:** "
                        "The following obligations from the statutory master list "
                        f"({backbone_label}) may not be fully addressed above: "
                        f"{_check_text}. Independent legal review recommended."
                    )
                    logger.info("Backbone self-check flagged gaps: %s", _check_text)
                else:
                    logger.debug("Backbone self-check: COMPLETE")
            except Exception as _exc:
                logger.warning("Backbone self-check skipped: %s", _exc)

        # --- 7c. Citation scope self-check (deterministic) ---
        # When the question is scoped to a single regulation, verify cited
        # Article/Annex/Recital refs appear in retrieved context refs.
        # Set env CRSS_CITATION_SCOPE_CHECK=0 to disable.
        if (
            target_celexes
            and len(target_celexes) == 1
            and mentioned_regs
            and len(mentioned_regs) == 1
            and os.environ.get("CRSS_CITATION_SCOPE_CHECK", "1") != "0"
            and full_answer
        ):
            _scope_reg_name = next(iter(mentioned_regs))
            try:
                _out_of_scope_refs = _out_of_scope_citation_refs(full_answer, provisions)
                if _out_of_scope_refs:
                    _scope_text = ", ".join(_out_of_scope_refs[:30])
                    full_answer += (
                        "\n\n---\n> **\u26a0 Citation scope note:** "
                        "The following citations are not present in the retrieved "
                        f"context for this question (scoped to {_scope_reg_name}): "
                        f"{_scope_text}. Please verify against the source provisions."
                    )
                    logger.info("Citation scope deterministic check flagged: %s", _scope_text)
                else:
                    logger.debug("Citation scope deterministic check: CLEAN")
            except Exception as _exc:
                logger.warning("Citation scope self-check skipped: %s", _exc)

        # --- 7d. Faithfulness verification (deterministic, no LLM call) ---
        # Verify every verbatim quote in the answer appears in the retrieved
        # corpus.  Flag mode prepends a warning block listing unverified
        # quotes.  Off by default — opt in with CRSS_FAITHFULNESS_CHECK=1
        # (flag) or =2 (strict; currently same behaviour, reserved for a
        # future single re-prompt loop).
        _faith_mode = _faithfulness_mode(os.environ.get("CRSS_FAITHFULNESS_CHECK"))
        if _faith_mode >= 1 and full_answer:
            try:
                _faith_report = _check_faithfulness(full_answer, provisions)
                if not _faith_report.ok:
                    # Enforce deterministic redaction so unverified verbatim
                    # quotations never survive in user-facing output.
                    full_answer = _remove_unverified_quotes(full_answer, _faith_report)
                    _faith_block = _build_faithfulness_warning(_faith_report)
                    if _faith_block:
                        full_answer = f"{_faith_block}\n\n{full_answer}"
                    logger.info(
                        "Faithfulness check flagged %d/%d quote(s)",
                        _faith_report.unverified_count,
                        _faith_report.total_quotes,
                    )
                else:
                    logger.debug(
                        "Faithfulness check: all %d quote(s) verified",
                        _faith_report.total_quotes,
                    )
                if _faith_mode == 2:
                    logger.warning(
                        "CRSS_FAITHFULNESS_CHECK=2 (strict) is not yet "
                        "implemented; behaving as flag mode."
                    )
            except Exception as _exc:
                logger.warning("Faithfulness check skipped: %s", _exc)

        # --- 7e. Compute composite confidence score ---
        _faith_report_for_conf = None
        if _faith_mode >= 1 and full_answer:
            try:
                _faith_report_for_conf = _check_faithfulness(full_answer, provisions)
            except Exception:
                pass
        _had_pointer_expansion = any(
            p.get("_pointer_expansion") for p in provisions
        )
        confidence = compute_confidence(
            sufficiency=sufficiency,
            provisions=provisions,
            faith_report=_faith_report_for_conf,
            had_corrective_pass=bool(corrective_actions),
            had_pointer_expansion=_had_pointer_expansion,
            had_role_provisions=bool(role_provisions),
            role_specs=role_specs,
            question=retrieval_question,
            mentioned_regs=mentioned_regs,
        )
        yield {
            "type": "confidence",
            "score": confidence["confidence_score"],
            "level": confidence["confidence_level"],
            "breakdown": confidence["breakdown"],
            "legal_force_distribution": confidence["legal_force_distribution"],
        }
        logger.info(
            "Confidence: %s (%.1f%%)",
            confidence["confidence_level"],
            confidence["confidence_score"] * 100,
        )

        final_answer = _postprocess_answer(
            full_answer,
            route,
            question=question,
            sufficiency=sufficiency,
            confidence=confidence,
            audited=audited,
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
