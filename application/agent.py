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
import random
import re
import time
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
from application._scoping import (                        # noqa: F401
    assess_scope as _assess_scope,
    render_clarification_markdown as _render_clarification_markdown,
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
    _collect_cites_targets,
    _format_definitions,
    _format_context,
    _trim_provisions_to_budget,
    _CONTEXT_CHAR_BUDGET,
    _community_summary_header,
)
from application._prompts import (                       # noqa: F401
    SYSTEM_PROMPT,
    structured_system_prompt,
    _build_route_answer_guidance,
    _build_user_message,
)
from application._grounded_citation import (             # noqa: F401
    build_pointer_index,
    resolve_pointers,
    _bold_references,
)
from application._grounded_answer import (               # noqa: F401
    GroundedAnswer,
    RenderResult,
    render_grounded_answer,
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
from application.verify import (                         # noqa: F401
    verify_answer as _verify_answer,
    VerificationResult as _VerificationResult,
)
from application.scenario import detect_scenario as _detect_scenario
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
# Transient-failure handling for Mistral streaming calls
# ---------------------------------------------------------------------------
# Mistral intermittently returns 5xx ("Service unavailable") or 429 (rate limit)
# for a request that succeeds moments later; without a retry these surface to the
# demo as an ``{"type": "error"}`` event (and a server-side stack trace) mid-answer.
# We retry such *transient* failures with exponential backoff + full jitter.
# Deterministic 4xx (bad request, auth) are not retryable and propagate at once.
_LLM_MAX_RETRIES = 4          # total attempts per streaming call
_LLM_BACKOFF_BASE = 0.75      # seconds; attempt N waits up to base * 2**N …
_LLM_BACKOFF_CAP = 8.0        # … capped here, before jitter


def _is_retryable_llm_error(exc: Exception) -> bool:
    """True for transient Mistral failures (429 rate-limit or any 5xx server error).

    Mistral's ``SDKError`` carries a numeric ``status_code``; we read it directly
    and fall back to sniffing the string form so the check still works if the SDK
    surface changes or a lower-level transport error is raised instead.
    """
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or 500 <= code <= 599
    return bool(re.search(r"\b(?:429|5\d{2})\b", str(exc)))


def _stream_chat_with_retry(client, *, model, messages, temperature):
    """Yield content deltas from a Mistral chat stream, retrying transient failures.

    Retries only while *no* delta has been emitted yet, so a client that appends
    tokens never sees a duplicated partial answer from a mid-stream restart. A
    failure after streaming has begun — or any non-transient error — propagates to
    the caller (surfaced as an ``error`` event by :func:`ask_stream`).
    """
    for attempt in range(_LLM_MAX_RETRIES):
        emitted = False
        try:
            with client.chat.stream(
                model=model, messages=messages, temperature=temperature,
            ) as stream:
                for chunk in stream:
                    delta = chunk.data.choices[0].delta.content
                    if delta:
                        emitted = True
                        yield delta
            return
        except Exception as exc:  # noqa: BLE001 — classify, then retry or re-raise
            last_attempt = attempt == _LLM_MAX_RETRIES - 1
            if emitted or last_attempt or not _is_retryable_llm_error(exc):
                raise
            ceiling = min(_LLM_BACKOFF_CAP, _LLM_BACKOFF_BASE * (2 ** attempt))
            wait = random.uniform(0, ceiling)
            logger.warning(
                "Transient LLM error (status %s); retry %d/%d in %.1fs: %s",
                getattr(exc, "status_code", "?"),
                attempt + 1, _LLM_MAX_RETRIES - 1, wait, str(exc)[:160],
            )
            time.sleep(wait)


def _grounded_structured() -> bool:
    """Whether to generate the answer via structured outputs (hard-enforced
    grounded generation contract).  Opt-in while under validation; the default
    remains the streaming free-text path.  See docs/grounded_generation_contract.md.
    """
    return os.environ.get("CRSS_GROUNDED_STRUCTURED", "0") == "1"


def _generate_grounded_answer(
    client, *, model, messages, provisions, definitions, fallback_refs=None
) -> "RenderResult":
    """Generate a structured GroundedAnswer and render it to markdown.

    The model returns prose + a typed ``citations`` channel (no field can hold
    authored quote text); the renderer substitutes verbatim source text / refs
    from the retrieved bag by node id.  ``fallback_refs`` is the retriever's
    global ``{id: (ref, regulation)}`` map, so a real but un-retrieved provision
    the model cites still renders its human-readable reference.  Non-streaming
    (``chat.parse``).
    """
    resp = client.chat.parse(
        response_format=GroundedAnswer,
        model=model,
        messages=messages,
        temperature=0.1,
    )
    parsed = resp.choices[0].message.parsed
    index = build_pointer_index(provisions, definitions)
    return render_grounded_answer(parsed, index, fallback_refs=fallback_refs)


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

        # --- 1. Detection ("understand the question") ---
        # One deterministic stage (no LLM), shared verbatim with the retrieval
        # net (scripts/eval_retrieval.py) so behaviour cannot drift between the
        # agent and its gate. The detection logic + diagnostics live in
        # scenario.py; here we surface its output as progress events and capture
        # the typed Scenario the spine organises around (understand -> clarify? ->
        # plan -> retrieve). Loose locals are rebound for the not-yet-migrated
        # plan/retrieve stages; the clarify gate below is the Scenario's first
        # consumer.
        det = _detect_scenario(retrieval_question, retriever, k)
        definitions = det.definitions
        mentioned_regs = det.mentioned_regs
        target_celexes = det.target_celexes
        role_specs = det.role_specs
        explicit_refs = det.explicit_refs
        context_anchor_refs = det.context_anchor_refs
        is_def_q = det.is_def_q
        concept_text = det.concept_text
        route = det.route
        k = det.k
        scenario = det.scenario

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
        yield {
            "type": "step",
            "id": "route",
            "label": f"Route: {route.label} — {route.rationale}",
        }

        # --- 2b. Ask-first scope gate ---
        # When an obligation question omits the decisive actor role, ask for it
        # before retrieving/generating rather than silently assuming a role
        # (the backbone of every compliance answer). Deterministic; no LLM.
        if os.environ.get("CRSS_CLARIFY", "1") != "0":
            scope = _assess_scope(scenario)
            if scope.needs_clarification and scope.clarification is not None:
                clar = scope.clarification
                logger.info("Scope gate: asking for missing slot %r", clar.slot)
                yield {
                    "type": "clarify",
                    "slot": clar.slot,
                    "question": clar.question,
                    "rationale": clar.rationale,
                    "options": [
                        {
                            "label": o.label,
                            "value": o.value,
                            "frameworks": o.frameworks,
                        }
                        for o in clar.options
                    ],
                }
                yield {
                    "type": "done",
                    "answer": _render_clarification_markdown(clar),
                    "clarification": True,
                }
                return

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
            context_anchor_refs=context_anchor_refs,
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
        # Promote cross-referenced provisions to first-class *citable* nodes so
        # the model has a real node id to point at instead of fabricating one
        # (the failure behind the [...] holes / invented node ids). Two sources,
        # precise first:
        #   (a) CITES graph edges the retriever already resolved into
        #       `cited_provisions` — reliable, unique node ids;
        #   (b) a regex fallback over prose for references not modeled as edges.
        # Both are bounded by _MAX_POINTER_REFS and marked `_pointer_expansion`
        # so they render with an `id:` line and a provenance tag.
        seen_ids = {p["article_id"] for p in provisions}
        added = 0
        inline_refs: list[str] = []  # populated by the textual fallback below

        # (a) CITES-edge promotion — by node id, so display_ref ambiguity can't
        # land the promotion on the wrong provision.
        cites_targets = _collect_cites_targets(
            provisions, seen_ids, celex_filter=target_celexes,
        )
        if cites_targets:
            for p in retriever.retrieve_by_ids(cites_targets[:_MAX_POINTER_REFS]):
                if p["article_id"] not in seen_ids:
                    p["_pointer_expansion"] = True
                    provisions.append(p)
                    seen_ids.add(p["article_id"])
                    added += 1

        # (b) Textual-ref fallback — fills the remaining budget for references
        # mentioned in prose but not modeled as CITES edges.
        if added < _MAX_POINTER_REFS:
            inline_refs = _extract_inline_refs(provisions)
            if inline_refs:
                for p in retriever.retrieve_by_refs(
                    inline_refs, celex_filter=target_celexes,
                ):
                    if p["article_id"] not in seen_ids:
                        p["_pointer_expansion"] = True
                        provisions.append(p)
                        seen_ids.add(p["article_id"])
                        added += 1
                        if added >= _MAX_POINTER_REFS:
                            break

        if added:
            logger.info(
                "Pointer expansion: added %d cross-referenced provision(s) "
                "(total %d).", added, len(provisions),
            )
            yield {
                "type": "step",
                "id": "pointers",
                "label": (
                    f"Cross-reference expansion: {added} additional "
                    f"provision(s) promoted to citable context"
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

        _sep = "\n\n---\n\n"
        _budgeted: list[dict] = provisions
        if provisions:
            if route.id == "community_summary_search":
                header = _community_summary_header(provisions)
                if header:
                    context_parts.append(header)

            # Bound the *total* prompt size: broad routes can retrieve 40+
            # provisions (~220 KB / ~56 K tokens), which inflates the large
            # model's time-to-first-token. Keep the highest-priority provisions
            # and drop the low-value tail. Budget the provision block against
            # what the non-provision parts already consume (definitions, the
            # community overview, applicability / negative-definition notes) —
            # those were previously uncounted, letting broad routes slip well
            # past the cap (166 KB seen in the demo). The backbone stays whole;
            # only the provision tail is trimmed to fit the remainder.
            _reserved = sum(len(part) for part in context_parts) + len(_sep) * len(
                context_parts
            )
            _prov_budget = max(0, _CONTEXT_CHAR_BUDGET - _reserved)
            _budgeted = _trim_provisions_to_budget(provisions, _prov_budget)
            if len(_budgeted) < len(provisions):
                logger.info(
                    "Context budget: kept %d of %d provisions (%d chars reserved "
                    "for definitions/overview; %d budgeted for provisions)",
                    len(_budgeted), len(provisions), _reserved, _prov_budget,
                )
            context_parts.append(_format_context(_budgeted))
        context = _sep.join(context_parts)

        logger.info(
            "Context assembled: %d of %d provisions + %d definitions, %d chars "
            "(~%d tokens) — sending to %s",
            len(_budgeted), len(provisions), len(definitions),
            len(context), len(context) // 4,
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

        _structured = _grounded_structured()
        # Global {id: (ref, regulation)} map so the citation resolver can render
        # a real but un-retrieved provision the model cites (e.g. AI Act Art 25)
        # as a human-readable reference instead of an empty husk.  Best-effort:
        # a retriever double without the method degrades to retrieved-bag only.
        _ref_index_fn = getattr(retriever, "reference_index", None)
        _fallback_refs = _ref_index_fn() if callable(_ref_index_fn) else {}
        _messages = [
            {
                "role": "system",
                "content": structured_system_prompt() if _structured else SYSTEM_PROMPT,
            },
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

        _gen_model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        full_answer = ""
        if _structured:
            # Hard-enforced grounded generation: the model returns a GroundedAnswer
            # (prose + typed citations by node id); quote text is rendered from the
            # retrieved bag, so authored/fabricated quotes are impossible in this
            # channel. chat.parse is non-streaming, so the answer is buffered and
            # emitted as one token event. Any failure falls back to streaming.
            yield {
                "type": "step",
                "id": "generate",
                "label": "Generating grounded answer (structured)…",
            }
            try:
                _render = _generate_grounded_answer(
                    client, model=_gen_model, messages=_messages,
                    provisions=provisions, definitions=definitions,
                    fallback_refs=_fallback_refs,
                )
                full_answer = _render.text
                logger.info(
                    "Structured grounded answer: %d quote / %d cite marker(s); "
                    "%d duplicate quote(s) downgraded; %d duplicate ref(s) suppressed%s",
                    len(_render.quoted_ids), len(_render.cited_ids),
                    len(_render.deduped_ids), len(_render.suppressed_ref_dups),
                    (
                        f"; dropped {len(_render.unresolved_markers)} marker(s) / "
                        f"{len(_render.unresolved_ids)} id(s)"
                        if (_render.unresolved_markers or _render.unresolved_ids)
                        else ""
                    ),
                )
                yield {"type": "token", "content": full_answer}
            except Exception as _exc:  # noqa: BLE001 — degrade to streaming
                logger.warning(
                    "Structured generation failed (%s); falling back to streaming",
                    _exc,
                )
                _structured = False
                full_answer = ""
        if not _structured:
            # Stream the draft as it is written so the reader gets immediate
            # feedback. These 'draft' tokens are shown in a secondary, clearly
            # in-progress preview (small, scrollable — NOT the final answer
            # bubble); the resolved, clean answer replaces them via the final
            # 'done' event. References are bold prose (nothing to resolve); a raw
            # `[quote: id]` pointer is stripped from the preview client-side, and
            # is resolved to a blockquote in the final answer. See the
            # buffered-vs-streaming note in docs/grounded_generation_contract.md.
            for delta in _stream_chat_with_retry(
                client,
                model=_gen_model,
                messages=_messages,
                temperature=0.1,
            ):
                full_answer += delta
                yield {"type": "draft", "content": delta}

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
                # The revision MUST use the same generation mode as the initial
                # answer: a structured answer revised via free-text streaming
                # reintroduces the very defects structured mode removes (inline
                # cites, misattribution) and can truncate mid-stream (observed on
                # HQ_001 only with audit on). So in structured mode the revision
                # regenerates through chat.parse + render too.
                _revision_messages = _build_revision_messages(
                    question, audit_context, findings, full_answer,
                    system_prompt=structured_system_prompt() if _structured else SYSTEM_PROMPT,
                    user_message=_revision_user,
                )
                # Signal the UI to clear the prior draft and stream the revision
                # fresh, so the user sees continuous output instead of a frozen
                # cursor while the answer is regenerated.
                yield {
                    "type": "revising",
                    "label": f"Revising answer with audit findings (pass {_audit_i + 1})…",
                }
                try:
                    if _structured:
                        _rev = _generate_grounded_answer(
                            client, model=_gen_llm, messages=_revision_messages,
                            provisions=provisions, definitions=definitions,
                            fallback_refs=_fallback_refs,
                        )
                        _revised = _strip_meta_leak(_rev.text).strip()
                        if _revised:
                            yield {"type": "token", "content": _revised}
                    else:
                        _revised = ""
                        for _delta in _stream_chat_with_retry(
                            client, model=_gen_llm, messages=_revision_messages,
                            temperature=0.1,
                        ):
                            _revised += _delta
                            yield {"type": "draft", "content": _delta}
                        _revised = _strip_meta_leak(_revised).strip()
                    if _revised:
                        full_answer = _revised
                        audited = True
                except Exception as _exc:
                    logger.warning("Audit revision skipped: %s", _exc)
                    break

        # --- 7b. Resolve grounded-citation pointers (deterministic) ---
        # The model emits [cite: <id>] / [quote: <id>] pointers (GROUNDED CITATION
        # CONTRACT in _prompts.py); resolve them to human refs / verbatim source
        # text before verification. Resolved [quote:] text is copied from the
        # pointed node, so it cannot be fabricated by construction; any residual
        # model-authored ">" quote is still caught by the faithfulness net below
        # (the "net backstop" strategy \u2014 see docs/grounded_generation_contract.md).
        # The index is built after the audit loop so it covers audit-added
        # provisions.
        if full_answer:
            _resolved = resolve_pointers(
                full_answer, build_pointer_index(provisions, definitions),
                fallback_refs=_fallback_refs,
            )
            logger.info(
                "Grounded citation: resolved %d quote / %d cite pointer(s) "
                "(%d via global ref map); %d duplicate ref(s) suppressed%s",
                len(_resolved.quoted_ids), len(_resolved.cited_ids),
                len(_resolved.global_ref_ids),
                len(_resolved.suppressed_ref_dups),
                (
                    f"; dropped {len(_resolved.unresolved_ids)} unresolved id(s): "
                    f"{_resolved.unresolved_ids[:10]}"
                    if _resolved.unresolved_ids else ""
                ),
            )
            full_answer = _resolved.text

        # --- 7c\u20137e. Post-generation verification (deterministic) ---
        # One stage over the retrieved evidence: citation-scope note (C4) \u2192
        # faithfulness/attribution redaction (C1/C2) \u2192 composite confidence (C5).
        # Env flags (CRSS_CITATION_SCOPE_CHECK, CRSS_FAITHFULNESS_CHECK) are read
        # inside verify_answer; see application/verify.py.
        _verification = _verify_answer(
            full_answer,
            provisions=provisions,
            definitions=definitions,
            role_provisions=role_provisions,
            sufficiency=sufficiency,
            target_celexes=target_celexes,
            mentioned_regs=mentioned_regs,
            role_specs=role_specs,
            corrective_actions=corrective_actions,
            question=retrieval_question,
        )
        full_answer = _verification.answer
        # Bold provision references deterministically — the model writes them as
        # plain prose and will not bold them itself. Runs after verification so
        # faithfulness matched against clean text; skips verbatim quote lines.
        full_answer = _bold_references(full_answer)
        confidence = _verification.confidence
        yield {
            "type": "confidence",
            "score": confidence["confidence_score"],
            "level": confidence["confidence_level"],
            "breakdown": confidence["breakdown"],
            "legal_force_distribution": confidence["legal_force_distribution"],
        }

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
