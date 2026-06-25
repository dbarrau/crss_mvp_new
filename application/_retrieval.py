"""Retrieval orchestration — all graph/vector retriever interactions.

Includes HyDE query generation, route-specific retrieval strategies,
context coverage evaluation, corrective retrieval passes, and audit
trace assembly.  Depends on ``_config`` and ``_routing`` but not on
any other application submodule.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from application._config import _REG_NAME_TO_CELEX
from application._routing import (
    _QuestionRoute,
    _ProvisionLookupTarget,
    _build_legal_qualification_targets,
)
from domain.legislation_catalog import (
    AI_ACT_CELEX,
    MDR_CELEX,
    IVDR_CELEX,
    GDPR_CELEX,
)

logger = logging.getLogger(__name__)

_AI_ACT_CELEX = AI_ACT_CELEX
_AI_ACT_PROHIBITED_PRACTICES_RE = re.compile(
    r"\b(prohibit(?:ed|ion|ions)?|ban(?:ned)?|forbidden|not\s+allowed)\b",
    re.IGNORECASE,
)

# Canonical definitions article per regulation. When a retrieved bag contains
# an EXEMPTS provision for one of these CELEX codes but no DEFINES provision
# for the same CELEX, the sufficiency check (status_anchor) treats the bag as
# incomplete and the corrective pass force-retrieves the definitions article.
# This prevents the failure mode where the LLM reads an obligation carve-out
# (e.g. MDR Article 5(5) in-house exemption) without the actor-status anchor
# (MDR Article 2 'manufacturer' definition) and concludes the actor has no
# status at all.
_DEFINITIONS_REF_BY_CELEX: dict[str, str] = {
    AI_ACT_CELEX: "Article 3",  # EU AI Act
    MDR_CELEX: "Article 2",     # MDR
    IVDR_CELEX: "Article 2",    # IVDR
    GDPR_CELEX: "Article 4",    # GDPR
}
_STATUS_ANCHOR_ROUTES: frozenset[str] = frozenset({
    "legal_qualification",
    "cross_regulation",
})

# ---------------------------------------------------------------------------
# HyDE query generation
# ---------------------------------------------------------------------------


def _decompose_question(question: str, client: Any) -> list[str]:
    """Decompose a broad compliance question into 3-5 targeted sub-questions.

    Each sub-question targets a distinct obligation tier, actor role, or
    condition implied by the original question.  Running community retrieval
    per sub-question ensures that each tier (e.g. GPAI general obligations,
    GPAI systemic-risk obligations, Article 5 prohibitions) gets its own
    independent retrieval slot instead of competing under a single query vector.

    Falls back to ``[question]`` if decomposition produces fewer than 2
    sub-questions (i.e. the question is already specific enough) or when
    *client* is None (e.g. in the eval harness where LLM calls are stubbed).
    """
    if client is None:
        return [question]
    resp = client.chat.complete(
        model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
        messages=[{
            "role": "user",
            "content": (
                "You are a regulatory analyst. Break the following compliance question "
                "into 3-5 specific sub-questions that together cover all obligation tiers, "
                "actor roles, and conditions it implies. Each sub-question should target "
                "a distinct regulatory tier or actor group.\n"
                "Return ONLY a numbered list, one sub-question per line. "
                "No introduction, no explanation.\n\n"
                f"Question: {question}"
            ),
        }],
        temperature=0.0,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()
    lines = [
        re.sub(r"^\d+[\.\ )]\s*", "", line).strip()
        for line in text.splitlines()
        if line.strip() and re.match(r"^\d+[\.\ )]", line.strip())
    ]
    return lines if len(lines) >= 2 else [question]


def _hyde_query(question: str, client: Any) -> str:
    """Generate a short hypothetical regulatory excerpt for HyDE retrieval.

    Produces a brief passage that resembles what EU regulatory text would
    say in response to *question*.  Encoding this passage with the same
    ``passage:`` prefix used for stored provisions places it in the same
    embedding space, yielding much better cosine similarity scores than
    encoding the question directly (query-to-document mismatch).

    The generation is capped at 100 tokens to minimise latency.

    HyDE output is never shown to the user — it is embedded and discarded — so it
    runs on a fast model by default (``CRSS_HYDE_MODEL``, default
    ``mistral-small-latest``).  Using the large generation model here adds several
    seconds per request with no quality benefit (the embedding only needs
    plausible regulatory vocabulary).
    """
    resp = client.chat.complete(
        model=os.environ.get("CRSS_HYDE_MODEL", "mistral-small-latest"),
        messages=[{
            "role": "user",
            "content": (
                "Write one short paragraph (50\u201380 words) of dense EU regulatory "
                "text that directly answers the question below. "
                "Use precise legal terminology. Output only the regulatory text, "
                "no headings, no citations, no explanations.\n\n"
                f"Question: {question}"
            ),
        }],
        temperature=0.0,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Provision merging helpers
# ---------------------------------------------------------------------------


def _merge_unique_provisions(
    base: list[dict],
    additions: list[dict],
    *,
    prepend: bool = False,
) -> int:
    """Merge provisions without duplicating ``article_id`` values."""
    existing_ids = {p.get("article_id") for p in base}
    new_items = [
        provision
        for provision in additions
        if provision.get("article_id") not in existing_ids
    ]
    if not new_items:
        return 0
    if prepend:
        base[:0] = new_items
    else:
        base.extend(new_items)
    return len(new_items)


# ---------------------------------------------------------------------------
# Lookup-target retrieval helpers
# ---------------------------------------------------------------------------


def _is_ai_act_prohibited_practices_question(
    question: str,
    target_celexes: set[str] | None,
) -> bool:
    """Return whether the question targets AI Act prohibited practices.

    Keeps the trigger deterministic and narrow: only fire when prohibition
    language is present and scope includes the AI Act.
    """
    if target_celexes and _AI_ACT_CELEX not in target_celexes:
        return False
    return bool(_AI_ACT_PROHIBITED_PRACTICES_RE.search(question))


def _retrieve_lookup_targets(
    retriever,
    targets: list[_ProvisionLookupTarget],
) -> list[dict]:
    """Retrieve curated provision targets with per-target CELEX scoping."""
    provisions: list[dict] = []
    for target in targets:
        matches = retriever.retrieve_by_refs(
            [target.ref],
            celex_filter=set(target.celexes) if target.celexes else None,
        )
        _merge_unique_provisions(provisions, matches)
    return provisions


def _has_lookup_target_coverage(
    provisions: list[dict],
    target: _ProvisionLookupTarget,
) -> bool:
    """Return whether a curated target is present in the retrieved provisions."""
    for provision in provisions:
        if provision.get("article_ref") != target.ref:
            continue
        if target.celexes and provision.get("celex") not in target.celexes:
            continue
        return True
    return False


def _detect_missing_status_anchors(
    provisions: list[dict],
    route_id: str,
) -> list[_ProvisionLookupTarget]:
    """Return DEFINES anchors required to ground EXEMPTS provisions in the bag.

    Structural sufficiency rule: when the retrieved bag contains a provision
    with ``provision_role == 'EXEMPTS'`` for some CELEX but no provision with
    ``provision_role == 'DEFINES'`` for that same CELEX, the bag is missing
    the actor-status definition that the exemption operates on. Without this
    anchor, the LLM cannot tell whether the exemption merely waives an
    obligation (correct) or removes the underlying legal status (incorrect).

    Only fires for routes where status reasoning matters
    (``legal_qualification``, ``cross_regulation``). For other routes the
    EXEMPTS provision is treated as a self-contained answer.

    Returns force-retrieval targets for the canonical definitions article of
    each affected CELEX. CELEXes outside ``_DEFINITIONS_REF_BY_CELEX`` are
    silently skipped (e.g. guidance documents do not have a definitions
    article in the same sense).
    """
    if route_id not in _STATUS_ANCHOR_ROUTES:
        return []

    exempts_celexes: set[str] = set()
    defines_celexes: set[str] = set()
    for provision in provisions:
        role = (provision.get("provision_role") or "").strip().upper()
        celex = provision.get("celex") or ""
        if not celex:
            continue
        if role == "EXEMPTS":
            exempts_celexes.add(celex)
        elif role == "DEFINES":
            defines_celexes.add(celex)

    missing: list[_ProvisionLookupTarget] = []
    for celex in sorted(exempts_celexes - defines_celexes):
        ref = _DEFINITIONS_REF_BY_CELEX.get(celex)
        if not ref:
            continue
        missing.append(_ProvisionLookupTarget(ref=ref, celexes=frozenset({celex})))
    return missing


# ---------------------------------------------------------------------------
# Direct and route-specific retrieval
# ---------------------------------------------------------------------------


def _retrieve_direct_provisions(
    question: str,
    retriever,
    *,
    explicit_refs: list[str],
    target_celexes: set[str] | None,
) -> list[dict]:
    """Run exact provision lookup, widening CELEX scope for paragraph misses."""
    if not explicit_refs:
        return []

    direct_provisions = retriever.retrieve_by_refs(
        explicit_refs,
        celex_filter=target_celexes,
    )
    if direct_provisions and target_celexes and len(target_celexes) < len(_REG_NAME_TO_CELEX):
        paragraph_match = re.search(r"paragraph\s+(\d+)", question, re.IGNORECASE)
        if paragraph_match:
            wanted_para = paragraph_match.group(1)
            para_ref = f"Paragraph {wanted_para}"
            has_para = any(
                child.get("ref") == para_ref
                for provision in direct_provisions
                for child in (provision.get("children") or [])
            )
            if not has_para:
                wider = retriever.retrieve_by_refs(explicit_refs, celex_filter=None)
                _merge_unique_provisions(direct_provisions, wider)
    return direct_provisions


# ---------------------------------------------------------------------------
# Retrieval expanders — the idempotent "pull more provisions" units
#
# Each expander performs exactly one of the mechanisms that used to live inline
# in ``_retrieve_route_provisions``'s ``if route.id == …`` ladder. They are the
# composable substrate the route plan sequences; A1.2 turns that sequencing into
# a declarative per-route plan, and A1.3 lets the corrective pass re-run the same
# expanders instead of duplicating their bodies. Each either returns its
# contribution or merges into a passed-in channel list in place, mirroring the
# pre-fold call sites exactly so the fold stays behaviour-neutral.
# ---------------------------------------------------------------------------


def _expand_legal_qualification_backbone(
    question: str,
    retriever,
    *,
    target_celexes: set[str] | None,
    role_specs: list[tuple[str, str]],
    explicit_refs: list[str],
    direct_provisions: list[dict],
) -> list[_ProvisionLookupTarget]:
    """Force-retrieve the curated legal-qualification backbone into the direct
    channel; return the targets (the sufficiency stage consumes them)."""
    mentioned_regs = {
        reg_name
        for reg_name, celex in _REG_NAME_TO_CELEX.items()
        if target_celexes and celex in target_celexes
    }
    targets = _build_legal_qualification_targets(
        question,
        mentioned_regs=mentioned_regs,
        role_specs=role_specs,
    )
    curated = _retrieve_lookup_targets(
        retriever,
        [target for target in targets if target.ref not in explicit_refs],
    )
    _merge_unique_provisions(direct_provisions, curated)
    return targets


def _inject_gdpr_cross_reg_backbone(
    question: str,
    retriever,
    *,
    target_celexes: set[str] | None,
    role_specs: list[tuple[str, str]],
    explicit_refs: list[str],
    direct_provisions: list[dict],
) -> None:
    """Force-retrieve the GDPR backbone for cross-regulation questions.

    The ``legal_qualification`` route already force-retrieves GDPR Articles 4, 6,
    9, 35 via ``_build_legal_qualification_targets``, but that route only fires
    when medical-device + AI Act overlap is detected. A GDPR + AI Act or
    GDPR + MDR cross-reg question (no medical-device signal) routes here instead,
    and without this the GDPR backbone arrives only by accident via vector
    retrieval — missing it caps answer quality. Reuses
    ``_build_legal_qualification_targets`` since it already handles GDPR correctly
    and is regulation-combination-aware.
    """
    mentioned = {
        reg_name
        for reg_name, celex in _REG_NAME_TO_CELEX.items()
        if target_celexes and celex in target_celexes
    }
    targets = _build_legal_qualification_targets(
        question,
        mentioned_regs=mentioned,
        role_specs=role_specs,
    )
    if not targets:
        return
    backbone = _retrieve_lookup_targets(
        retriever,
        [target for target in targets if target.ref not in explicit_refs],
    )
    added = _merge_unique_provisions(direct_provisions, backbone)
    if added:
        logger.info(
            "Cross-regulation backbone injection (GDPR in scope): "
            "%d provision(s) force-retrieved.",
            added,
        )


def _expand_classification_chain(
    question: str,
    retriever,
    *,
    client: Any,
    k: int,
    target_celexes: set[str] | None,
    hyde_text: str | None,
    hyde_builder,
) -> tuple[list[dict], str | None]:
    """Traverse the classification gate articles for the detected regulations.

    Seeds the chain from well-known classification gates; in-scope CELEXes with
    no gate entry (notably MDCG guidance, GDPR, implementing regs) are covered by
    a scoped vector pass so a named source still reaches context. Falls back to a
    plain HyDE vector pass when no graph edges resolve. Returns the provisions
    and the (possibly newly built) HyDE text.
    """
    chain_provisions: list[dict] = []

    # Map detected regulations to their primary classification gate articles.
    _GATE_ARTICLES: dict[str, list[str]] = {
        AI_ACT_CELEX: ["Article 6", "Article 51", "Article 5"],
        MDR_CELEX: ["Article 52", "Article 10"],
        IVDR_CELEX: ["Article 48", "Article 10"],
    }
    for celex, gate_refs in _GATE_ARTICLES.items():
        if target_celexes and celex not in target_celexes:
            continue
        chain_results = retriever.retrieve_by_chain(gate_refs, celex)
        for p in chain_results:
            if p.get("article_id") not in {cp.get("article_id") for cp in chain_provisions}:
                chain_provisions.append(p)

    # In-scope CELEXes that have no entry in _GATE_ARTICLES (notably MDCG
    # guidance documents, but also GDPR and implementing regulations) would
    # otherwise be silently dropped on this route — the gate-article loop only
    # knows how to traverse MDR/IVDR/AI-Act classification chains. When the user
    # explicitly scopes the question to such a document, retrieve it via a scoped
    # vector pass so the named source actually reaches the context.
    uncovered_celexes = (target_celexes or set()) - set(_GATE_ARTICLES)
    if uncovered_celexes:
        if hyde_text is None:
            hyde_text = hyde_builder(question, client)
        hyde_vec = retriever.encode_as_passage(hyde_text)
        uncovered_results = retriever.retrieve(
            question,
            k=k,
            target_celexes=uncovered_celexes,
            query_vec=hyde_vec,
        )
        for p in uncovered_results:
            if p.get("article_id") not in {cp.get("article_id") for cp in chain_provisions}:
                chain_provisions.append(p)

    if chain_provisions:
        return chain_provisions, hyde_text

    # No graph edges loaded yet — fall back to HyDE
    logger.debug("classification_chain: no graph edges found, falling back to HyDE")
    if hyde_text is None:
        hyde_text = hyde_builder(question, client)
    hyde_vec = retriever.encode_as_passage(hyde_text)
    provisions = retriever.retrieve(
        question,
        k=k,
        target_celexes=target_celexes,
        query_vec=hyde_vec,
    )
    return provisions, hyde_text


def _expand_community_summary(
    question: str,
    retriever,
    *,
    client: Any,
    target_celexes: set[str] | None,
    role_specs: list[tuple[str, str]],
) -> list[dict]:
    """Breadth-first community retrieval: decompose into sub-questions so each
    obligation tier gets an independent slot, then pin the detected roles'
    complete ``OBLIGATION_OF`` set so niche obligation articles (e.g. GPAI
    Art 53/55) are never lost to the vector contest.
    """
    _k_comm = 30
    sub_questions = _decompose_question(question, client)
    logger.debug(
        "community_summary_search: decomposed into %d sub-questions: %s",
        len(sub_questions), sub_questions,
    )
    # Distribute k budget across sub-questions; floor at 5 so thin sub-queries
    # still get meaningful community coverage.
    per_q_k = max(5, _k_comm // len(sub_questions))

    seen_article_ids: set[str | None] = set()
    provisions: list[dict] = []
    for sq in sub_questions:
        sq_vec = retriever.encode_as_query(sq)
        sq_provisions = retriever.retrieve_by_communities_hierarchical(
            sq,
            k_communities=per_q_k,
            k_provisions=per_q_k * 3,
            target_celexes=target_celexes,
            query_vec=sq_vec,
        )
        for p in sq_provisions:
            aid = p.get("article_id")
            if aid not in seen_article_ids:
                seen_article_ids.add(aid)
                provisions.append(p)

    # Pin the detected roles' statutory obligations via the same graph traversal
    # the role routes use (A2). This breadth route owes completeness, so it takes
    # the role's *complete* article obligation set (high k lifts the cap past the
    # largest role's article count; query_vec only orders the set), not a
    # relevance-capped top-k that would drop niche articles like GPAI
    # systemic-risk Art 55.
    if role_specs:
        comm_role_obligations = retriever.retrieve_by_roles(
            role_specs,
            k=40,
            query_vec=retriever.encode_as_query(question),
            target_celexes=target_celexes,
        )
        for p in comm_role_obligations:
            aid = p.get("article_id")
            if aid not in seen_article_ids:
                seen_article_ids.add(aid)
                provisions.append(p)
    return provisions


def _expand_hyde_vector(
    question: str,
    retriever,
    *,
    client: Any,
    k: int,
    target_celexes: set[str] | None,
    hyde_builder,
) -> tuple[list[dict], str]:
    """Single HyDE-expanded vector pass — the general retrieval default."""
    hyde_text = hyde_builder(question, client)
    hyde_vec = retriever.encode_as_passage(hyde_text)
    provisions = retriever.retrieve(
        question,
        k=k,
        target_celexes=target_celexes,
        query_vec=hyde_vec,
    )
    return provisions, hyde_text


def _inject_prohibited_practices_safety_net(
    question: str,
    retriever,
    *,
    target_celexes: set[str] | None,
    provisions: list[dict],
) -> None:
    """For AI Act prohibition-focused questions, force-include Article 5 so
    quote/citation checks validate against retrieved text."""
    if not _is_ai_act_prohibited_practices_question(question, target_celexes):
        return
    has_article_5 = any(
        p.get("celex") == _AI_ACT_CELEX
        and (p.get("article_ref") or "") == "Article 5"
        for p in provisions
    )
    if has_article_5:
        return
    art5 = retriever.retrieve_by_refs(["Article 5"], celex_filter={_AI_ACT_CELEX})
    added = _merge_unique_provisions(provisions, art5, prepend=True)
    if added:
        logger.info(
            "AI Act prohibited-practices safety net: injected Article 5 "
            "for prohibition-focused question.",
        )


def _retrieve_route_provisions(
    question: str,
    retriever,
    *,
    client: Any,
    k: int,
    route: _QuestionRoute,
    target_celexes: set[str] | None,
    explicit_refs: list[str],
    role_specs: list[tuple[str, str]],
    context_anchor_refs: list[str] | None = None,
    hyde_builder=_hyde_query,
) -> dict[str, Any]:
    """Execute the retrieval plan selected by the deterministic router.

    The route is a thin *policy* over the expanders defined above: it selects
    which seeds and which primary-bag expander run, in four phases —
    seed (direct + curated channels) → role → primary bag → merge → safety net.
    """
    direct_provisions: list[dict] = []
    role_provisions: list[dict] = []
    provisions: list[dict] = []
    hyde_text: str | None = None
    legal_qualification_targets: list[_ProvisionLookupTarget] = []

    # ── Seed phase: direct refs + curated backbones into the direct channel ──
    if route.id in {"provision_lookup", "cross_regulation", "legal_qualification"}:
        direct_provisions = _retrieve_direct_provisions(
            question,
            retriever,
            explicit_refs=explicit_refs,
            target_celexes=target_celexes,
        )
        if route.id == "provision_lookup":
            provisions = list(direct_provisions)

    if route.id == "legal_qualification":
        legal_qualification_targets = _expand_legal_qualification_backbone(
            question,
            retriever,
            target_celexes=target_celexes,
            role_specs=role_specs,
            explicit_refs=explicit_refs,
            direct_provisions=direct_provisions,
        )

    _GDPR_CELEX = GDPR_CELEX
    if route.id == "cross_regulation" and target_celexes and _GDPR_CELEX in target_celexes:
        _inject_gdpr_cross_reg_backbone(
            question,
            retriever,
            target_celexes=target_celexes,
            role_specs=role_specs,
            explicit_refs=explicit_refs,
            direct_provisions=direct_provisions,
        )

    # ── Role channel ─────────────────────────────────────────────────────────
    if route.id in {"role_obligations", "cross_regulation", "legal_qualification"} and role_specs:
        role_provisions = retriever.retrieve_by_roles(
            role_specs, k=max(6, k // 2),
            query_vec=retriever.encode_as_query(question),
            target_celexes=target_celexes,
        )
        if route.id == "role_obligations":
            provisions = list(role_provisions)

    should_run_hyde = (
        route.id in {"general_compliance", "cross_regulation", "legal_qualification"}
        or (route.id == "provision_lookup" and not provisions)
        or (route.id == "role_obligations" and not provisions)
        # Always run HyDE for definition_lookup so definition context and
        # retrieved provisions are both present.  Without this, detecting a
        # defined term (e.g. "personal data") in the question sets has_definitions=True
        # and suppresses vector retrieval entirely — leaving the LLM with only the
        # definition text and no authoritative provision (e.g. Art 6(1) for "lawful
        # bases") to ground its answer.
        or route.id == "definition_lookup"
    )

    # ── Primary-bag phase: exactly one expander produces the main bag ────────
    if route.id == "classification_chain":
        provisions, hyde_text = _expand_classification_chain(
            question,
            retriever,
            client=client,
            k=k,
            target_celexes=target_celexes,
            hyde_text=hyde_text,
            hyde_builder=hyde_builder,
        )
    elif route.id == "community_summary_search":
        provisions = _expand_community_summary(
            question,
            retriever,
            client=client,
            target_celexes=target_celexes,
            role_specs=role_specs,
        )
    elif should_run_hyde:
        provisions, hyde_text = _expand_hyde_vector(
            question,
            retriever,
            client=client,
            k=k,
            target_celexes=target_celexes,
            hyde_builder=hyde_builder,
        )

    # ── Merge phase: prepend the curated channels for the merge routes ───────
    if route.id in {"cross_regulation", "legal_qualification"}:
        if role_provisions:
            _merge_unique_provisions(provisions, role_provisions, prepend=True)
        if direct_provisions:
            _merge_unique_provisions(provisions, direct_provisions, prepend=True)

    # ── Context-anchor phase: force decisive topic anchors into the bag for ANY
    #    route. The router has already run, so this cannot reclassify the
    #    question; it reuses the direct-lookup expander and prepends, so the
    #    anchor (e.g. MDR Annex XVI for a wellbeing-app qualification that routes
    #    general_compliance) leads the rendered context.
    if context_anchor_refs:
        anchors = _retrieve_direct_provisions(
            question,
            retriever,
            explicit_refs=context_anchor_refs,
            target_celexes=target_celexes,
        )
        if anchors:
            _merge_unique_provisions(provisions, anchors, prepend=True)

    # ── Safety-net phase ─────────────────────────────────────────────────────
    _inject_prohibited_practices_safety_net(
        question,
        retriever,
        target_celexes=target_celexes,
        provisions=provisions,
    )

    return {
        "provisions": provisions,
        "direct_provisions": direct_provisions,
        "role_provisions": role_provisions,
        "hyde_text": hyde_text,
        "legal_qualification_targets": legal_qualification_targets,
    }


# ---------------------------------------------------------------------------
# Sufficiency evaluation
# ---------------------------------------------------------------------------


def _collect_context_celexes(
    provisions: list[dict],
    definitions: list[dict],
) -> set[str]:
    """Return the CELEX codes covered by the assembled context."""
    context_celexes = {
        provision.get("celex")
        for provision in provisions
        if provision.get("celex")
    }
    context_celexes.update(
        definition.get("celex")
        for definition in definitions
        if definition.get("celex")
    )
    return context_celexes


def _collect_context_refs(provisions: list[dict]) -> set[str]:
    """Return the provision refs already present in context."""
    return {
        provision.get("article_ref")
        for provision in provisions
        if provision.get("article_ref")
    }


def _collect_context_communities(provisions: list[dict]) -> set[str]:
    """Return non-empty community IDs present in retrieved provisions."""
    return {
        provision.get("community_id")
        for provision in provisions
        if provision.get("community_id")
    }


def _has_community_diversity(
    provisions: list[dict],
    *,
    min_communities: int = 2,
) -> bool:
    """Return whether retrieved provisions span at least *min_communities* clusters."""
    return len(_collect_context_communities(provisions)) >= min_communities


def _has_role_context(
    provisions: list[dict],
    role_provisions: list[dict],
) -> bool:
    """Return whether role-targeted retrieval evidence is present."""
    return bool(
        role_provisions
        or any(
            provision.get("matched_role") or provision.get("matched_role_id")
            for provision in provisions
        )
    )


def _evaluate_route_sufficiency(
    *,
    route: _QuestionRoute,
    question: str,
    explicit_refs: list[str],
    target_celexes: set[str] | None,
    role_specs: list[tuple[str, str]],
    provisions: list[dict],
    definitions: list[dict],
    direct_provisions: list[dict],
    role_provisions: list[dict],
    legal_qualification_targets: list[_ProvisionLookupTarget],
) -> dict[str, Any]:
    """Evaluate whether the selected route retrieved minimally sufficient evidence."""
    checks: list[dict[str, Any]] = []
    all_provisions = provisions + direct_provisions
    context_celexes = _collect_context_celexes(provisions, definitions)
    context_refs = _collect_context_refs(all_provisions)
    context_communities = _collect_context_communities(all_provisions)
    missing_refs = [ref for ref in explicit_refs if ref not in context_refs]
    missing_celexes = sorted((target_celexes or set()) - context_celexes)
    has_role_ctx = _has_role_context(provisions, role_provisions)
    missing_qualification_targets = [
        target
        for target in legal_qualification_targets
        if not _has_lookup_target_coverage(provisions + direct_provisions, target)
    ]
    missing_status_anchors = _detect_missing_status_anchors(all_provisions, route.id)

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    def add_community_diversity_check() -> None:
        if not context_communities:
            add_check(
                "community_diversity",
                True,
                "community metadata unavailable; diversity check skipped",
            )
            return
        has_diversity = _has_community_diversity(all_provisions)
        add_check(
            "community_diversity",
            has_diversity,
            "retrieval spans multiple communities"
            if has_diversity
            else "retrieval is concentrated in a single community",
        )

    if route.id == "definition_lookup":
        add_check(
            "definition_coverage",
            bool(definitions or provisions),
            "formal definitions or fallback provisions are present"
            if (definitions or provisions)
            else "no definition evidence was retrieved",
        )
    elif route.id == "provision_lookup":
        add_check(
            "explicit_refs",
            not missing_refs,
            "all explicitly requested provisions were retrieved"
            if not missing_refs
            else "missing explicit provision refs: " + ", ".join(missing_refs),
        )
    elif route.id == "role_obligations":
        add_check(
            "role_coverage",
            has_role_ctx,
            "role-aware evidence is present"
            if has_role_ctx
            else "no role-linked provisions were retrieved",
        )
    elif route.id == "cross_regulation":
        add_check(
            "cross_reg_coverage",
            not missing_celexes,
            "all targeted regulations are represented in context"
            if not missing_celexes
            else "missing regulation coverage for CELEX: " + ", ".join(missing_celexes),
        )
        if explicit_refs:
            add_check(
                "explicit_refs",
                not missing_refs,
                "all explicitly requested provisions were retrieved"
                if not missing_refs
                else "missing explicit provision refs: " + ", ".join(missing_refs),
            )
        if role_specs:
            add_check(
                "role_coverage",
                has_role_ctx,
                "role-aware evidence is present"
                if has_role_ctx
                else "no role-linked provisions were retrieved",
            )
        add_community_diversity_check()
        if missing_status_anchors:
            add_check(
                "status_anchor",
                False,
                "EXEMPTS provisions retrieved without matching DEFINES anchor for "
                + ", ".join(
                    f"{','.join(sorted(target.celexes or []))}:{target.ref}"
                    for target in missing_status_anchors
                ),
            )
    elif route.id == "legal_qualification":
        add_check(
            "qualification_backbone",
            not missing_qualification_targets,
            "all qualification backbone provisions were retrieved"
            if not missing_qualification_targets
            else "missing qualification targets: " + ", ".join(
                f"{target.ref}@{','.join(sorted(target.celexes or []))}"
                for target in missing_qualification_targets
            ),
        )
        add_check(
            "cross_reg_coverage",
            not missing_celexes,
            "all targeted regulations are represented in context"
            if not missing_celexes
            else "missing regulation coverage for CELEX: " + ", ".join(missing_celexes),
        )
        if explicit_refs:
            add_check(
                "explicit_refs",
                not missing_refs,
                "all explicitly requested provisions were retrieved"
                if not missing_refs
                else "missing explicit provision refs: " + ", ".join(missing_refs),
            )
        if role_specs:
            add_check(
                "role_coverage",
                has_role_ctx,
                "role-aware evidence is present"
                if has_role_ctx
                else "no role-linked provisions were retrieved",
            )
        add_community_diversity_check()
        if missing_status_anchors:
            add_check(
                "status_anchor",
                False,
                "EXEMPTS provisions retrieved without matching DEFINES anchor for "
                + ", ".join(
                    f"{','.join(sorted(target.celexes or []))}:{target.ref}"
                    for target in missing_status_anchors
                ),
            )
    elif route.id == "community_summary_search":
        has_community_results = any(
            p.get("_community_retrieval") for p in all_provisions
        )
        add_check(
            "community_coverage",
            bool(all_provisions),
            "community-sourced provisions are present"
            if all_provisions
            else "no provisions were retrieved from community search",
        )
        add_community_diversity_check()
        if _is_ai_act_prohibited_practices_question(question, target_celexes):
            has_article_5 = any(
                p.get("celex") == _AI_ACT_CELEX
                and (p.get("article_ref") or "") == "Article 5"
                for p in all_provisions
            )
            add_check(
                "ai_act_article5_anchor",
                has_article_5,
                "Article 5 anchor is present for prohibited-practices question"
                if has_article_5
                else "missing Article 5 anchor for prohibited-practices question",
            )
        if not has_community_results and all_provisions:
            add_check(
                "community_index_present",
                False,
                "community index not yet built; results came from HyDE fallback",
            )
    else:
        add_check(
            "context_presence",
            bool(definitions or provisions),
            "general compliance context is present"
            if (definitions or provisions)
            else "no provisions or definitions were retrieved",
        )
        if provisions:
            add_community_diversity_check()

    return {
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
        "missing_refs": missing_refs,
        "missing_celexes": missing_celexes,
        "missing_qualification_targets": [
            {
                "ref": target.ref,
                "celexes": sorted(target.celexes or []),
            }
            for target in missing_qualification_targets
        ],
        "missing_status_anchors": [
            {
                "ref": target.ref,
                "celexes": sorted(target.celexes or []),
            }
            for target in missing_status_anchors
        ],
        "needs_role_recovery": bool(role_specs) and not has_role_ctx,
        "context_celexes": sorted(context_celexes),
        "context_refs": sorted(context_refs),
        "context_communities": sorted(context_communities),
    }


# ---------------------------------------------------------------------------
# Corrective retrieval pass
# ---------------------------------------------------------------------------


def _run_corrective_retrieval_pass(
    question: str,
    retriever,
    *,
    client: Any,
    k: int,
    route: _QuestionRoute,
    target_celexes: set[str] | None,
    explicit_refs: list[str],
    role_specs: list[tuple[str, str]],
    provisions: list[dict],
    direct_provisions: list[dict],
    role_provisions: list[dict],
    definitions: list[dict],
    sufficiency: dict[str, Any],
    hyde_text: str | None,
    legal_qualification_targets: list[_ProvisionLookupTarget],
    hyde_builder=_hyde_query,
) -> dict[str, Any]:
    """Run a single bounded recovery pass when route coverage is insufficient.

    Each recovery re-runs one of the retrieval expanders/primitives with seeds
    derived from the sufficiency gap (missing curated targets, status anchors,
    explicit refs, role obligations, regulation coverage, the Article 5 anchor)
    and merges the result through one channel-aware helper that logs the action
    and recomputes sufficiency only when something new actually lands. It is not
    a parallel retrieval codepath — it reuses the same primitives the initial
    plan does, in priority order, with the running ``sufficiency`` shared so each
    later check sees what the earlier recoveries already filled.
    """
    actions: list[str] = []

    def recompute() -> dict[str, Any]:
        return _evaluate_route_sufficiency(
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

    def _recover(
        recovered: list[dict],
        action: str,
        *,
        channel: list[dict] | None = None,
        prepend: bool = True,
    ) -> None:
        """Merge a recovery into the main bag (and, optionally, the source
        channel it belongs to), then log + recompute only if anything landed."""
        nonlocal sufficiency
        added = 0
        if channel is not None:
            added += _merge_unique_provisions(channel, recovered)
        added += _merge_unique_provisions(provisions, recovered, prepend=prepend)
        if added:
            actions.append(action)
            sufficiency = recompute()

    def _targets(items: list[dict]) -> list[_ProvisionLookupTarget]:
        return [
            _ProvisionLookupTarget(
                ref=item["ref"],
                celexes=frozenset(item["celexes"]) if item["celexes"] else None,
            )
            for item in items
        ]

    if sufficiency["missing_qualification_targets"]:
        _recover(
            _retrieve_lookup_targets(
                retriever, _targets(sufficiency["missing_qualification_targets"])
            ),
            "recovered qualification backbone provisions",
            channel=direct_provisions,
        )

    if sufficiency.get("missing_status_anchors"):
        anchor_celexes = sorted(
            c for item in sufficiency["missing_status_anchors"] for c in item["celexes"]
        )
        _recover(
            _retrieve_lookup_targets(
                retriever, _targets(sufficiency["missing_status_anchors"])
            ),
            "recovered status-anchor (DEFINES) provisions for " + ", ".join(anchor_celexes),
            channel=direct_provisions,
        )

    if sufficiency["missing_refs"]:
        _recover(
            _retrieve_direct_provisions(
                question,
                retriever,
                explicit_refs=sufficiency["missing_refs"],
                target_celexes=target_celexes,
            ),
            f"recovered {len(sufficiency['missing_refs'])} explicit ref target(s)",
            channel=direct_provisions,
        )

    if sufficiency["needs_role_recovery"] and role_specs:
        _recover(
            retriever.retrieve_by_roles(
                role_specs, k=max(6, k // 2),
                query_vec=retriever.encode_as_query(question),
                target_celexes=target_celexes,
            ),
            "recovered role-linked provisions",
            channel=role_provisions,
        )

    # cross_regulation and legal_qualification recover missing-CELEX coverage
    # identically (the routes are mutually exclusive, so this fires for at most
    # one of them) via a HyDE-scoped vector pass over the missing regulations.
    if route.id in {"cross_regulation", "legal_qualification"} and sufficiency["missing_celexes"]:
        if not hyde_text:
            hyde_text = hyde_builder(question, client)
        hyde_vec = retriever.encode_as_passage(hyde_text)
        _recover(
            retriever.retrieve(
                question,
                k=max(k, len(sufficiency["missing_celexes"]) * 3),
                target_celexes=set(sufficiency["missing_celexes"]),
                query_vec=hyde_vec,
            ),
            "recovered missing regulation coverage for "
            + ", ".join(sufficiency["missing_celexes"]),
            prepend=False,
        )

    needs_article5_anchor = any(
        check["name"] == "ai_act_article5_anchor" and not check["passed"]
        for check in sufficiency["checks"]
    )
    if route.id == "community_summary_search" and needs_article5_anchor:
        _recover(
            retriever.retrieve_by_refs(["Article 5"], celex_filter={_AI_ACT_CELEX}),
            "recovered AI Act Article 5 anchor",
        )

    return {
        "actions": actions,
        "hyde_text": hyde_text,
        "sufficiency": sufficiency if actions else recompute(),
    }


# ---------------------------------------------------------------------------
# Audit trace
# ---------------------------------------------------------------------------


def _build_audit_trace(
    *,
    question: str,
    route: _QuestionRoute,
    mentioned_regs: set[str],
    target_celexes: set[str] | None,
    explicit_refs: list[str],
    role_specs: list[tuple[str, str]],
    definitions: list[dict],
    provisions: list[dict],
    direct_provisions: list[dict],
    role_provisions: list[dict],
    legal_qualification_targets: list[_ProvisionLookupTarget],
    hyde_text: str | None,
    inline_refs: list[str],
    corrective_actions: list[str],
    sufficiency: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured audit artifact for the full retrieval run."""
    return {
        "question": question,
        "route": {
            "id": route.id,
            "label": route.label,
            "rationale": route.rationale,
        },
        "signals": {
            "mentioned_regulations": sorted(mentioned_regs),
            "target_celexes": sorted(target_celexes or set()),
            "explicit_refs": explicit_refs,
            "role_specs": [
                {"term_normalized": term_normalized, "celex": celex}
                for term_normalized, celex in role_specs
            ],
            "legal_qualification_targets": [
                {
                    "ref": target.ref,
                    "celexes": sorted(target.celexes or []),
                }
                for target in legal_qualification_targets
            ],
        },
        "retrieval": {
            "definition_terms": [definition.get("term") for definition in definitions],
            "direct_provision_refs": [
                provision.get("article_ref") for provision in direct_provisions
            ],
            "role_provision_refs": [
                provision.get("article_ref") for provision in role_provisions
            ],
            "provision_refs": [
                provision.get("article_ref") for provision in provisions
            ],
            "provision_celexes": [
                provision.get("celex") for provision in provisions if provision.get("celex")
            ],
            "hyde_used": bool(hyde_text),
            "hyde_query_preview": hyde_text[:160] if hyde_text else None,
            "inline_refs": inline_refs,
            "corrective_actions": corrective_actions,
        },
        "sufficiency": sufficiency,
        "legal_force_distribution": {
            "binding": sum(
                1 for p in provisions if p.get("binding_force") == "binding"
            ),
            "non_binding": sum(
                1 for p in provisions if p.get("binding_force") == "non_binding"
            ),
            "unknown": sum(
                1 for p in provisions
                if p.get("binding_force") not in ("binding", "non_binding")
            ),
        },
    }
