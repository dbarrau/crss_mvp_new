"""Retrieval orchestration — all graph/vector retriever interactions.

Includes HyDE query generation, route-specific retrieval strategies,
context coverage evaluation, corrective retrieval passes, and audit
trace assembly.  Depends on ``_config`` and ``_routing`` but not on
any other application submodule.
"""
from __future__ import annotations

from collections import Counter

import logging
import os
import re
from typing import Any

from application._config import _OBLIGATION_MASTER_ARTICLES, _REG_NAME_TO_CELEX
from application._routing import (
    _QuestionRoute,
    _ProvisionLookupTarget,
    _build_legal_qualification_targets,
    _has_obligation_focus,
    _COMMUNITY_SUMMARY_Q_RE,
    _is_classification_chain_question,
)

logger = logging.getLogger(__name__)

_AI_ACT_CELEX = "32024R1689"
_AI_ACT_GPAI_REFS = frozenset({
    "Article 51",
    "Article 52",
    "Article 53",
    "Article 54",
    "Article 55",
})
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
    "32024R1689": "Article 3",  # EU AI Act
    "32017R0745": "Article 2",  # MDR
    "32017R0746": "Article 2",  # IVDR
    "32016R0679": "Article 4",  # GDPR
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
    sub-questions (i.e. the question is already specific enough).
    """
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
    """
    resp = client.chat.complete(
        model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
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


def _is_obligation_breadth_question(
    question: str,
    route: _QuestionRoute,
    role_specs: list[tuple[str, str]],
) -> bool:
    """Return True when the question asks broadly about an actor's obligations
    and a master article is known for at least one detected role.

    Requires explicit breadth language (all/every/comprehensive/...) so that
    specific obligation questions ("what does Article 26 require?") are NOT
    treated as backbone-breadth questions and continue to route normally.
    Triggers statutory backbone injection: the master list article is
    force-retrieved and prepended to context so the LLM uses it as a
    structural skeleton rather than reconstructing the list bottom-up.
    """
    if route.id not in {"community_summary_search", "role_obligations"}:
        return False
    if not role_specs or not _has_obligation_focus(question):
        return False
    if not _COMMUNITY_SUMMARY_Q_RE.search(question):
        return False
    return any(
        (role_term, celex) in _OBLIGATION_MASTER_ARTICLES
        for role_term, celex in role_specs
    )


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


def _get_obligation_backbone_refs(
    role_specs: list[tuple[str, str]],
    target_celexes: set[str] | None = None,
) -> list[_ProvisionLookupTarget]:
    """Return force-retrieval targets for obligation master articles.

    One target per (role, celex, ref) triple that has a known master article.
    A single actor may have multiple backbone articles (e.g. AI Act providers
    have Article 16 for High-Risk AI and Article 53 for GPAI models).
    CELEX is scoped so the retriever returns only the correct regulation.
    """
    targets: list[_ProvisionLookupTarget] = []
    seen: set[tuple[str, str]] = set()
    for role_term, celex in role_specs:
        if target_celexes and celex not in target_celexes:
            continue
        master_refs = _OBLIGATION_MASTER_ARTICLES.get((role_term, celex))
        if not master_refs:
            continue
        for master_ref in master_refs:
            if (master_ref, celex) not in seen:
                seen.add((master_ref, celex))
                targets.append(
                    _ProvisionLookupTarget(ref=master_ref, celexes=frozenset({celex}))
                )
    return targets


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
    has_definitions: bool,
    hyde_builder=_hyde_query,
) -> dict[str, Any]:
    """Execute the retrieval plan selected by the deterministic router."""
    direct_provisions: list[dict] = []
    curated_provisions: list[dict] = []
    role_provisions: list[dict] = []
    provisions: list[dict] = []
    hyde_text: str | None = None
    legal_qualification_targets: list[_ProvisionLookupTarget] = []
    map_results: list[dict] = []

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
        mentioned_regs = {
            reg_name
            for reg_name, celex in _REG_NAME_TO_CELEX.items()
            if target_celexes and celex in target_celexes
        }
        legal_qualification_targets = _build_legal_qualification_targets(
            question,
            mentioned_regs=mentioned_regs,
            role_specs=role_specs,
        )
        curated_provisions = _retrieve_lookup_targets(
            retriever,
            [
                target
                for target in legal_qualification_targets
                if target.ref not in explicit_refs
            ],
        )
        _merge_unique_provisions(direct_provisions, curated_provisions)

    if route.id in {"role_obligations", "cross_regulation", "legal_qualification"} and role_specs:
        role_provisions = retriever.retrieve_by_roles(role_specs, k=max(6, k // 2))
        if route.id == "role_obligations":
            provisions = list(role_provisions)

    should_run_hyde = (
        route.id in {"general_compliance", "cross_regulation", "legal_qualification"}
        or (route.id == "provision_lookup" and not provisions)
        or (route.id == "role_obligations" and not provisions)
        or (route.id == "definition_lookup" and not has_definitions)
    )

    if route.id == "classification_chain":
        # Determine the classification gate articles for the detected regulations.
        # Seed the chain from well-known classification gates; fall back to
        # vector retrieval when no gate can be inferred.
        from domain.ontology.legal_reasoning_chains import get_obligation_chain
        chain_provisions: list[dict] = []

        # Map detected regulations to their primary classification gate articles.
        _GATE_ARTICLES: dict[str, list[str]] = {
            "32024R1689": ["Article 6", "Article 51", "Article 5"],
            "32017R0745": ["Article 52", "Article 10"],
            "32017R0746": ["Article 48", "Article 10"],
        }
        for celex, gate_refs in _GATE_ARTICLES.items():
            if target_celexes and celex not in target_celexes:
                continue
            chain_results = retriever.retrieve_by_chain(gate_refs, celex)
            for p in chain_results:
                if p.get("article_id") not in {cp.get("article_id") for cp in chain_provisions}:
                    chain_provisions.append(p)

        if chain_provisions:
            provisions = chain_provisions
        else:
            # No graph edges loaded yet — fall back to HyDE
            logger.debug(
                "classification_chain: no graph edges found, falling back to HyDE"
            )
            hyde_text = hyde_builder(question, client)
            hyde_vec = retriever.encode_as_passage(hyde_text)
            provisions = retriever.retrieve(
                question,
                k=k,
                target_celexes=target_celexes,
                query_vec=hyde_vec,
            )

    elif route.id == "community_summary_search":
        # Decompose the question into sub-questions so each obligation tier
        # gets its own independent retrieval slot.  A broad question like
        # "all provider obligations" would otherwise compete with itself —
        # GPAI, Article 5, and high-risk conformity obligations sit at
        # different angles in embedding space.  Per-sub-question retrieval
        # guarantees structural coverage of every tier without relying on
        # a single query vector to capture all of them.
        _k_comm = 30
        sub_questions = _decompose_question(question, client)
        # For AI Act provider obligation-breadth questions, always guarantee a
        # GPAI-specific sub-question.  The LLM decomposer won't generate one
        # unless the user explicitly mentioned GPAI, causing GPAI community
        # embeddings to lose the cosine similarity contest against the denser
        # High-Risk AI clusters even though both are in scope.
        if _is_obligation_breadth_question(question, route, role_specs) and any(
            role_term == "provider" and celex == "32024R1689"
            for role_term, celex in role_specs
        ):
            _has_gpai_sq = any(
                "general-purpose" in sq.lower() or "gpai" in sq.lower()
                or "article 53" in sq.lower()
                for sq in sub_questions
            )
            if not _has_gpai_sq:
                sub_questions.append(
                    "What are the specific obligations of providers of general-purpose "
                    "AI models under Articles 53 to 55 of the EU AI Act, including "
                    "additional obligations for models posing systemic risk?"
                )
        logger.debug(
            "community_summary_search: decomposed into %d sub-questions: %s",
            len(sub_questions), sub_questions,
        )
        # Distribute k budget across sub-questions; floor at 5 so thin
        # sub-queries still get meaningful community coverage.
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

    elif should_run_hyde:
        hyde_text = hyde_builder(question, client)
        hyde_vec = retriever.encode_as_passage(hyde_text)
        provisions = retriever.retrieve(
            question,
            k=k,
            target_celexes=target_celexes,
            query_vec=hyde_vec,
        )

    if route.id in {"cross_regulation", "legal_qualification"}:
        if role_provisions:
            _merge_unique_provisions(provisions, role_provisions, prepend=True)
        if direct_provisions:
            _merge_unique_provisions(provisions, direct_provisions, prepend=True)

    # ── GPAI coverage safety net ─────────────────────────────────────────────
    # Regardless of route, whenever the detected roles include an AI Act
    # provider, verify that at least one GPAI article (51–55) is in the result
    # set.  If not, force-add Articles 53 and 55 via direct lookup.
    #
    # Rationale: the GPAI sub-question injection and backbone mechanism both
    # gate on specific routes and breadth-language keywords.  A question like
    # "what are the obligations of providers under the AI Act?" (no "all")
    # hitting role_obligations would only run retrieve_by_roles(), and GPAI
    # coverage then depends entirely on OBLIGATION_OF edges — which, even
    # after Fix 1, may still lose the cosine-similarity contest against the
    # much denser High-Risk AI cluster in community search paths.
    #
    # This check is deterministic, route-agnostic, and fires last so it never
    # duplicates provisions already present.
    if any(
        role_term == "provider" and celex == _AI_ACT_CELEX
        for role_term, celex in role_specs
    ) and (not target_celexes or _AI_ACT_CELEX in target_celexes):
        _has_gpai = any(
            p.get("celex") == _AI_ACT_CELEX
            and (p.get("article_ref") or "") in _AI_ACT_GPAI_REFS
            for p in provisions
        )
        if not _has_gpai:
            _gpai_provisions = retriever.retrieve_by_refs(
                ["Article 53", "Article 55"],
                celex_filter={_AI_ACT_CELEX},
            )
            added = _merge_unique_provisions(provisions, _gpai_provisions)
            if added:
                logger.info(
                    "GPAI safety net: injected %d GPAI provision(s) "
                    "(Articles 53/55) — none were present after primary retrieval.",
                    added,
                )
            else:
                logger.debug("GPAI safety net: provisions already present.")

    # ── AI Act prohibited-practices safety net ─────────────────────────────
    # For prohibition-focused questions (Article 5 family), force-include
    # Article 5 so quote/citation checks validate against retrieved text.
    if _is_ai_act_prohibited_practices_question(question, target_celexes):
        _has_article_5 = any(
            p.get("celex") == _AI_ACT_CELEX
            and (p.get("article_ref") or "") == "Article 5"
            for p in provisions
        )
        if not _has_article_5:
            _art5 = retriever.retrieve_by_refs(
                ["Article 5"],
                celex_filter={_AI_ACT_CELEX},
            )
            added = _merge_unique_provisions(provisions, _art5, prepend=True)
            if added:
                logger.info(
                    "AI Act prohibited-practices safety net: injected Article 5 "
                    "for prohibition-focused question.",
                )


    # Force-retrieve the master list article (e.g. Article 16 for providers)
    # and expose it separately so agent.py can render it as a completeness
    # anchor BEFORE the main provisions block.  Not merged into provisions so
    # it does not appear twice in context.
    has_backbone = False
    backbone_provisions: list[dict] = []
    backbone_label: str | None = None
    if _is_obligation_breadth_question(question, route, role_specs):
        backbone_targets = _get_obligation_backbone_refs(role_specs, target_celexes)
        if backbone_targets:
            backbone_provisions = _retrieve_lookup_targets(retriever, backbone_targets)
            if backbone_provisions:
                has_backbone = True
                backbone_label = ", ".join(t.ref for t in backbone_targets)
                logger.debug(
                    "Backbone injection: %s for roles %s",
                    backbone_label, role_specs,
                )

    return {
        "provisions": provisions,
        "direct_provisions": direct_provisions,
        "curated_provisions": curated_provisions,
        "role_provisions": role_provisions,
        "hyde_text": hyde_text,
        "legal_qualification_targets": legal_qualification_targets,
        "has_backbone": has_backbone,
        "backbone_provisions": backbone_provisions,
        "backbone_label": backbone_label,
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
    """Run a single bounded recovery pass when route coverage is insufficient."""
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

    if sufficiency["missing_qualification_targets"]:
        missing_targets = [
            _ProvisionLookupTarget(
                ref=item["ref"],
                celexes=frozenset(item["celexes"]) if item["celexes"] else None,
            )
            for item in sufficiency["missing_qualification_targets"]
        ]
        recovered_curated = _retrieve_lookup_targets(retriever, missing_targets)
        added = _merge_unique_provisions(direct_provisions, recovered_curated)
        added += _merge_unique_provisions(provisions, recovered_curated, prepend=True)
        if added:
            actions.append("recovered qualification backbone provisions")
            sufficiency = recompute()

    if sufficiency.get("missing_status_anchors"):
        anchor_targets = [
            _ProvisionLookupTarget(
                ref=item["ref"],
                celexes=frozenset(item["celexes"]) if item["celexes"] else None,
            )
            for item in sufficiency["missing_status_anchors"]
        ]
        recovered_anchors = _retrieve_lookup_targets(retriever, anchor_targets)
        added = _merge_unique_provisions(direct_provisions, recovered_anchors)
        added += _merge_unique_provisions(provisions, recovered_anchors, prepend=True)
        if added:
            actions.append(
                "recovered status-anchor (DEFINES) provisions for "
                + ", ".join(
                    sorted(
                        c
                        for item in sufficiency["missing_status_anchors"]
                        for c in item["celexes"]
                    )
                )
            )
            sufficiency = recompute()

    if sufficiency["missing_refs"]:
        recovered_direct = _retrieve_direct_provisions(
            question,
            retriever,
            explicit_refs=sufficiency["missing_refs"],
            target_celexes=target_celexes,
        )
        added = _merge_unique_provisions(direct_provisions, recovered_direct)
        added += _merge_unique_provisions(provisions, recovered_direct, prepend=True)
        if added:
            actions.append(
                f"recovered {len(sufficiency['missing_refs'])} explicit ref target(s)"
            )
            sufficiency = recompute()

    if sufficiency["needs_role_recovery"] and role_specs:
        recovered_roles = retriever.retrieve_by_roles(role_specs, k=max(6, k // 2))
        added = _merge_unique_provisions(role_provisions, recovered_roles)
        added += _merge_unique_provisions(provisions, recovered_roles, prepend=True)
        if added:
            actions.append("recovered role-linked provisions")
            sufficiency = recompute()

    if route.id == "cross_regulation" and sufficiency["missing_celexes"]:
        if not hyde_text:
            hyde_text = hyde_builder(question, client)
        hyde_vec = retriever.encode_as_passage(hyde_text)
        recovered_provisions = retriever.retrieve(
            question,
            k=max(k, len(sufficiency["missing_celexes"]) * 3),
            target_celexes=set(sufficiency["missing_celexes"]),
            query_vec=hyde_vec,
        )
        added = _merge_unique_provisions(provisions, recovered_provisions)
        if added:
            actions.append(
                "recovered missing regulation coverage for "
                + ", ".join(sufficiency["missing_celexes"])
            )
            sufficiency = recompute()

    if route.id == "legal_qualification" and sufficiency["missing_celexes"]:
        if not hyde_text:
            hyde_text = hyde_builder(question, client)
        hyde_vec = retriever.encode_as_passage(hyde_text)
        recovered_provisions = retriever.retrieve(
            question,
            k=max(k, len(sufficiency["missing_celexes"]) * 3),
            target_celexes=set(sufficiency["missing_celexes"]),
            query_vec=hyde_vec,
        )
        added = _merge_unique_provisions(provisions, recovered_provisions)
        if added:
            actions.append(
                "recovered missing regulation coverage for "
                + ", ".join(sufficiency["missing_celexes"])
            )
            sufficiency = recompute()

    needs_article5_anchor = any(
        check["name"] == "ai_act_article5_anchor" and not check["passed"]
        for check in sufficiency["checks"]
    )
    if route.id == "community_summary_search" and needs_article5_anchor:
        recovered_article_5 = retriever.retrieve_by_refs(
            ["Article 5"],
            celex_filter={_AI_ACT_CELEX},
        )
        added = _merge_unique_provisions(provisions, recovered_article_5, prepend=True)
        if added:
            actions.append("recovered AI Act Article 5 anchor")
            sufficiency = recompute()

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
