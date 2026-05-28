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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HyDE query generation
# ---------------------------------------------------------------------------


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

    if should_run_hyde:
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

    return {
        "provisions": provisions,
        "direct_provisions": direct_provisions,
        "curated_provisions": curated_provisions,
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
    context_celexes = _collect_context_celexes(provisions, definitions)
    context_refs = _collect_context_refs(provisions + direct_provisions)
    missing_refs = [ref for ref in explicit_refs if ref not in context_refs]
    missing_celexes = sorted((target_celexes or set()) - context_celexes)
    has_role_ctx = _has_role_context(provisions, role_provisions)
    missing_qualification_targets = [
        target
        for target in legal_qualification_targets
        if not _has_lookup_target_coverage(provisions + direct_provisions, target)
    ]

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

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
    else:
        add_check(
            "context_presence",
            bool(definitions or provisions),
            "general compliance context is present"
            if (definitions or provisions)
            else "no provisions or definitions were retrieved",
        )

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
        "needs_role_recovery": bool(role_specs) and not has_role_ctx,
        "context_celexes": sorted(context_celexes),
        "context_refs": sorted(context_refs),
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
    }
