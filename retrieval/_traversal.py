"""Cypher-driven retrieval modes and lookups (stateless).

Free functions over ``(driver, db)`` — direct-reference lookup, node-id
promotion, actor-role obligations, legal-reasoning chains, cited-container
drilldown, and the DefinedTerm / reference-index lookups.  The only state any
of these need is the shared Neo4j driver (and, for role relevance ranking,
the dense index), so they are functions rather than a class.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from retrieval._config import _ROLE_OBLIGATION_CAP
from retrieval._cypher import (
    _CHAIN_SEED_LOOKUP_CYPHER,
    _CHAIN_TRAVERSE_CYPHER,
    _CITED_CHILDREN_CYPHER,
    _DIRECT_REF_CYPHER,
    _EXPAND_CYPHER,
    _ROLE_OBLIGATIONS_CYPHER,
    _SUBTREE_CYPHER,
    ref_norm_key,
)
from retrieval._dense import DenseIndex

logger = logging.getLogger(__name__)


def expand(driver, db: str, ids: list[str]) -> list[dict[str, Any]]:
    """Run the shared graph expansion over the given node ids."""
    with driver.session(database=db) as s:
        return s.run(_EXPAND_CYPHER, ids=ids).data()


def expand_cited_containers(driver, db: str, results: list[dict]) -> None:
    """Drill into cited provisions that are high-level containers.

    When a CITES edge points to a container node (e.g. "Annex XIV")
    whose own text is very short but has children, this method fetches
    the children's text and appends them to the cited_provisions list
    so the LLM sees the actual substantive content.
    """
    container_ids: list[str] = []
    for r in results:
        for c in r.get("cited_provisions") or []:
            text = c.get("text") or ""
            # Short text + id present → likely a container heading
            if c.get("id") and len(text) <= 80:
                container_ids.append(c["id"])
    if not container_ids:
        return
    # Deduplicate
    container_ids = list(dict.fromkeys(container_ids))
    with driver.session(database=db) as s:
        children_rows = s.run(
            _CITED_CHILDREN_CYPHER, ids=container_ids,
        ).data()
    if not children_rows:
        return
    # Group children by container
    children_by_container: dict[str, list[dict]] = {}
    for row in children_rows:
        cid = row["container_id"]
        children_by_container.setdefault(cid, []).append({
            "id": row["id"],
            "kind": row["kind"],
            "ref": row["ref"],
            "text": (row["text"] or "")[:500],
            "binding_force": row.get("binding_force"),
        })
    # Inject children into the cited_provisions entries
    injected = 0
    for r in results:
        cited = r.get("cited_provisions") or []
        extras: list[dict] = []
        for c in cited:
            kids = children_by_container.get(c.get("id", ""))
            if kids:
                for kid in kids[:8]:
                    extras.append(kid)
                    injected += 1
        if extras:
            cited.extend(extras)
    if injected:
        logger.info(
            "Container drilldown: injected %d children from %d "
            "container(s).",
            injected, len(children_by_container),
        )


_QUALIFIED_REF_RE = re.compile(r"^article\s+(\d{1,3}[a-z]?)\s*\(", re.IGNORECASE)


def _parent_article_ref(ref: str) -> str | None:
    """The bare-article parent of a qualified reference, else ``None``.

    "Article 2(68)" / "Article 2(58)(b)" -> "Article 2".  Annex/recital refs
    have no article parent -> ``None``.
    """
    m = _QUALIFIED_REF_RE.match(ref.strip())
    return f"Article {m.group(1)}" if m else None


def retrieve_by_refs(
    driver,
    db: str,
    refs: list[str],
    celex_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Direct lookup of provisions by display_ref + graph expansion."""
    if not refs:
        return []

    celexes = sorted(celex_filter) if celex_filter else None
    # CELEX scoping happens inside the Cypher (before its per-ref LIMIT)
    # so in-scope nodes can never be evicted by out-of-scope duplicates.
    with driver.session(database=db) as s:
        rows = s.run(_DIRECT_REF_CYPHER, refs=refs, celexes=celexes).data()

    # Parent-article fallback: a qualified ref that matched no node — e.g.
    # "Article 2(68)" when AI-Act Article 2 (Scope) has only 12 paragraphs, or a
    # mistyped point number — retries at the article level.  This grounds the
    # answer in the REAL article (its full subtree), so the model can state what
    # the article actually contains and that the point does not exist, instead of
    # confabulating from unrelated vector-fallback hits.
    matched = {ref_norm_key(r["display_ref"]) for r in rows if r.get("display_ref")}
    fallbacks: list[str] = []
    for ref in refs:
        if ref_norm_key(ref) in matched:
            continue
        parent = _parent_article_ref(ref)
        if parent and ref_norm_key(parent) not in matched and parent not in fallbacks:
            fallbacks.append(parent)
    if fallbacks:
        with driver.session(database=db) as s:
            rows += s.run(_DIRECT_REF_CYPHER, refs=fallbacks, celexes=celexes).data()

    # Deduplicate, prefer the shortest display_ref (outermost/parent node)
    seen: set[str] = set()
    top_ids: list[str] = []
    for row in rows:
        art_id = row["article_id"]
        if art_id not in seen:
            seen.add(art_id)
            top_ids.append(art_id)

    if not top_ids:
        return []

    results = expand(driver, db, top_ids)

    for r in results:
        r["score"] = 1.0  # perfect score — explicit structural match
        r["matched_leaf_id"] = None
        r["_direct_ref_match"] = True

    # Attach the ordered, faithfully-nested subtree for each directly-looked-up
    # provision so the context renderer can show it exactly as numbered in the
    # source document (chapeau → points → roman sub-items), rather than the
    # flattened embedding string.  One batched query for the whole result set.
    _attach_subtrees(driver, db, results)

    return results


def _attach_subtrees(driver, db: str, results: list[dict[str, Any]]) -> None:
    """Populate ``result["subtree"]`` with the ordered HAS_PART descendants.

    Each subtree entry is ``{id, ref, number, kind, text, depth}`` in document
    (pre-order) order, with the root at ``depth == 0``.  ``text`` is the node's
    *own* text — not the flattened ``text_for_analysis`` — so every semantic unit
    renders as the exact, referenceable text of the regulation.
    """
    ids = [r["article_id"] for r in results if r.get("article_id")]
    if not ids:
        return
    with driver.session(database=db) as s:
        rows = s.run(_SUBTREE_CYPHER, ids=ids).data()
    by_root: dict[str, list[dict]] = {}
    for row in rows:
        by_root.setdefault(row["root_id"], []).append({
            "id": row["id"],
            "ref": row["ref"],
            "number": row["number"],
            "kind": row["kind"],
            "text": row["text"],
            "depth": row["depth"],
        })
    for r in results:
        subtree = by_root.get(r["article_id"])
        if subtree:
            r["subtree"] = subtree


def retrieve_by_ids(driver, db: str, ids: list[str]) -> list[dict[str, Any]]:
    """Direct lookup + graph expansion of provisions by exact node id."""
    seen: set[str] = set()
    unique: list[str] = []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            unique.append(i)
    if not unique:
        return []

    results = expand(driver, db, unique)

    for r in results:
        r["score"] = 0.0
        r["matched_leaf_id"] = None
    return results


def retrieve_by_roles(
    driver,
    db: str,
    dense: DenseIndex,
    role_specs: list[tuple[str, str]],
    *,
    k: int = 8,
    query_vec: np.ndarray | None = None,
    target_celexes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the obligations of one or more resolved actor roles.

    See :meth:`retrieval.graph_retriever.GraphRetriever.retrieve_by_roles`
    for the full parameter documentation.
    """
    if not role_specs:
        return []

    role_ids = [f"{celex}::role::{term_normalized}" for term_normalized, celex in role_specs]
    celex_param = sorted(target_celexes) if target_celexes else None
    with driver.session(database=db) as s:
        rows = s.run(
            _ROLE_OBLIGATIONS_CYPHER, role_ids=role_ids, target_celexes=celex_param,
        ).data()

    if not rows:
        return []

    # Dedup by provision id, first occurrence wins (preserves the Cypher's
    # article-first ordering as the no-query fallback). Cross-regulation
    # scoping already happened in the Cypher.
    role_hits: dict[str, tuple[str | None, str | None]] = {}
    uniq: list[dict[str, Any]] = []
    for row in rows:
        art_id = row["article_id"]
        if art_id not in role_hits:
            role_hits[art_id] = (row.get("matched_role_id"), row.get("matched_role"))
            uniq.append(row)

    def _relevance(row: dict[str, Any]) -> float:
        if query_vec is None:
            return 0.0
        idx = dense.id_index.get(row["article_id"])
        if idx is None or dense.matrix is None:
            return -1.0  # no embedding: keep, but rank last
        return float(dense.matrix[idx] @ query_vec)

    articles = sorted(
        (r for r in uniq if r.get("kind") == "article"),
        key=_relevance, reverse=True,
    )
    others = sorted(
        (r for r in uniq if r.get("kind") != "article"),
        key=_relevance, reverse=True,
    )

    # Article-grained obligations lead; annex/recital fragments fill the
    # remaining budget. The cap is generous and floored so a role's full
    # article set is never crowded out by a small caller k.
    cap = max(k, _ROLE_OBLIGATION_CAP)
    top_rows = (articles + others)[:cap]
    top_ids = [r["article_id"] for r in top_rows]

    results = expand(driver, db, top_ids)

    for r in results:
        matched_role_id, matched_role = role_hits.get(r["article_id"], (None, None))
        r["score"] = 1.0
        r["matched_leaf_id"] = None
        r["matched_role_id"] = matched_role_id
        r["matched_role"] = matched_role
        r["_role_retrieval"] = True

    return results


def retrieve_by_chain(
    driver,
    db: str,
    refs: list[str],
    celex: str,
    *,
    seed_only: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve provisions reachable via legal reasoning edges.

    See :meth:`retrieval.graph_retriever.GraphRetriever.retrieve_by_chain`
    for the full parameter documentation.
    """
    if not refs:
        return []

    with driver.session(database=db) as s:
        seed_rows = s.run(
            _CHAIN_SEED_LOOKUP_CYPHER, refs=refs, celex=celex,
        ).data()

    seed_ids = [r["seed_id"] for r in seed_rows]
    if not seed_ids:
        logger.debug(
            "retrieve_by_chain: no seed nodes found for %s in %s — "
            "falling back to retrieve_by_refs",
            refs, celex,
        )
        return retrieve_by_refs(driver, db, refs, celex_filter={celex})

    if seed_only:
        linked_ids = seed_ids
    else:
        with driver.session(database=db) as s:
            chain_rows = s.run(
                _CHAIN_TRAVERSE_CYPHER, seed_ids=seed_ids,
            ).data()

        linked_ids = [r["article_id"] for r in chain_rows if r["article_id"]]
        if not linked_ids:
            logger.debug(
                "retrieve_by_chain: no linked provisions found for %s — "
                "returning seed provisions only",
                seed_ids,
            )
            linked_ids = seed_ids

    # Merge: seeds first, then chain-linked provisions
    all_ids = list(dict.fromkeys(seed_ids + linked_ids))

    results = expand(driver, db, all_ids)

    for r in results:
        is_seed = r["article_id"] in set(seed_ids)
        r["score"] = 1.0 if is_seed else 0.9
        r["matched_leaf_id"] = None
        r["_chain_retrieval"] = True
        r["_chain_seed"] = is_seed

    expand_cited_containers(driver, db, results)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# DefinedTerm / reference lookups
# ---------------------------------------------------------------------------

def load_reference_index(driver, db: str) -> dict[str, tuple[str, str]]:
    """Fetch ``{node_id: (display_ref, regulation)}`` for every provision."""
    with driver.session(database=db) as s:
        rows = s.run(
            "MATCH (p:Provision) "
            "WHERE p.display_ref IS NOT NULL "
            "RETURN p.id AS id, p.display_ref AS ref, "
            "       p.regulation_id AS reg "
            "UNION ALL "
            "MATCH (g:Guidance) "
            "WHERE g.display_ref IS NOT NULL "
            "RETURN g.id AS id, g.display_ref AS ref, "
            "       g.regulation_id AS reg"
        ).data()
    index = {r["id"]: (r["ref"] or "", r["reg"] or "") for r in rows}
    logger.info(
        "Loaded %d provision references for citation resolution.",
        len(index),
    )
    return index


def load_defined_terms_index(driver, db: str) -> dict[str, str]:
    """Fetch ``{lowercase_term: term_normalized}`` for all DefinedTerm nodes."""
    with driver.session(database=db) as s:
        rows = s.run(
            "MATCH (d:DefinedTerm) "
            "RETURN d.term AS term, d.term_normalized AS tn"
        ).data()
    return {r["term"].lower(): r["tn"] for r in rows}


def find_by_term(driver, db: str, term: str) -> list[dict[str, Any]]:
    """Exact-match lookup for a :DefinedTerm by its normalized term name."""
    # Mirror the normalisation used in definitions.py
    term_normalized = re.sub(r"\s+", "_", term.strip().lower())

    cypher = """\
MATCH (d:DefinedTerm {term_normalized: $term_normalized})
MATCH (d)-[:DEFINED_BY]->(p:Provision)
OPTIONAL MATCH (p)<-[:HAS_PART]-(art:Provision)
RETURN
    d.term                AS term,
    d.term_normalized     AS term_normalized,
    d.category            AS category,
    d.definition_type     AS definition_type,
    d.regulation          AS regulation,
    d.celex               AS celex,
    p.text                AS definition_text,
    p.id                  AS source_provision_id,
    art.display_ref       AS article_ref,
    art.display_path      AS article_path
"""
    with driver.session(database=db) as s:
        return s.run(cypher, term_normalized=term_normalized).data()


def find_by_category(
    driver,
    db: str,
    category: str,
    celex: str | None = None,
) -> list[dict[str, Any]]:
    """Return all :DefinedTerm nodes for a given semantic category."""
    if celex:
        cypher = """\
MATCH (d:DefinedTerm {category: $category, celex: $celex})
RETURN d.term AS term, d.term_normalized AS term_normalized,
       d.category AS category, d.regulation AS regulation,
       d.celex AS celex, d.source_provision_id AS source_provision_id
ORDER BY d.term_normalized
"""
        with driver.session(database=db) as s:
            return s.run(cypher, category=category, celex=celex).data()
    else:
        cypher = """\
MATCH (d:DefinedTerm {category: $category})
RETURN d.term AS term, d.term_normalized AS term_normalized,
       d.category AS category, d.regulation AS regulation,
       d.celex AS celex, d.source_provision_id AS source_provision_id
ORDER BY d.celex, d.term_normalized
"""
        with driver.session(database=db) as s:
            return s.run(cypher, category=category).data()
