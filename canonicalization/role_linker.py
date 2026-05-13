"""Materialise actor-role awareness in Neo4j.

Creates:
- (:DefinedTerm)-[:INSTANTIATES]->(:ActorRole)
- (:ActorRole)-[:INCLUDES_ROLE]->(:ActorRole) for composite definitions
- (:Provision)-[:OBLIGATION_OF]->(:ActorRole) for high-confidence role-bearing provisions
- (:ActorRole)-[:EQUIVALENT_ROLE]->(:ActorRole) for curated retrieval alignments

This is a precision-first semantic enrichment stage. It does not attempt to
extract fully structured deontic objects.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from domain.ontology.actor_roles import (
    COMPOSITE_ROLE_BASIS,
    COMPOSITE_ROLE_COMPONENTS,
    CROSS_REG_EQUIVALENCES,
    DERIVED_ROLE_SPECS,
    EXACT_LEGAL_ROLE_SPECS,
    ROLE_SOURCE_TYPE_DEFINED_TERM,
    normalize_role_term,
)
from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)

_BATCH = 500
_MODAL_RE = re.compile(r"\b(shall(?:\s+not)?|must(?:\s+not)?|is\s+required\s+to|are\s+required\s+to)\b", re.I)


def _role_node_id(term_normalized: str, celex: str) -> str:
    return f"{celex}::role::{term_normalized}"


def _load_defined_terms(session) -> list[dict[str, Any]]:
    return session.run(
        "MATCH (d:DefinedTerm)-[:DEFINED_BY]->(p:Provision) "
        "RETURN d.id AS defined_term_id, d.term AS term, d.category AS category, "
        "       d.term_normalized AS term_normalized, d.celex AS celex, "
        "       d.regulation AS regulation, d.source_provision_id AS source_provision_id, "
        "       p.text AS definition_text"
    ).data()


def _load_provisions(session) -> list[dict[str, Any]]:
    return session.run(
        "MATCH (p:Provision) "
        "WHERE p.text IS NOT NULL AND p.text <> '' "
        "RETURN p.id AS id, p.celex AS celex, p.kind AS kind, "
        "       p.title AS title, p.text AS text, p.display_ref AS display_ref"
    ).data()


def _pluralize_last_word(term: str) -> str:
    parts = term.split()
    if not parts:
        return term
    last = parts[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = last[:-1] + "ies"
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        parts[-1] = last + "es"
    else:
        parts[-1] = last + "s"
    return " ".join(parts)


def _build_role_regex(term: str) -> re.Pattern:
    variants = {term.lower(), _pluralize_last_word(term.lower())}
    escaped = [re.escape(v) for v in sorted(variants, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.I)


def _first_sentence(text: str) -> str:
    normalized = text.replace("\xa0", " ").strip()
    match = re.split(r"(?<=[.;])\s+", normalized, maxsplit=1)
    return match[0][:400] if match else normalized[:400]


def _definition_body(text: str) -> str:
    parts = re.split(r"\bmeans\b", text or "", maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return parts[1].strip()
    return (text or "").strip()


def _cleanup_role_layer(session) -> None:
    session.run("MATCH (r:ActorRole) DETACH DELETE r")


def _detect_composite_role_ids(defined_terms: list[dict[str, Any]]) -> set[str]:
    by_celex: dict[str, list[dict[str, Any]]] = {}
    for row in defined_terms:
        by_celex.setdefault(row["celex"], []).append(row)

    composite_ids: set[str] = set()
    for celex, terms in by_celex.items():
        promotable_ids = {
            row["defined_term_id"]
            for row in terms
            if row.get("category") == "actor"
        }

        changed = True
        while changed:
            changed = False
            promotable_terms = [
                row for row in terms
                if row["defined_term_id"] in promotable_ids
            ]
            available_term_normalized = {row["term_normalized"] for row in terms}
            for row in terms:
                if row["defined_term_id"] in promotable_ids:
                    continue
                curated_components = COMPOSITE_ROLE_COMPONENTS.get(
                    (celex, row["term_normalized"]),
                    [],
                )
                usable_components = [
                    component for component in curated_components
                    if component in available_term_normalized or (celex, component) in DERIVED_ROLE_SPECS
                ]
                if usable_components and all(
                    any(candidate["term_normalized"] == component for candidate in promotable_terms)
                    or (celex, component) in DERIVED_ROLE_SPECS
                    for component in usable_components
                ):
                    promotable_ids.add(row["defined_term_id"])
                    composite_ids.add(row["defined_term_id"])
                    changed = True
                    continue
                clause = re.split(
                    r"[.;]",
                    _definition_body(row.get("definition_text") or "").lower(),
                    maxsplit=1,
                )[0]
                matched_terms: list[str] = []
                for actor in promotable_terms:
                    if re.search(r"\b" + re.escape(actor["term"].lower()) + r"\b", clause):
                        matched_terms.append(actor["term"].lower())
                if _is_composite_definition(clause, matched_terms):
                    promotable_ids.add(row["defined_term_id"])
                    composite_ids.add(row["defined_term_id"])
                    changed = True
    return composite_ids


def _select_actor_terms(defined_terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    composite_ids = _detect_composite_role_ids(defined_terms)
    return [
        row for row in defined_terms
        if row.get("category") == "actor"
        or row["defined_term_id"] in composite_ids
        or (row["celex"], row["term_normalized"]) in EXACT_LEGAL_ROLE_SPECS
    ]


def _augment_with_derived_roles(actor_terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented = [dict(row) for row in actor_terms]
    if not augmented:
        return augmented

    by_celex: dict[str, list[dict[str, Any]]] = {}
    for row in augmented:
        by_celex.setdefault(row["celex"], []).append(row)

    for (celex, term_normalized), spec in DERIVED_ROLE_SPECS.items():
        celex_terms = by_celex.get(celex)
        if not celex_terms:
            continue
        if any(row["term_normalized"] == term_normalized for row in celex_terms):
            continue
        if not any(
            composite_celex == celex and term_normalized in children
            for (composite_celex, _), children in COMPOSITE_ROLE_COMPONENTS.items()
        ):
            continue
        exemplar = celex_terms[0]
        derived_row = {
            "defined_term_id": None,
            "term": spec["term"],
            "category": "derived",
            "term_normalized": term_normalized,
            "celex": celex,
            "regulation": exemplar.get("regulation"),
            "source_provision_id": None,
            "definition_text": "",
            "source_type": spec["source_type"],
            "basis_note": spec["basis_note"],
        }
        celex_terms.append(derived_row)
        augmented.append(derived_row)

    return augmented


def _build_actor_roles(actor_terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in actor_terms:
        role_id = _role_node_id(row["term_normalized"], row["celex"])
        if role_id in seen:
            continue
        seen.add(role_id)
        roles.append({
            "id": role_id,
            "term": row["term"],
            "term_normalized": row["term_normalized"],
            "celex": row["celex"],
            "regulation": row.get("regulation"),
            "source_defined_term_id": row["defined_term_id"],
            "source_provision_id": row.get("source_provision_id"),
            "source_type": row.get("source_type", ROLE_SOURCE_TYPE_DEFINED_TERM),
            "basis_note": row.get("basis_note")
            or EXACT_LEGAL_ROLE_SPECS.get((row["celex"], row["term_normalized"]), {}).get("basis_note"),
        })
    return roles


def _build_instantiates_edges(actor_terms: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "defined_term_id": row["defined_term_id"],
            "role_id": _role_node_id(row["term_normalized"], row["celex"]),
        }
        for row in actor_terms
        if row.get("defined_term_id")
    ]


def _is_composite_definition(definition_text: str, component_terms: list[str]) -> bool:
    clause = re.split(r"[.;]", _definition_body(definition_text).lower(), maxsplit=1)[0]
    if len(component_terms) < 2:
        return False
    reduced = clause
    for term in sorted(component_terms, key=len, reverse=True):
        reduced = re.sub(r"\b" + re.escape(term.lower()) + r"\b", " ", reduced)
    reduced = re.sub(r"[,:;()]", " ", reduced)
    reduced = re.sub(r"\b(a|an|the|or|and|any|other)\b", " ", reduced)
    reduced = re.sub(r"\s+", " ", reduced).strip()
    return reduced == ""


def _build_includes_edges(actor_terms: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_celex: dict[str, list[dict[str, Any]]] = {}
    for row in actor_terms:
        by_celex.setdefault(row["celex"], []).append(row)

    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for celex, roles in by_celex.items():
        available_term_normalized = {role["term_normalized"] for role in roles}
        candidates = [
            {
                "term": role["term"],
                "term_lower": role["term"].lower(),
                "term_normalized": role["term_normalized"],
                "role_id": _role_node_id(role["term_normalized"], celex),
                "definition_text": role.get("definition_text") or "",
                "category": role.get("category"),
            }
            for role in roles
        ]
        for role in candidates:
            if role.get("category") == "actor":
                continue
            curated_components = COMPOSITE_ROLE_COMPONENTS.get(
                (celex, role["term_normalized"]),
                [],
            )
            usable_components = [
                component for component in curated_components
                if component in available_term_normalized
            ]
            if usable_components:
                for other in candidates:
                    if other["role_id"] == role["role_id"]:
                        continue
                    if other["term_normalized"] not in usable_components:
                        continue
                    pair = (role["role_id"], other["role_id"])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    edges.append({
                        "parent_role_id": role["role_id"],
                        "child_role_id": other["role_id"],
                        "mapping_kind": "definition_component",
                        "basis_note": COMPOSITE_ROLE_BASIS.get((celex, role["term_normalized"]), role.get("basis_note")),
                        "source_provision_id": role.get("source_provision_id"),
                    })
                continue
            clause = re.split(r"[.;]", _definition_body(role["definition_text"]).lower(), maxsplit=1)[0]
            matched_terms: list[str] = []
            for other in candidates:
                if other["role_id"] == role["role_id"]:
                    continue
                if re.search(r"\b" + re.escape(other["term_lower"]) + r"\b", clause):
                    matched_terms.append(other["term_lower"])
            if not _is_composite_definition(clause, matched_terms):
                continue
            for other in candidates:
                if other["role_id"] == role["role_id"]:
                    continue
                if other["term_lower"] not in matched_terms:
                    continue
                pair = (role["role_id"], other["role_id"])
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append({
                    "parent_role_id": role["role_id"],
                    "child_role_id": other["role_id"],
                    "mapping_kind": "definition_component",
                    "basis_note": COMPOSITE_ROLE_BASIS.get((celex, role["term_normalized"]), role.get("basis_note")),
                    "source_provision_id": role.get("source_provision_id"),
                })
    return edges


def _detect_modality(text: str, title: str | None) -> str | None:
    title_lower = (title or "").lower()
    if "obligation" in title_lower:
        return "obligation"
    if "shall not" in text.lower() or "must not" in text.lower():
        return "prohibition"
    if _MODAL_RE.search(text):
        return "obligation"
    return None


def _build_obligation_edges(
    actor_terms: list[dict[str, Any]],
    provisions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    by_celex: dict[str, list[dict[str, Any]]] = {}
    for row in actor_terms:
        row = dict(row)
        row["role_id"] = _role_node_id(row["term_normalized"], row["celex"])
        row["term_regex"] = _build_role_regex(row["term"])
        by_celex.setdefault(row["celex"], []).append(row)

    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for prov in provisions:
        candidates = by_celex.get(prov["celex"], [])
        if not candidates:
            continue
        title = prov.get("title") or ""
        sentence = _first_sentence(prov.get("text") or "")
        modality = _detect_modality(sentence, title)
        if not modality:
            continue
        title_lower = title.lower()
        for role in candidates:
            role_in_title = bool(title and role["term_regex"].search(title_lower))
            role_in_sentence = bool(role["term_regex"].search(sentence))
            if not role_in_title and not role_in_sentence:
                continue
            if role_in_title and "obligation" not in title_lower and not role_in_sentence:
                continue
            pair = (prov["id"], role["role_id"])
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "provision_id": prov["id"],
                "role_id": role["role_id"],
                "modality": modality,
                "cue": "title" if role_in_title and not role_in_sentence else "text",
            })
    return edges


def _build_equivalent_edges() -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for left, right, metadata in CROSS_REG_EQUIVALENCES:
        left_id = _role_node_id(normalize_role_term(left[0]), left[1])
        right_id = _role_node_id(normalize_role_term(right[0]), right[1])
        edges.append({
            "left_role_id": left_id,
            "right_role_id": right_id,
            "basis_note": metadata["basis_note"],
            "mapping_kind": metadata["mapping_kind"],
            "scope": metadata["scope"],
            "confidence": metadata["confidence"],
        })
        edges.append({
            "left_role_id": right_id,
            "right_role_id": left_id,
            "basis_note": metadata["basis_note"],
            "mapping_kind": metadata["mapping_kind"],
            "scope": metadata["scope"],
            "confidence": metadata["confidence"],
        })
    return edges


def _write_in_batches(session, cypher: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    total = 0
    for start in range(0, len(rows), _BATCH):
        chunk = rows[start : start + _BATCH]
        total += session.run(cypher, batch=chunk).single()["c"]
    return total


def link_roles(dry_run: bool = False) -> dict[str, int]:
    """Materialize actor-role awareness in Neo4j."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            defined_terms = _load_defined_terms(session)
            actor_terms = _augment_with_derived_roles(_select_actor_terms(defined_terms))
            provisions = _load_provisions(session)

            roles = _build_actor_roles(actor_terms)
            instantiates = _build_instantiates_edges(actor_terms)
            includes = _build_includes_edges(actor_terms)
            obligations = _build_obligation_edges(actor_terms, provisions)
            equivalents = _build_equivalent_edges()

            if dry_run:
                print("\n=== Role Linker (dry run) ===")
                print(f"  Actor terms:              {len(actor_terms)}")
                print(f"  ActorRole nodes:          {len(roles)}")
                print(
                    "  Derived ActorRole nodes:  "
                    f"{sum(1 for row in roles if row['source_type'] != ROLE_SOURCE_TYPE_DEFINED_TERM)}"
                )
                print(f"  INSTANTIATES edges:       {len(instantiates)}")
                print(f"  INCLUDES_ROLE edges:      {len(includes)}")
                print(f"  OBLIGATION_OF edges:      {len(obligations)}")
                print(f"  EQUIVALENT_ROLE edges:    {len(equivalents)}")
                print("  (dry run — no changes written)\n")
                return {
                    "actor_terms": len(actor_terms),
                    "actor_roles": len(roles),
                    "derived_roles": sum(1 for row in roles if row["source_type"] != ROLE_SOURCE_TYPE_DEFINED_TERM),
                    "instantiates": len(instantiates),
                    "includes_role": len(includes),
                    "obligation_of": len(obligations),
                    "equivalent_role": len(equivalents),
                }

            _cleanup_role_layer(session)

            role_count = _write_in_batches(
                session,
                "UNWIND $batch AS row "
                "MERGE (r:ActorRole {id: row.id}) "
                "SET r.term = row.term, "
                "    r.term_normalized = row.term_normalized, "
                "    r.celex = row.celex, "
                "    r.regulation = row.regulation, "
                "    r.source_defined_term_id = row.source_defined_term_id, "
                "    r.source_provision_id = row.source_provision_id, "
                "    r.source_type = row.source_type, "
                "    r.basis_note = row.basis_note "
                "RETURN count(r) AS c",
                roles,
            )
            inst_count = _write_in_batches(
                session,
                "UNWIND $batch AS row "
                "MATCH (d:DefinedTerm {id: row.defined_term_id}) "
                "MATCH (r:ActorRole {id: row.role_id}) "
                "MERGE (d)-[rel:INSTANTIATES]->(r) "
                "RETURN count(rel) AS c",
                instantiates,
            )
            inc_count = _write_in_batches(
                session,
                "UNWIND $batch AS row "
                "MATCH (p:ActorRole {id: row.parent_role_id}) "
                "MATCH (c:ActorRole {id: row.child_role_id}) "
                "MERGE (p)-[rel:INCLUDES_ROLE]->(c) "
                "SET rel.mapping_kind = row.mapping_kind, "
                "    rel.basis_note = row.basis_note, "
                "    rel.source_provision_id = row.source_provision_id "
                "RETURN count(rel) AS c",
                includes,
            )
            obl_count = _write_in_batches(
                session,
                "UNWIND $batch AS row "
                "MATCH (p:Provision {id: row.provision_id}) "
                "MATCH (r:ActorRole {id: row.role_id}) "
                "MERGE (p)-[rel:OBLIGATION_OF]->(r) "
                "SET rel.modality = row.modality, rel.cue = row.cue "
                "RETURN count(rel) AS c",
                obligations,
            )
            eq_count = _write_in_batches(
                session,
                "UNWIND $batch AS row "
                "MATCH (l:ActorRole {id: row.left_role_id}) "
                "MATCH (r:ActorRole {id: row.right_role_id}) "
                "MERGE (l)-[rel:EQUIVALENT_ROLE]->(r) "
                "SET rel.basis_note = row.basis_note, "
                "    rel.mapping_kind = row.mapping_kind, "
                "    rel.scope = row.scope, "
                "    rel.confidence = row.confidence "
                "RETURN count(rel) AS c",
                equivalents,
            )

            summary = {
                "actor_terms": len(actor_terms),
                "actor_roles": role_count,
                "derived_roles": sum(1 for row in roles if row["source_type"] != ROLE_SOURCE_TYPE_DEFINED_TERM),
                "instantiates": inst_count,
                "includes_role": inc_count,
                "obligation_of": obl_count,
                "equivalent_role": eq_count,
            }
            logger.info("Role linker summary: %s", summary)
            return summary
    finally:
        driver.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(link_roles())
