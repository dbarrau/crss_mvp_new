#!/usr/bin/env python3
"""Audit the actor-role semantic layer in Neo4j.

This is a graph-hardening utility, not an agent benchmark. It verifies:
- which role terms are expected from the current definition layer
- which ActorRole nodes actually exist in Neo4j
- whether curated composite-role parents/children are materialized
- how many OBLIGATION_OF edges attach to each role
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from canonicalization.role_linker import _augment_with_derived_roles, _load_defined_terms, _select_actor_terms
from domain.ontology.actor_roles import COMPOSITE_ROLE_COMPONENTS
from infrastructure.graphdb.neo4j.loader import RegulationGraphLoader


def main() -> None:
    with RegulationGraphLoader.from_env() as loader:
        with loader._driver.session(database=loader._database) as session:  # noqa: SLF001
            defined_terms = _load_defined_terms(session)
            expected_terms = _augment_with_derived_roles(_select_actor_terms(defined_terms))

            expected_by_celex: dict[str, set[str]] = defaultdict(set)
            for row in expected_terms:
                expected_by_celex[row["celex"]].add(row["term_normalized"])

            actual_rows = session.run(
                "MATCH (r:ActorRole) "
                "RETURN r.celex AS celex, r.term_normalized AS term, "
                "       coalesce(r.source_type, 'unclassified') AS source_type "
                "ORDER BY r.celex, r.term_normalized"
            ).data()
            actual_by_celex: dict[str, set[str]] = defaultdict(set)
            actual_types_by_celex: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            for row in actual_rows:
                actual_by_celex[row["celex"]].add(row["term"])
                actual_types_by_celex[row["celex"]][row["source_type"]].add(row["term"])

            counts = session.run(
                "MATCH (r:ActorRole) "
                "OPTIONAL MATCH (p:Provision)-[:OBLIGATION_OF]->(r) "
                "RETURN r.celex AS celex, r.term_normalized AS term, count(p) AS obligations "
                "ORDER BY celex, term"
            ).data()

            include_rows = session.run(
                "MATCH (p:ActorRole)-[rel:INCLUDES_ROLE]->(c:ActorRole) "
                "RETURN p.celex AS celex, p.term_normalized AS parent, c.term_normalized AS child, "
                "       coalesce(rel.mapping_kind, 'unclassified') AS mapping_kind, "
                "       rel.basis_note AS basis_note "
                "ORDER BY celex, parent, child"
            ).data()

            equivalent_rows = session.run(
                "MATCH (l:ActorRole)-[rel:EQUIVALENT_ROLE]->(r:ActorRole) "
                "RETURN l.celex AS left_celex, l.term_normalized AS left_term, "
                "       r.celex AS right_celex, r.term_normalized AS right_term, "
                "       coalesce(rel.mapping_kind, 'unclassified') AS mapping_kind, "
                "       rel.scope AS scope, rel.confidence AS confidence "
                "ORDER BY left_celex, left_term, right_celex, right_term"
            ).data()

    print("=== ActorRole Coverage Audit ===")
    all_celexes = sorted(set(expected_by_celex) | set(actual_by_celex))
    for celex in all_celexes:
        expected = expected_by_celex.get(celex, set())
        actual = actual_by_celex.get(celex, set())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        print(f"\n[{celex}]")
        print(f"  Expected role terms: {len(expected)}")
        print(f"  Actual ActorRole nodes: {len(actual)}")
        print(f"  Missing: {missing if missing else 'none'}")
        print(f"  Extra:   {extra if extra else 'none'}")

        type_groups = actual_types_by_celex.get(celex, {})
        if type_groups:
            print("  Role provenance:")
            for source_type in sorted(type_groups):
                print(f"    - {source_type}: {sorted(type_groups[source_type])}")

        role_counts = [
            row for row in counts
            if row["celex"] == celex
        ]
        if role_counts:
            print("  OBLIGATION_OF counts:")
            for row in role_counts:
                print(f"    - {row['term']}: {row['obligations']}")

    print("\n=== Composite Role Coverage ===")
    includes_by_parent: dict[tuple[str, str], set[str]] = defaultdict(set)
    include_meta_by_pair: dict[tuple[str, str, str], tuple[str, str | None]] = {}
    for row in include_rows:
        includes_by_parent[(row["celex"], row["parent"])].add(row["child"])
        include_meta_by_pair[(row["celex"], row["parent"], row["child"])] = (
            row["mapping_kind"],
            row["basis_note"],
        )

    for (celex, parent), configured_children in sorted(COMPOSITE_ROLE_COMPONENTS.items()):
        actual_children = includes_by_parent.get((celex, parent), set())
        usable_children = {child for child in configured_children if child in actual_by_celex.get(celex, set())}
        missing = sorted(usable_children - actual_children)
        print(f"  - {celex} / {parent}")
        print(f"    expected usable children: {sorted(usable_children) if usable_children else 'none'}")
        print(f"    actual children:          {sorted(actual_children) if actual_children else 'none'}")
        print(f"    missing:                  {missing if missing else 'none'}")
        for child in sorted(actual_children):
            mapping_kind, basis_note = include_meta_by_pair[(celex, parent, child)]
            print(f"    relation {parent} -> {child}: {mapping_kind} ({basis_note or 'no basis note'})")

    print("\n=== Cross-Reg Role Mappings ===")
    if not equivalent_rows:
        print("  none")
    for row in equivalent_rows:
        print(
            "  - "
            f"{row['left_celex']} / {row['left_term']} -> "
            f"{row['right_celex']} / {row['right_term']} | "
            f"{row['mapping_kind']} | scope={row['scope'] or 'none'} | "
            f"confidence={row['confidence'] or 'none'}"
        )


if __name__ == "__main__":
    main()
