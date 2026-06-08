#!/usr/bin/env python3
"""Build graph communities for Provision nodes and persist them in Neo4j.

This script performs an offline community-detection pass over the existing
Provision graph using HAS_PART and CITES relationships. It writes:

- p.community_id on each :Provision node
- :Community nodes (level=0)
- (:Provision)-[:MEMBER_OF]->(:Community) relationships

Quick start::

    python scripts/build_communities.py --wipe-existing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from dotenv import load_dotenv
from neo4j import GraphDatabase

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

logger = logging.getLogger(__name__)
_BATCH = 500


def _batched(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_nodes_and_edges(session, celex_filter: set[str] | None):
    where = ""
    params: dict[str, object] = {}
    if celex_filter:
        where = "WHERE p.celex IN $celexes"
        params["celexes"] = sorted(celex_filter)

    nodes = session.run(
        f"""
        MATCH (p:Provision)
        {where}
        RETURN p.id AS id, p.celex AS celex, p.kind AS kind
        """,
        **params,
    ).data()

    ids = [row["id"] for row in nodes]
    if not ids:
        return nodes, []

    edges = session.run(
        """
        UNWIND $ids AS nid
        MATCH (a:Provision {id: nid})-[r:HAS_PART|CITES]-(b:Provision)
        WHERE b.id IN $ids
        RETURN DISTINCT a.id AS src, b.id AS dst, type(r) AS rel_type
        """,
        ids=ids,
    ).data()
    return nodes, edges


def _build_graph(nodes: list[dict], edges: list[dict]) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes:
        graph.add_node(row["id"], celex=row.get("celex"), kind=row.get("kind"))

    for row in edges:
        src = row["src"]
        dst = row["dst"]
        if src == dst:
            continue
        # HAS_PART (structural adjacency within a chapter) should dominate over
        # CITES (legal cross-references) so that Louvain respects chapter
        # boundaries.  A HAS_PART edge of 3.0 vs CITES of 1.5 ensures that
        # article-level cross-references cannot override the chapter structure.
        weight = 1.5 if row.get("rel_type") == "CITES" else 3.0
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += weight
        else:
            graph.add_edge(src, dst, weight=weight)

    return graph


def _detect_communities(graph: nx.Graph, seed: int, celex_prefix: str = "") -> dict[str, str]:
    if graph.number_of_nodes() == 0:
        return {}

    # Prefer python-louvain when available; fall back to networkx's
    # built-in implementation to keep this script usable without extra setup.
    partition_by_node: dict[str, int]
    try:
        import community as community_louvain  # type: ignore

        partition_by_node = community_louvain.best_partition(
            graph,
            weight="weight",
            random_state=seed,
        )
    except Exception:
        louvain_fn = getattr(nx.algorithms.community, "louvain_communities", None)
        if louvain_fn is None:
            communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        else:
            communities = list(louvain_fn(graph, weight="weight", seed=seed))

        partition_by_node = {}
        for community_idx, community_nodes in enumerate(communities):
            for node_id in community_nodes:
                partition_by_node[node_id] = community_idx

    by_partition: dict[int, list[str]] = defaultdict(list)
    for node_id, partition_id in partition_by_node.items():
        by_partition[int(partition_id)].append(node_id)

    ordered_groups = sorted(
        by_partition.values(),
        key=lambda group: (-len(group), min(group)),
    )

    # Include the CELEX prefix so IDs are globally unique across regulations.
    # e.g. "community::32024R1689::0001" — prevents cross-regulation collisions.
    prefix = f"community::{celex_prefix}::" if celex_prefix else "community::"
    node_to_community: dict[str, str] = {}
    for idx, group in enumerate(ordered_groups, start=1):
        community_id = f"{prefix}{idx:04d}"
        for node_id in group:
            node_to_community[node_id] = community_id
    return node_to_community


def _wipe_existing(session) -> None:
    session.run("MATCH (p:Provision) REMOVE p.community_id")
    session.run("MATCH (:Provision)-[r:MEMBER_OF]->(:Community) DELETE r")
    session.run("MATCH (c:Community) DETACH DELETE c")


def _persist_assignments(session, node_to_community: dict[str, str], node_meta: dict[str, dict]) -> None:
    assignments = [
        {"id": node_id, "community_id": community_id}
        for node_id, community_id in node_to_community.items()
    ]

    for chunk in _batched(assignments, _BATCH):
        session.run(
            """
            UNWIND $batch AS row
            MATCH (p:Provision {id: row.id})
            SET p.community_id = row.community_id
            """,
            batch=chunk,
        )

    community_rows: list[dict] = []
    by_community: dict[str, list[str]] = defaultdict(list)
    for node_id, community_id in node_to_community.items():
        by_community[community_id].append(node_id)

    for community_id, members in by_community.items():
        regulations = sorted(
            {
                node_meta[node_id]["celex"]
                for node_id in members
                if node_meta[node_id].get("celex")
            }
        )
        community_rows.append(
            {
                "id": community_id,
                "level": 0,
                "member_count": len(members),
                "regulations": regulations,
                "source": "louvain",
            }
        )

    for chunk in _batched(community_rows, _BATCH):
        session.run(
            """
            UNWIND $batch AS row
            MERGE (c:Community {id: row.id})
            SET c.level = row.level,
                c.member_count = row.member_count,
                c.regulations = row.regulations,
                c.source = row.source,
                c.updated_at = datetime()
            """,
            batch=chunk,
        )

    for chunk in _batched(assignments, _BATCH):
        session.run(
            """
            UNWIND $batch AS row
            MATCH (p:Provision {id: row.id})
            MATCH (c:Community {id: row.community_id})
            MERGE (p)-[:MEMBER_OF]->(c)
            """,
            batch=chunk,
        )


def _chapter_key_from_path(path_string: str | None) -> str | None:
    """Extract the chapter-level segment from a ``path_string``.

    Example: ``"32024R1689_document/32024R1689_enc_1/32024R1689_cpt_III/..."``
    → ``"32024R1689_cpt_III"``

    Returns ``None`` when there is no chapter segment (recitals, preamble, etc.)
    """
    if not path_string:
        return None
    parts = path_string.split("/")
    if len(parts) < 3:
        return None
    seg = parts[2]
    # Accept both chapter (_cpt_) and annex (_anx_) as chapter-level groupings
    if "_cpt_" in seg or "_anx_" in seg:
        return seg
    return None


def _build_level1_communities(
    session,
    node_to_community: dict[str, str],
) -> int:
    """Aggregate Level-0 communities into chapter-level Level-1 communities.

    For each Level-0 community, the dominant chapter (most-represented
    ``path_string`` segment[2]) is used as the Level-1 grouping key.
    The Level-1 community ID is ``"community::L1::<chapter_seg>"``.

    Returns the number of Level-1 Community nodes created / updated.
    """
    # ------------------------------------------------------------------
    # 1. Fetch path_string for every provision in the communities
    # ------------------------------------------------------------------
    provision_ids = list(node_to_community.keys())
    rows = session.run(
        """
        UNWIND $ids AS pid
        MATCH (p:Provision {id: pid})
        RETURN p.id AS id, p.path_string AS path_string, p.celex AS celex
        """,
        ids=provision_ids,
    ).data()

    provision_chapter: dict[str, str | None] = {
        r["id"]: _chapter_key_from_path(r.get("path_string"))
        for r in rows
    }
    provision_celex: dict[str, str] = {
        r["id"]: (r.get("celex") or "")
        for r in rows
    }

    # ------------------------------------------------------------------
    # 2. Find dominant chapter for each Level-0 community
    # ------------------------------------------------------------------
    by_l0: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    l0_celexes: dict[str, set[str]] = defaultdict(set)

    for node_id, community_id in node_to_community.items():
        chapter = provision_chapter.get(node_id)
        if chapter:
            by_l0[community_id][chapter] += 1
        celex = provision_celex.get(node_id, "")
        if celex:
            l0_celexes[community_id].add(celex)

    l0_to_l1: dict[str, str] = {}
    for l0_id, chapter_counts in by_l0.items():
        dominant = max(chapter_counts, key=chapter_counts.__getitem__)
        l0_to_l1[l0_id] = f"community::L1::{dominant}"

    if not l0_to_l1:
        return 0

    # ------------------------------------------------------------------
    # 3. Build Level-1 metadata
    # ------------------------------------------------------------------
    l1_l0_members: dict[str, list[str]] = defaultdict(list)
    l1_regulations: dict[str, set[str]] = defaultdict(set)
    for l0_id, l1_id in l0_to_l1.items():
        l1_l0_members[l1_id].append(l0_id)
        l1_regulations[l1_id].update(l0_celexes.get(l0_id, set()))

    # Derive human-readable label from the chapter segment
    # e.g. "32024R1689_cpt_III" → "Chapter III (32024R1689)"
    def _label(l1_id: str) -> str:
        seg = l1_id.replace("community::L1::", "")
        celex_part, _, rest = seg.partition("_")
        kind, _, roman = rest.partition("_")
        kind_word = "Chapter" if kind == "cpt" else "Annex"
        return f"{kind_word} {roman} ({celex_part})"

    l1_rows = [
        {
            "id": l1_id,
            "level": 1,
            "source_communities": sorted(members),
            "member_count": len(members),
            "regulations": sorted(l1_regulations[l1_id]),
            "source": "chapter_aggregation",
            "label": _label(l1_id),
        }
        for l1_id, members in l1_l0_members.items()
    ]

    # ------------------------------------------------------------------
    # 4. Persist Level-1 Community nodes
    # ------------------------------------------------------------------
    for chunk in _batched(l1_rows, _BATCH):
        session.run(
            """
            UNWIND $batch AS row
            MERGE (c:Community {id: row.id})
            SET c.level = row.level,
                c.source_communities = row.source_communities,
                c.member_count = row.member_count,
                c.regulations = row.regulations,
                c.source = row.source,
                c.label = row.label,
                c.updated_at = datetime()
            """,
            batch=chunk,
        )

    # ------------------------------------------------------------------
    # 5. Set parent_community_id on Level-0 Community nodes
    # ------------------------------------------------------------------
    l0_parent_rows = [
        {"l0_id": l0_id, "l1_id": l1_id}
        for l0_id, l1_id in l0_to_l1.items()
    ]
    for chunk in _batched(l0_parent_rows, _BATCH):
        session.run(
            """
            UNWIND $batch AS row
            MATCH (l0:Community {id: row.l0_id})
            SET l0.parent_community_id = row.l1_id
            """,
            batch=chunk,
        )

    logger.info(
        "Level-1 communities built: %d chapter groups from %d Level-0 communities.",
        len(l1_rows),
        len(l0_to_l1),
    )
    return len(l1_rows)


def _fetch_all_celexes(session) -> list[str]:
    """Return the distinct CELEX codes present in the Provision graph."""
    rows = session.run(
        "MATCH (p:Provision) WHERE p.celex IS NOT NULL "
        "RETURN DISTINCT p.celex AS celex ORDER BY celex"
    ).data()
    return [r["celex"] for r in rows]


def build_communities(*, seed: int, celex_filter: set[str] | None, wipe_existing: bool) -> dict[str, int]:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            # Determine the set of regulations to process.
            # Each regulation gets its own independent Louvain run so that
            # communities are regulation-pure.  Cross-regulation CITES edges
            # are excluded from each subgraph — they were the root cause of
            # mixed communities that diluted GPAI and Article 5 slots at
            # retrieval time.
            celexes_to_process: list[str]
            if celex_filter:
                celexes_to_process = sorted(celex_filter)
            else:
                celexes_to_process = _fetch_all_celexes(session)

            if not celexes_to_process:
                logger.warning("No Provision nodes found in the graph.")
                return {"nodes": 0, "edges": 0, "communities": 0}

            if wipe_existing:
                _wipe_existing(session)

            combined_node_to_community: dict[str, str] = {}
            combined_node_meta: dict[str, dict] = {}
            total_nodes = 0
            total_edges = 0

            for celex in celexes_to_process:
                nodes, edges = _fetch_nodes_and_edges(session, {celex})
                if not nodes:
                    logger.info("No Provision nodes for CELEX %s — skipping.", celex)
                    continue

                graph = _build_graph(nodes, edges)
                # Pass celex as prefix so IDs are globally unique:
                # community::32024R1689::0001, community::32017R0745::0001, etc.
                node_to_community = _detect_communities(graph, seed=seed, celex_prefix=celex)

                for row in nodes:
                    combined_node_meta[row["id"]] = row
                combined_node_to_community.update(node_to_community)
                total_nodes += len(nodes)
                total_edges += len(edges)

                logger.info(
                    "CELEX %s: %d nodes, %d edges → %d communities.",
                    celex, len(nodes), len(edges), len(set(node_to_community.values())),
                )

            if not combined_node_to_community:
                logger.warning("No provisions assigned to any community.")
                return {"nodes": 0, "edges": 0, "communities": 0}

            _persist_assignments(session, combined_node_to_community, combined_node_meta)
            l1_count = _build_level1_communities(session, combined_node_to_community)

            community_count = len(set(combined_node_to_community.values()))
            stats = {
                "nodes": total_nodes,
                "edges": total_edges,
                "communities": community_count,
                "level1_communities": l1_count,
            }
            logger.info(
                "Community build complete: %d nodes, %d edges, %d L0 communities, "
                "%d L1 chapter groups.",
                stats["nodes"],
                stats["edges"],
                stats["communities"],
                stats["level1_communities"],
            )
            return stats
    finally:
        driver.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Provision communities and persist Community metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for community detection (default: 42).",
    )
    parser.add_argument(
        "--celex",
        nargs="*",
        help="Optional CELEX scope (e.g. 32024R1689 32017R0745).",
    )
    parser.add_argument(
        "--wipe-existing",
        action="store_true",
        help="Remove prior community assignments and Community nodes first.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    celex_filter = set(args.celex) if args.celex else None
    stats = build_communities(
        seed=args.seed,
        celex_filter=celex_filter,
        wipe_existing=args.wipe_existing,
    )
    print(
        "Built communities: "
        f"nodes={stats['nodes']} edges={stats['edges']} communities={stats['communities']}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
