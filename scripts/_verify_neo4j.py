#!/usr/bin/env python3
"""Verify annex hierarchy in Neo4j after loading."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from neo4j import GraphDatabase

d = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "password")),
)

lines = []
with d.session() as s:
    # Label distribution
    r = s.run(
        "MATCH (n) WITH labels(n) AS lbls UNWIND lbls AS lbl "
        "WITH lbl WHERE lbl IN ['AnnexChapter','AnnexSection','AnnexPoint','AnnexSubpoint','AnnexBullet','Annex'] "
        "RETURN lbl, count(*) AS cnt ORDER BY cnt DESC"
    )
    lines.append("=== Annex Label Distribution ===")
    for rec in r:
        lines.append(f"  {rec['lbl']}: {rec['cnt']}")

    # Max depth
    r2 = s.run("MATCH (n:AnnexPoint) RETURN max(n.hierarchy_depth) AS maxD")
    lines.append(f"\nMax annex point depth: {r2.single()['maxD']}")

    # MDR Annex I hierarchy
    r3 = s.run(
        "MATCH (a:Annex {number: 'I'})-[:HAS_CHILD]->(c) "
        "WHERE a.node_id STARTS WITH '32017R0745' "
        "RETURN labels(c) AS lbls, c.number AS num, "
        "left(coalesce(c.title, c.text, ''), 60) AS title "
        "ORDER BY c.hierarchy_depth, c.node_id LIMIT 10"
    )
    lines.append("\n=== MDR Annex I Direct Children ===")
    for rec in r3:
        lines.append(f"  {rec['lbls']} num={rec['num']}: {rec['title']}")

    # Chapter I grandchildren
    r4 = s.run(
        "MATCH (a:Annex {number: 'I'})-[:HAS_CHILD]->(ch:AnnexChapter)-[:HAS_CHILD]->(gc) "
        "WHERE a.node_id STARTS WITH '32017R0745' AND ch.number = 'I' "
        "RETURN labels(gc) AS lbls, gc.number AS num, "
        "left(coalesce(gc.title, gc.text, ''), 60) AS title "
        "ORDER BY gc.node_id LIMIT 10"
    )
    lines.append("\n=== MDR Annex I > Chapter I Children ===")
    for rec in r4:
        lines.append(f"  {rec['lbls']} num={rec['num']}: {rec['title']}")

    # Chapter II > Section 10 grandchildren
    r5 = s.run(
        "MATCH (s:AnnexSection {number: '10'})-[:HAS_CHILD]->(gc) "
        "WHERE s.node_id STARTS WITH '32017R0745' "
        "RETURN labels(gc) AS lbls, gc.number AS num, "
        "left(coalesce(gc.title, gc.text, ''), 50) AS title "
        "ORDER BY gc.node_id LIMIT 10"
    )
    lines.append("\n=== MDR Section 10 Children (depth test) ===")
    for rec in r5:
        lines.append(f"  {rec['lbls']} num={rec['num']}: {rec['title']}")

d.close()

out = "\n".join(lines)
Path("/tmp/crss_neo4j_verify.txt").write_text(out)
print(out)
