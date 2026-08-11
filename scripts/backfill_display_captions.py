"""Backfill graph-visualisation captions for Neo4j Browser.

Two display-only fixes (nothing in the retrieval/answer pipeline reads these
fields — verified: `title`/`display_caption` are not referenced in application/
or retrieval/):

1. HEADINGS. The EUR-Lex parser captured Chapter/Section headings into `title`
   for the AI Act but dropped them for MDR/IVDR/GDPR (and they are not
   recoverable from the node — the chapter `text` is empty). This restores the
   authoritative headings — extracted from each regulation's cached EUR-Lex
   `raw.html` (`<p class="title-division-2">` under each `cpt_*`/`sct_*` div),
   pinned in `_CONTAINER_HEADINGS` below and cross-checked against the live graph
   (58/58 matched, 0 already populated) on 2026-08-10 — into `title` where empty.
   Source casing is preserved (MDR/IVDR all-caps, GDPR title-case) as it appears
   in the Official Journal. Never overwrites a non-empty `title`.

2. CAPTIONS. Neo4j Browser needs one always-populated caption property.
   `display_caption` is set to:
     - containers  -> `title` (real heading) if present, else `display_ref`
                      ("Chapter II") for anything still without a heading;
     - communities -> the chapter-aligned `label` (level 1), or a synthetic
                      "<reg> cluster <n> (<members>)" (level 0).
   `presentation/crss_graph.grass` captions Chapter/Section/Community by
   `{display_caption}`.

Idempotent. A full re-ingest (`build_all.py`) wipes both → re-run this after.

    python scripts/backfill_display_captions.py            # apply
    python scripts/backfill_display_captions.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Authoritative Chapter/Section headings for the regulations whose parser dropped
# them, keyed by graph node id (= "<celex>_<raw.html anchor>"). Extracted from the
# cached EUR-Lex raw.html and verified against the graph (see module docstring).
_CONTAINER_HEADINGS: dict[str, str] = {
    # --- GDPR 2016/679 (title-case in the OJ) ---
    "32016R0679_cpt_I": "General provisions",
    "32016R0679_cpt_II": "Principles",
    "32016R0679_cpt_III": "Rights of the data subject",
    "32016R0679_cpt_IV": "Controller and processor",
    "32016R0679_cpt_V": "Transfers of personal data to third countries or international organisations",
    "32016R0679_cpt_VI": "Independent supervisory authorities",
    "32016R0679_cpt_VII": "Cooperation and consistency",
    "32016R0679_cpt_VIII": "Remedies, liability and penalties",
    "32016R0679_cpt_IX": "Provisions relating to specific processing situations",
    "32016R0679_cpt_X": "Delegated acts and implementing acts",
    "32016R0679_cpt_XI": "Final provisions",
    "32016R0679_cpt_III.sct_1": "Transparency and modalities",
    "32016R0679_cpt_III.sct_2": "Information and access to personal data",
    "32016R0679_cpt_III.sct_3": "Rectification and erasure",
    "32016R0679_cpt_III.sct_4": "Right to object and automated individual decision-making",
    "32016R0679_cpt_III.sct_5": "Restrictions",
    "32016R0679_cpt_IV.sct_1": "General obligations",
    "32016R0679_cpt_IV.sct_2": "Security of personal data",
    "32016R0679_cpt_IV.sct_3": "Data protection impact assessment and prior consultation",
    "32016R0679_cpt_IV.sct_4": "Data protection officer",
    "32016R0679_cpt_IV.sct_5": "Codes of conduct and certification",
    "32016R0679_cpt_VI.sct_1": "Independent status",
    "32016R0679_cpt_VI.sct_2": "Competence, tasks and powers",
    "32016R0679_cpt_VII.sct_1": "Cooperation",
    "32016R0679_cpt_VII.sct_2": "Consistency",
    "32016R0679_cpt_VII.sct_3": "European data protection board",
    # --- MDR 2017/745 (all-caps in the OJ) ---
    "32017R0745_cpt_I": "SCOPE AND DEFINITIONS",
    "32017R0745_cpt_II": "MAKING AVAILABLE ON THE MARKET AND PUTTING INTO SERVICE OF DEVICES, OBLIGATIONS OF ECONOMIC OPERATORS, REPROCESSING, CE MARKING, FREE MOVEMENT",
    "32017R0745_cpt_III": "IDENTIFICATION AND TRACEABILITY OF DEVICES, REGISTRATION OF DEVICES AND OF ECONOMIC OPERATORS, SUMMARY OF SAFETY AND CLINICAL PERFORMANCE, EUROPEAN DATABASE ON MEDICAL DEVICES",
    "32017R0745_cpt_IV": "NOTIFIED BODIES",
    "32017R0745_cpt_V": "CLASSIFICATION AND CONFORMITY ASSESSMENT",
    "32017R0745_cpt_VI": "CLINICAL EVALUATION AND CLINICAL INVESTIGATIONS",
    "32017R0745_cpt_VII": "POST-MARKET SURVEILLANCE, VIGILANCE AND MARKET SURVEILLANCE",
    "32017R0745_cpt_VIII": "COOPERATION BETWEEN MEMBER STATES, MEDICAL DEVICE COORDINATION GROUP, EXPERT LABORATORIES, EXPERT PANELS AND DEVICE REGISTERS",
    "32017R0745_cpt_IX": "CONFIDENTIALITY, DATA PROTECTION, FUNDING AND PENALTIES",
    "32017R0745_cpt_X": "FINAL PROVISIONS",
    "32017R0745_cpt_V.sct_1": "Classification",
    "32017R0745_cpt_V.sct_2": "Conformity assessment",
    "32017R0745_cpt_VII.sct_1": "Post-market surveillance",
    "32017R0745_cpt_VII.sct_2": "Vigilance",
    "32017R0745_cpt_VII.sct_3": "Market surveillance",
    # --- IVDR 2017/746 (all-caps in the OJ) ---
    "32017R0746_cpt_I": "INTRODUCTORY PROVISIONS",
    "32017R0746_cpt_II": "MAKING AVAILABLE ON THE MARKET AND PUTTING INTO SERVICE OF DEVICES, OBLIGATIONS OF ECONOMIC OPERATORS, CE MARKING, FREE MOVEMENT",
    "32017R0746_cpt_III": "IDENTIFICATION AND TRACEABILITY OF DEVICES, REGISTRATION OF DEVICES AND OF ECONOMIC OPERATORS, SUMMARY OF SAFETY AND CLINICAL PERFORMANCE, EUROPEAN DATABASE ON MEDICAL DEVICES",
    "32017R0746_cpt_IV": "NOTIFIED BODIES",
    "32017R0746_cpt_V": "CLASSIFICATION AND CONFORMITY ASSESSMENT",
    "32017R0746_cpt_VI": "CLINICAL EVIDENCE, PERFORMANCE EVALUATION AND PERFORMANCE STUDIES",
    "32017R0746_cpt_VII": "POST-MARKET SURVEILLANCE, VIGILANCE AND MARKET SURVEILLANCE",
    "32017R0746_cpt_VIII": "COOPERATION BETWEEN MEMBER STATES, MEDICAL DEVICE COORDINATION GROUP, EU REFERENCE LABORATORIES AND DEVICE REGISTERS",
    "32017R0746_cpt_IX": "CONFIDENTIALITY, DATA PROTECTION, FUNDING AND PENALTIES",
    "32017R0746_cpt_X": "FINAL PROVISIONS",
    "32017R0746_cpt_I.sct_1": "Scope and definitions",
    "32017R0746_cpt_I.sct_2": "Regulatory status of products and counselling",
    "32017R0746_cpt_V.sct_1": "Classification",
    "32017R0746_cpt_V.sct_2": "Conformity assessment",
    "32017R0746_cpt_VII.sct_1": "Post-market surveillance",
    "32017R0746_cpt_VII.sct_2": "Vigilance",
    "32017R0746_cpt_VII.sct_3": "Market surveillance",
}

# Labels captioned via display_caption = title-else-ref (Annex* already have title
# in every reg; including them is harmless and future-proofs the caption source).
_CONTAINER_LABELS = [
    "Chapter", "Section",
    "Annex", "AnnexChapter", "AnnexPart", "AnnexSection",
]

_HEADINGS_SET_CYPHER = """
UNWIND $rows AS row
MATCH (n {id: row.id})
WHERE n.title IS NULL OR trim(n.title) = ''
SET n.title = row.title
RETURN count(n) AS set
"""

_HEADINGS_REPORT_CYPHER = """
UNWIND $rows AS row
MATCH (n {id: row.id})
RETURN count(n) AS matched,
       sum(CASE WHEN n.title IS NULL OR trim(n.title) = '' THEN 1 ELSE 0 END) AS empty
"""

_SET_CYPHER = """
MATCH (c:`{label}`)
WITH c, CASE
        WHEN c.title IS NULL OR trim(c.title) = '' THEN c.display_ref
        ELSE c.title
     END AS cap
SET c.display_caption = cap
RETURN count(c) AS n,
       sum(CASE WHEN cap = c.display_ref THEN 1 ELSE 0 END) AS from_ref,
       sum(CASE WHEN cap <> c.display_ref THEN 1 ELSE 0 END) AS from_title
"""

_REPORT_CYPHER = """
MATCH (c:`{label}`)
RETURN count(c) AS n,
       sum(CASE WHEN c.title IS NULL OR trim(c.title) = '' THEN 1 ELSE 0 END) AS missing_title
"""

# Community nodes have no `title`. Level-1 communities carry a chapter-aligned
# `label` ("Chapter III (32016R0679)"); level-0 Louvain clusters have neither, so
# synthesise a legible caption from the regulation + cluster number + member count.
_COMMUNITY_SET_CYPHER = """
MATCH (c:Community)
WITH c, CASE c.regulations[0]
          WHEN '32017R0745' THEN 'MDR'
          WHEN '32017R0746' THEN 'IVDR'
          WHEN '32024R1689' THEN 'AI Act'
          WHEN '32016R0679' THEN 'GDPR'
          WHEN '32026R0977' THEN 'CIR'
          WHEN '32026R1744' THEN 'Omnibus'
          ELSE coalesce(c.regulations[0], 'multi-reg')
        END AS reg
SET c.display_caption = CASE
        WHEN c.label IS NOT NULL AND trim(c.label) <> '' THEN c.label
        ELSE reg + ' cluster ' + split(c.id, '::')[-1]
             + ' (' + toString(c.member_count) + ')'
     END
RETURN count(c) AS n,
       sum(CASE WHEN c.label IS NULL OR trim(c.label) = '' THEN 1 ELSE 0 END) AS synthesised
"""

_COMMUNITY_REPORT_CYPHER = """
MATCH (c:Community)
RETURN count(c) AS n,
       sum(CASE WHEN c.label IS NULL OR trim(c.label) = '' THEN 1 ELSE 0 END) AS missing_label
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    args = ap.parse_args()

    rows = [{"id": k, "title": v} for k, v in _CONTAINER_HEADINGS.items()]

    load_dotenv()
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session() as session:
            # 1. restore authoritative Chapter/Section headings into `title`
            if args.dry_run:
                r = session.run(_HEADINGS_REPORT_CYPHER, rows=rows).single()
                print(f"  headings: {r['matched']}/{len(rows)} ids in graph, "
                      f"{r['empty']} empty title to fill")
            else:
                r = session.run(_HEADINGS_SET_CYPHER, rows=rows).single()
                print(f"  headings: filled title on {r['set']}/{len(rows)} chapter/section nodes")

            # 2. derive display_caption for the container labels
            for label in _CONTAINER_LABELS:
                if args.dry_run:
                    r = session.run(_REPORT_CYPHER.format(label=label)).single()
                    print(f"  {label:14s} nodes={r['n']:>4}  missing title={r['missing_title']:>4}")
                else:
                    r = session.run(_SET_CYPHER.format(label=label)).single()
                    print(f"  {label:14s} set={r['n']:>4}  "
                          f"(from title={r['from_title']:>4}, from ref={r['from_ref']:>4})")

            # 3. community captions
            if args.dry_run:
                r = session.run(_COMMUNITY_REPORT_CYPHER).single()
                print(f"  {'Community':14s} nodes={r['n']:>4}  missing label={r['missing_label']:>4}")
            else:
                r = session.run(_COMMUNITY_SET_CYPHER).single()
                print(f"  {'Community':14s} set={r['n']:>4}  "
                      f"(kept label={r['n'] - r['synthesised']:>4}, synthesised={r['synthesised']:>4})")
        print("dry-run: no changes written." if args.dry_run
              else "headings + display_caption backfilled.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
