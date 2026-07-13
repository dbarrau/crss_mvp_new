#!/usr/bin/env python3
"""
scripts/build_all.py
====================
One-command orchestrator for the full CRSS build DAG.

Runs every stage in the only correct order, so the pipeline cannot be run
out of sequence or with a step forgotten:

    preflight → scrape+parse (per doc) → load Neo4j → embed → canonicalize
    → community summaries

The document set is derived from the catalogs
(``domain/legislation_catalog.py`` + ``domain/mdcg_catalog.py``) so it can
never drift from a hand-maintained list.

A **preflight** check runs first and fails fast with an actionable report if a
dependency or environment variable is missing, or Neo4j is unreachable — this
is the fix for silent, half-built graphs.

Quick start
-----------
    python scripts/build_all.py                 # full from-scratch build (wipes)
    python scripts/build_all.py --check         # preflight only, no work
    python scripts/build_all.py --no-mdcg        # regulations only
    python scripts/build_all.py --docs 32024R1689 32016R0679   # subset
    python scripts/build_all.py --no-wipe        # incremental (keep existing)
    python scripts/build_all.py --no-summaries   # skip the LLM community summaries

Exit codes: 0 success · 1 a build stage raised · 2 preflight failed.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

# ── allow running from the project root without installing the package ────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv(REPO / ".env", override=False)

from domain.legislation_catalog import LEGISLATION
from domain.mdcg_catalog import MDCG_DOCUMENTS, DEFAULT_INGEST_TIER, default_doc_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_all")

DATA_DIR = REPO / "data" / "legislation"
GUIDANCE_DIR = REPO / "data" / "guidance"

# Third-party imports the build itself relies on, mapped to their pip name for
# actionable error messages.  Checked via find_spec (no heavy import).
_BUILD_IMPORTS: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "neo4j": "neo4j",
    "dotenv": "python-dotenv",
    "mistralai": "mistralai",
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
    "playwright": "playwright",
    "requests": "requests",
    "networkx": "networkx",
    "numpy": "numpy",
    "yaml": "PyYAML",
}
# Only needed when an in-scope MDCG doc must be parsed from PDF.
_MDCG_IMPORTS: dict[str, str] = {"llama_cloud": "llama-cloud"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _neo4j_params() -> tuple[str, str, str, str]:
    from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

    uri = _normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    return uri, user, password, database


def _select_docs(args: argparse.Namespace) -> list[str]:
    """Resolve the ordered document set to build.

    MDCG selection follows the catalog's ``tier`` upload-priority: by default
    only tier <= ``--mdcg-tier`` (1 = curated core, matching the README) is
    ingested; ``--mdcg-all`` pulls in every tier.
    """
    if args.docs:
        return list(args.docs)
    docs = list(LEGISLATION.keys())
    if not args.no_mdcg:
        if args.mdcg_all:
            docs += list(MDCG_DOCUMENTS.keys())
        else:
            docs += default_doc_ids(max_tier=args.mdcg_tier)
    return docs


def _mdcg_needs_parse(doc_id: str, lang: str) -> bool:
    """True if this MDCG doc has no cached clean markdown (so LlamaParse runs)."""
    meta = MDCG_DOCUMENTS.get(doc_id)
    if not meta:
        return False
    stem = Path(meta["pdf_filename"]).stem
    clean_md = GUIDANCE_DIR / doc_id / lang / f"{stem}_clean.md"
    return not clean_md.exists()


# ── preflight ──────────────────────────────────────────────────────────────────

def preflight(docs: list[str], lang: str, *, want_summaries: bool, strict: bool) -> list[str]:
    """Validate deps / env / connectivity. Returns the (possibly trimmed) doc set.

    Raises SystemExit(2) on any hard failure.
    """
    print("\n=== Preflight ===")
    failures: list[str] = []

    # 1. Python dependencies (no heavy import — just availability).
    missing = [pip for mod, pip in _BUILD_IMPORTS.items() if importlib.util.find_spec(mod) is None]
    if missing:
        failures.append(
            f"Missing Python packages: {', '.join(sorted(missing))}. "
            "Install with: pip install -r requirements.txt"
        )
    else:
        print("  [OK]   build dependencies importable")

    # 2. MDCG / LlamaParse: only required if an in-scope MDCG doc must be parsed.
    mdcg_in_scope = [d for d in docs if d in MDCG_DOCUMENTS]
    to_parse = [d for d in mdcg_in_scope if _mdcg_needs_parse(d, lang)]
    if to_parse:
        llama_missing = importlib.util.find_spec("llama_cloud") is None
        key_missing = not os.environ.get("LLAMA_CLOUD_API_KEY")
        if llama_missing or key_missing:
            need = []
            if llama_missing:
                need.append("the 'llama-cloud' package (pip install -r requirements.txt)")
            if key_missing:
                need.append("LLAMA_CLOUD_API_KEY in .env")
            reason = " and ".join(need)
            if strict:
                failures.append(
                    f"{len(to_parse)} MDCG doc(s) need parsing but {reason} is unavailable: "
                    f"{', '.join(to_parse)}"
                )
            else:
                print(
                    f"  [WARN] {len(to_parse)} MDCG doc(s) need {reason}; "
                    f"skipping them: {', '.join(to_parse)}"
                )
                docs = [d for d in docs if d not in to_parse]
        else:
            print(f"  [OK]   LlamaParse ready for {len(to_parse)} MDCG doc(s) needing parse")
    elif mdcg_in_scope:
        print(f"  [OK]   {len(mdcg_in_scope)} MDCG doc(s) already parsed (clean markdown cached)")

    # 3. MISTRAL_API_KEY — needed for the community-summary stage.
    if want_summaries and not os.environ.get("MISTRAL_API_KEY"):
        failures.append(
            "MISTRAL_API_KEY is not set but community summaries are enabled. "
            "Set it in .env, or pass --no-summaries."
        )
    elif want_summaries:
        print("  [OK]   MISTRAL_API_KEY present (community summaries)")

    # 4. Neo4j connectivity.
    uri, user, password, database = _neo4j_params()
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print(f"  [OK]   Neo4j reachable at {uri} (db={database})")
    except Exception as exc:  # noqa: BLE001 — surface any connectivity error
        failures.append(f"Neo4j not reachable at {uri}: {exc}")

    if failures:
        print("\n  PREFLIGHT FAILED:")
        for f in failures:
            print(f"    ✗ {f}")
        raise SystemExit(2)

    print("  Preflight OK.\n")
    return docs


# ── build stages ───────────────────────────────────────────────────────────────

def stage_ingest(docs: list[str], lang: str, *, strict: bool) -> dict[str, bool]:
    from ingestion.run_pipeline import run as run_pipeline

    print(f"=== [1/5] Scrape & parse ({len(docs)} docs) ===")
    results: dict[str, bool] = {}
    for i, doc in enumerate(docs, start=1):
        print(f"  ({i}/{len(docs)}) {doc}")
        try:
            path = run_pipeline(doc, lang)
            ok = path is not None
        except Exception as exc:  # noqa: BLE001 — one bad doc shouldn't kill the build
            logger.exception("  ingest failed for %s: %s", doc, exc)
            ok = False
        results[doc] = ok
        if not ok and strict:
            raise SystemExit(f"--strict: ingest failed for {doc}")
    failed = [d for d, ok in results.items() if not ok]
    if failed:
        logger.warning("Ingest produced no parsed.json for: %s", ", ".join(failed))
    return results


def stage_load(lang: str, *, wipe: bool) -> int:
    from infrastructure.graphdb.neo4j.loader import RegulationGraphLoader

    files = sorted(DATA_DIR.glob(f"*/{lang}/parsed.json"))
    if GUIDANCE_DIR.is_dir():
        files += sorted(GUIDANCE_DIR.glob(f"*/{lang}/parsed.json"))
    if not files:
        raise SystemExit(f"No parsed.json found under data/*/<{lang}>/ — nothing to load.")

    print(f"=== [2/5] Load into Neo4j ({len(files)} files, wipe={wipe}) ===")
    uri, user, password, database = _neo4j_params()
    total_nodes = 0
    with RegulationGraphLoader(uri=uri, user=user, password=password, database=database) as loader:
        loader.setup_schema()
        for path in files:
            doc = path.parts[-3]
            stats = loader.load_file(path, wipe=wipe)
            total_nodes += stats["nodes"]
            print(f"  {doc:<18} nodes={stats['nodes']:>6}  rels={stats['relationships']:>6}")
    print(f"  Loaded {total_nodes} nodes.\n")
    return total_nodes


def stage_embed() -> int:
    from infrastructure.embeddings.batch_embedder import run as embed_run

    print("=== [3/5] Embed provisions ===")
    n = embed_run(celex_filter=None)
    print(f"  Embedded {n} nodes.\n")
    return n


def stage_canonicalize(*, no_communities: bool) -> dict:
    from canonicalization.__main__ import run_pipeline as canon_run

    print("=== [4/5] Canonicalize (cleanup) ===")
    summary = canon_run(cleanup=True, skip_communities=no_communities)
    print("  Canonicalization done.\n")
    return summary


def stage_verify_role_coverage() -> list[str]:
    """Warn loudly when a loaded regulation has no actor-role machinery.

    Every regulation in the catalog is role-partitioned EU law: after
    canonicalization it must carry at least one ActorRole node and at least one
    OBLIGATION_OF edge, or the role-obligation retrieval channel is silently
    empty for that regulation and every role-scoped question degrades to
    vector-only.  This has happened twice in practice (CIR 2026/977 missing
    from ``_REG_PATTERNS``; the same CIR with zero ActorRole nodes), and both
    times nothing failed — the gap only surfaced as quietly worse answers.

    Returns the list of warning strings (empty = healthy) so the build summary
    can repeat them at the end; the build is not failed, because a partial
    graph is still more useful than no graph.
    """
    from neo4j import GraphDatabase

    from domain.legislation_catalog import LEGISLATION

    print("=== Verify role/obligation coverage ===")
    uri, user, password, database = _neo4j_params()
    warnings: list[str] = []
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session(database=database) as session:
            rows = session.run(
                "MATCH (p:Provision) WITH DISTINCT p.celex AS celex "
                "OPTIONAL MATCH (r:ActorRole {celex: celex}) "
                "WITH celex, count(r) AS roles "
                "OPTIONAL MATCH (:Provision {celex: celex})-[o:OBLIGATION_OF]->() "
                "RETURN celex, roles, count(o) AS obligations"
            ).data()
    by_celex = {row["celex"]: row for row in rows}
    for celex, meta in LEGISLATION.items():
        row = by_celex.get(celex)
        if row is None:
            continue  # not loaded in this build (e.g. --docs subset)
        problems = []
        if not row["roles"]:
            problems.append("0 ActorRole nodes")
        if not row["obligations"]:
            problems.append("0 OBLIGATION_OF edges")
        if problems:
            warnings.append(
                f"{meta.get('name', celex)} ({celex}): {' and '.join(problems)} — "
                "role-obligation retrieval is EMPTY for this regulation. "
                "Check role_linker patterns / curated OBLIGATION_OF patches."
            )
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    else:
        print(f"  [OK]   all {len(by_celex)} loaded regulations have roles + obligations")
    print()
    return warnings


def stage_summaries() -> dict:
    from scripts.generate_community_summaries import generate_summaries

    print("=== [5/5] Community summaries ===")
    stats = generate_summaries(rescan=False, batch_size=12, dry_run=False)
    print(f"  Community summaries done: {stats}\n")
    return stats


# ── confirmation ────────────────────────────────────────────────────────────────

def _confirm_wipe(docs: list[str], assume_yes: bool) -> None:
    if assume_yes:
        return
    msg = (
        f"\nThis will WIPE and rebuild graph data for {len(docs)} document(s) in Neo4j.\n"
        "Continue? [y/N] "
    )
    if not sys.stdin.isatty():
        raise SystemExit(
            "Refusing to wipe non-interactively without confirmation. "
            "Re-run with --yes (or --no-wipe)."
        )
    if input(msg).strip().lower() not in {"y", "yes"}:
        raise SystemExit("Aborted.")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build_all", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", default="EN", help="Language sub-directory (default: EN).")
    p.add_argument("--docs", nargs="+", metavar="ID",
                   help="Explicit doc subset (CELEX or MDCG ids). Default: full catalog.")
    p.add_argument("--no-mdcg", action="store_true", help="Skip MDCG guidance documents.")
    p.add_argument("--mdcg-all", action="store_true",
                   help="Ingest every MDCG doc (all tiers), not just the default upload-priority set.")
    p.add_argument("--mdcg-tier", type=int, default=DEFAULT_INGEST_TIER, metavar="N",
                   help=f"Ingest MDCG docs with tier <= N (default: {DEFAULT_INGEST_TIER}).")
    p.add_argument("--no-wipe", action="store_true",
                   help="Incremental load — keep existing graph data (default: wipe).")
    p.add_argument("--no-communities", action="store_true",
                   help="Skip community detection (and summaries).")
    p.add_argument("--no-summaries", action="store_true",
                   help="Skip the LLM community-summary stage.")
    p.add_argument("--check", action="store_true",
                   help="Run preflight only, then exit (no build work).")
    p.add_argument("--strict", action="store_true",
                   help="Abort on the first per-doc ingest failure / missing MDCG dep.")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the wipe confirmation prompt.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    docs = _select_docs(args)
    wipe = not args.no_wipe
    # Summaries depend on communities; --no-communities implies --no-summaries.
    want_summaries = not args.no_summaries and not args.no_communities

    print(f"build_all: {len(docs)} docs · lang={args.lang} · wipe={wipe} · "
          f"communities={not args.no_communities} · summaries={want_summaries}")

    docs = preflight(docs, args.lang, want_summaries=want_summaries, strict=args.strict)
    if args.check:
        print("Preflight only (--check): exiting.")
        return

    if wipe:
        _confirm_wipe(docs, args.yes)

    ingest = stage_ingest(docs, args.lang, strict=args.strict)
    stage_load(args.lang, wipe=wipe)
    stage_embed()
    stage_canonicalize(no_communities=args.no_communities)
    coverage_warnings = stage_verify_role_coverage()
    if want_summaries:
        stage_summaries()

    ok = sum(1 for v in ingest.values() if v)
    failed = [d for d, v in ingest.items() if not v]
    print("=== Build complete ===")
    print(f"  docs ingested: {ok}/{len(ingest)}")
    if failed:
        print(f"  docs FAILED to ingest (not loaded): {', '.join(failed)}")
    for w in coverage_warnings:
        print(f"  [WARN] {w}")
    print("  Graph is ready. Try: python scripts/chat.py")


if __name__ == "__main__":
    main()
