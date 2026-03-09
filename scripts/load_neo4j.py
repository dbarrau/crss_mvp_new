#!/usr/bin/env python3
"""
scripts/load_neo4j.py
======================
CLI entry-point: load one or more EU regulation ``parsed.json`` files
into a running Neo4j instance.

Quick start
-----------
# Load both regulations into the default local Neo4j (bolt://localhost:7687):
  python scripts/load_neo4j.py

# Load only the AI Act, wiping previous data first:
  python scripts/load_neo4j.py --celex 32024R1689 --wipe

# Point to a remote instance via environment variables:
  NEO4J_URI=bolt://myserver:7687 NEO4J_PASSWORD=secret python scripts/load_neo4j.py

# Show all options:
  python scripts/load_neo4j.py --help

Neo4j environment variables
----------------------------
  NEO4J_URI        bolt://localhost:7687   (default)
  NEO4J_USER       neo4j                   (default)
  NEO4J_PASSWORD   password                (default)
  NEO4J_DATABASE   neo4j                   (default)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── allow running from the project root without installing the package ────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root before reading os.environ defaults
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from infrastructure.graphdb.neo4j.loader import RegulationGraphLoader, _normalize_neo4j_uri

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── project layout ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "regulations"


def discover_parsed_files(lang: str) -> list[Path]:
    return sorted(DATA_DIR.glob(f"*/{lang}/parsed.json"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="load_neo4j",
        description="Load EU regulation parsed.json files into Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--celex",
        nargs="+",
        metavar="CELEX",
        help=(
            "One or more CELEX IDs to load "
            "(e.g. 32024R1689 32017R0745).  "
            "Omit to load ALL discovered regulations."
        ),
    )
    p.add_argument(
        "--lang",
        default="EN",
        metavar="LANG",
        help="Language sub-directory (default: EN).",
    )
    p.add_argument(
        "--wipe",
        action="store_true",
        help=(
            "Delete existing graph data for the target regulation(s) "
            "before loading.  Use this to force a clean re-import."
        ),
    )
    p.add_argument(
        "--uri",
        default=_normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
        help="Neo4j bolt URI  (env: NEO4J_URI).",
    )
    p.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
        help="Neo4j username  (env: NEO4J_USERNAME).",
    )
    p.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        metavar="PASSWORD",
        help="Neo4j password  (env: NEO4J_PASSWORD).",
    )
    p.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name  (env: NEO4J_DATABASE).",
    )
    return p


def resolve_files(args: argparse.Namespace) -> list[Path]:
    if args.celex:
        files: list[Path] = []
        for celex in args.celex:
            p = DATA_DIR / celex / args.lang / "parsed.json"
            if not p.exists():
                logger.error("File not found: %s", p)
                sys.exit(1)
            files.append(p)
        return files

    files = discover_parsed_files(args.lang)
    if not files:
        logger.error(
            "No parsed.json files found under %s/<celex>/%s/",
            DATA_DIR,
            args.lang,
        )
        sys.exit(1)
    return files


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    files  = resolve_files(args)

    celex_list = [f.parts[-3] for f in files]
    print(f"Regulations to load : {celex_list}")
    print(f"Neo4j               : {args.uri}  db={args.database}")
    print(f"Wipe before load    : {args.wipe}")
    print()

    total_nodes  = 0
    total_rels   = 0
    total_xrefs  = 0

    with RegulationGraphLoader(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
    ) as loader:
        loader.setup_schema()

        for path in files:
            celex = path.parts[-3]
            stats = loader.load_file(path, wipe=args.wipe)
            total_nodes  += stats["nodes"]
            total_rels   += stats["relationships"]
            total_xrefs  += stats.get("cross_references", 0)
            print(
                f"  {celex:<20}  nodes={stats['nodes']:>6}  "
                f"structural_rels={stats['relationships']:>6}  "
                f"cross_refs={stats.get('cross_references', 0):>6}"
            )

    print()
    print(f"Total  nodes              : {total_nodes}")
    print(f"Total  structural rels    : {total_rels}")
    print(f"Total  cross-ref edges    : {total_xrefs}")
    print("Done ✓")


if __name__ == "__main__":
    main()
