"""CLI entrypoint for the canonicalization pipeline."""

from __future__ import annotations

import argparse
import logging

from .crosslinker import crosslink
from .delegation_linker import link_delegations
from .role_linker import link_roles
from .term_linker import link_terms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full canonicalization pipeline: "
            "crosslinker -> delegation_linker -> term_linker -> role_linker."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview all stages without writing to Neo4j.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Pass cleanup through to the crosslinker stage only; removes stale "
            "ExternalAct nodes and resolved CITES_EXTERNAL edges."
        ),
    )
    return parser


def run_pipeline(*, dry_run: bool = False, cleanup: bool = False) -> dict[str, dict[str, int]]:
    print("\n=== Canonicalization Pipeline ===")
    print(f"  dry_run={dry_run}  cleanup={cleanup}\n")

    print("[1/4] Crosslinking external references...")
    crosslink_summary = crosslink(dry_run=dry_run, cleanup=cleanup)

    print("[2/4] Materializing delegation edges...")
    delegation_summary = link_delegations(dry_run=dry_run)

    print("[3/4] Materializing defined-term usage edges...")
    term_summary = link_terms(dry_run=dry_run)

    print("[4/4] Materializing actor-role awareness edges...")
    role_summary = link_roles(dry_run=dry_run)

    return {
        "crosslinker": crosslink_summary,
        "delegation_linker": delegation_summary,
        "term_linker": term_summary,
        "role_linker": role_summary,
    }


def main() -> dict[str, dict[str, int]]:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(dry_run=args.dry_run, cleanup=args.cleanup)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
