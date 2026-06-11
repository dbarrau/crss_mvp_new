"""CLI entrypoint for the canonicalization pipeline."""

from __future__ import annotations

import argparse
import logging

from .community_linker import link_communities
from .crosslinker import crosslink
from .delegation_linker import link_delegations
from .provision_role_classifier import classify_provision_roles
from .role_linker import link_roles
from .term_linker import link_terms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full canonicalization pipeline: "
            "crosslinker -> delegation_linker -> term_linker -> role_linker "
            "-> provision_role_classifier -> community_linker."
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
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Skip the community_linker stage (graph partitioning + Community nodes).",
    )
    parser.add_argument(
        "--community-seed",
        type=int,
        default=42,
        help="Deterministic random seed for community detection (default: 42).",
    )
    return parser


def run_pipeline(
    *,
    dry_run: bool = False,
    cleanup: bool = False,
    skip_communities: bool = False,
    community_seed: int = 42,
) -> dict[str, dict[str, int]]:
    print("\n=== Canonicalization Pipeline ===")
    print(f"  dry_run={dry_run}  cleanup={cleanup}  skip_communities={skip_communities}\n")

    print("[1/6] Crosslinking external references...")
    crosslink_summary = crosslink(dry_run=dry_run, cleanup=cleanup)

    print("[2/6] Materializing delegation edges...")
    delegation_summary = link_delegations(dry_run=dry_run)

    print("[3/6] Materializing defined-term usage edges...")
    term_summary = link_terms(dry_run=dry_run)

    print("[4/6] Materializing actor-role awareness edges...")
    role_summary = link_roles(dry_run=dry_run)

    print("[5/6] Classifying provisions by legal role...")
    provision_role_summary = classify_provision_roles(dry_run=dry_run)

    if skip_communities:
        print("[6/6] Community detection skipped (--no-communities).")
        community_summary: dict[str, int] = {"nodes": 0, "edges": 0, "communities": 0}
    else:
        print("[6/6] Building graph communities...")
        community_summary = link_communities(dry_run=dry_run, seed=community_seed)

    return {
        "crosslinker": crosslink_summary,
        "delegation_linker": delegation_summary,
        "term_linker": term_summary,
        "role_linker": role_summary,
        "provision_role_classifier": provision_role_summary,
        "community_linker": community_summary,
    }


def main() -> dict[str, dict[str, int]]:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(
        dry_run=args.dry_run,
        cleanup=args.cleanup,
        skip_communities=args.no_communities,
        community_seed=args.community_seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
