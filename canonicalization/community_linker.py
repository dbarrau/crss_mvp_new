"""Materialise graph communities in Neo4j.

Wraps ``scripts.build_communities.build_communities`` so community detection
runs as the **optional** 5th stage of the canonicalization pipeline:

    crosslinker → delegation_linker → term_linker → role_linker → community_linker

This stage is *skipped* (and returns all-zeros) **only** when the
``--no-communities`` flag is passed to the canonicalization CLI (handled
upstream in ``__main__`` — ``link_communities`` is never called in that case).

If the stage *is* requested but its dependency (``networkx``) is missing, it
**raises** rather than skipping silently: a missing dep would otherwise disable
community-level retrieval with no visible error. Install deps via
``pip install -r requirements.txt``, or pass ``--no-communities`` to opt out
deliberately.

Usage::

    python -m canonicalization                   # includes community stage
    python -m canonicalization --no-communities  # skip community stage
    python -m canonicalization --dry-run         # preview, no writes
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def link_communities(
    *,
    dry_run: bool = False,
    seed: int = 42,
    celex_filter: set[str] | None = None,
) -> dict[str, int]:
    """Detect and persist Provision communities.

    Delegates to :func:`scripts.build_communities.build_communities` so all
    detection logic lives in one place.  The ``scripts`` directory must be on
    ``sys.path``; the package root normally ensures this.

    Parameters
    ----------
    dry_run:
        When *True*, compute communities but do not write to Neo4j.
    seed:
        Deterministic random seed for community detection (default: 42).
    celex_filter:
        Optional set of CELEX codes to restrict community building to a
        subset of regulations.  *None* means all loaded Provision nodes.

    Returns
    -------
    dict with keys: ``nodes``, ``edges``, ``communities``
    """
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    try:
        import sys
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.build_communities import build_communities
    except ImportError as exc:
        # Fail loud: this function is only called when community detection is
        # actually wanted (the --no-communities path short-circuits upstream in
        # __main__, never reaching here). A missing dependency must not silently
        # disable community-level retrieval — surface it with a fix and an
        # explicit opt-out.
        raise RuntimeError(
            "community_linker: cannot run community detection — failed to import "
            f"build_communities ({exc}). Its dependency 'networkx' is likely not "
            "installed. Install it with 'pip install -r requirements.txt', or "
            "re-run canonicalization with --no-communities to skip this stage "
            "deliberately."
        ) from exc

    if dry_run:
        logger.info(
            "community_linker: dry_run=True — skipping Neo4j writes. "
            "Pass --no-communities to skip detection entirely."
        )
        return {"nodes": 0, "edges": 0, "communities": 0}

    stats = build_communities(
        seed=seed,
        celex_filter=celex_filter,
        wipe_existing=True,
    )

    logger.info(
        "community_linker: built %d communities across %d nodes (%d edges).",
        stats["communities"],
        stats["nodes"],
        stats["edges"],
    )
    return stats
