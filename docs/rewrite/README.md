# Read-path rewrite

Working track for re-architecting the **read path** — `retrieval/` and
`application/` (the agent) — behind clean, frozen contracts. The graph build
(`domain/`, `canonicalization/`, `ingestion/`) is a stable asset and is **out of
scope**: the blast radius is the read path only.

## Why

The read path is ~7,500 LOC (`application/` 6,160 + `retrieval/` 1,355) carrying
accumulated patch debt. The core diagnosis (see [PATCH_LEDGER.md](PATCH_LEDGER.md)):

1. **Five overlapping "pull more provisions" mechanisms** grew separately and
   now duplicate each other, each deduping at its own call site with no shared
   retrieval-plan abstraction.
2. **Four hardcoded provision-ref / role→article tables** encode, as Python
   constants, obligation knowledge the graph already holds as `OBLIGATION_OF`
   and reasoning-chain edges. This is the "we built it in canonicalization and
   never plugged it into retrieval" gap.

## North star

Move from *"GraphRAG over legal text"* to *"retrieval-augmented normative
reasoning over a graph of norms"*: the agent derives applicable obligations by
**graph traversal**, never by hardcoded Article lists. The rewrite's clean
contracts (`Scenario`, `Evidence`, `RetrievalPlan`, composable expanders) are
the substrate for the later normative-reasoning layer.

**Pending decision (does not block Phase 0):** which pillar leads the ambitious
phase —
- **Defeasibility / exception reasoning** (most legally distinctive: model
  `DEROGATES_FROM` / `EXCEPTION_TO` and retrieve rules *with* their defeaters), or
- **Proof-carrying answers** (most product-distinctive: answers as graph-checked
  derivations; generalizes the faithfulness/attribution guard).

## Approach: strangler, not big-bang

Rebuild behind contracts incrementally, route by route, with a frozen regression
net as the safety line. Keep the old path live behind a flag; flip only when the
net says new ≥ old.

## Phases

- **Phase 0 — safety net + map** *(in progress)*
  - [x] Patch ledger ([PATCH_LEDGER.md](PATCH_LEDGER.md))
  - [x] Extend the regression net ([REGRESSION_NET.md](REGRESSION_NET.md)): quality
        set 2→32, faithfulness/attribution metrics, retrieval `--snapshot`/`--diff`
  - [ ] Capture the `main` baseline (needs live Neo4j) before Phase 1
- **Phase 1 — contracts**: typed `Provision` / `Evidence` / `Scenario`; adapt
  current code to emit/consume them with zero behaviour change.
- **Phase 2 — retrieval core + expanders** behind `RetrievalPlan`; migrate one
  route at a time, deleting bespoke retrieval as each lands.
- **Phase 3 — agent spine**: fold detection into `scenario.py`; route the audit
  gap-fill and corrective pass through `RetrievalPlan`.
- **Phase 4 — delete** subsumed patches, hardcoded tables, and dead env flags.
