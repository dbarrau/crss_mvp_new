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
- **Phase 1 — contracts** *(in progress)*
  - [x] Define typed `Provision` / `Definition` / `Scenario` / `Evidence`
        (`application/contracts.py`) — additive, zero behaviour change.
        `Provision`/`Definition` are lossless typed views over the canonical
        dict; equivalence tests pin `text_payload()` to the existing
        `_faithfulness` helpers so adoption cannot drift.
  - [x] Migrate `_faithfulness` corpus building onto `Provision.text_payload()`
        — `_provision_text` / `_definition_text` now delegate to the contract
        (its first real consumer), deleting the duplicate text-payload impl. The
        contract is the single source of truth; behaviour-neutral (37
        faithfulness tests + retrieval net stay green), equivalence tests
        strengthened to pin the literal payload.
  - [x] Thread a `Scenario` through `ask_stream`'s detection stage — the
        detection locals (mentioned regs, target celexes, role specs, explicit
        refs, route id, definition flag) are captured into one typed `Scenario`
        right after routing; the ask-first scope gate (`assess_scope`) is its
        first consumer, now reading `scenario.has_role` / `.target_celexes` etc.
        Behaviour-neutral (`assess_scope` treats empty/None celexes identically;
        scoping + full suite green). Loose locals remain for the not-yet-migrated
        plan/retrieve stages. (Then `verify_answer` can take a typed `Evidence`
        instead of provisions / definitions dicts — deferred until the pipeline
        threads `Evidence` end-to-end, to avoid wrap-then-unwrap churn.)
- **Phase 2 — retrieval core + expanders** *(in progress)* — graph-derive the
  obligation set, then delete the hardcodes it duplicated. Landed:
  - [x] **A1** — folded `_retrieve_route_provisions`'s `if route.id == …` ladder
        into named idempotent expanders + a thin four-phase orchestrator (277 →
        124 LOC, behaviour-neutral); extended the retrieval net to drive the full
        sufficiency + corrective pipeline (`86e4aa2`) so the corrective pass is
        gateable; then folded the corrective pass (A4) to re-run those expanders
        via one channel-aware `_recover`, collapsing the duplicate missing-CELEX
        branches (`9e09e0c`, `8652328`, behaviour-exact). The rigid route→expander
        *table* (A1.2) was intentionally not built — the phase structure already
        is the "thin policy" end state.
  - [x] **A5** — assessed and **kept**: `_gap_retrieve` already drives retrieval
        from the auditor's seeds via the retriever's *public* primitives (not a
        private path), with audit-specific dedup/budget/tag that don't belong in
        the general expanders; folding it would be net-additive over-engineering
        and isn't deterministically gateable (auditor is an LLM). Removed only a
        dead `_audit_gap` tag.
  - [x] **A2** — `retrieve_by_roles` relevance-ranks the role's `OBLIGATION_OF`
        set (article-preferred, celex-scoped, guaranteed cap) instead of an
        arbitrary first-k; celex filter pushed into the Cypher before its LIMIT
        so cross-reg equivalence never crowds out in-scope obligations
        (`d94c01f`, `cceae41`). TC_011 passes via traversal; retrieval net 20/20.
  - [x] **AR Article 11** graph gap closed (`role_linker` `_title_is_role_named`,
        `6a248f1`) + edges materialised — last B1 gap.
  - [x] **Community route at A2 parity** — derives the role obligation set via
        the same traversal (`b5e98c6`), unblocking deletion.
  - [x] **Deleted** the GPAI safety net + sub-question and the **B1 obligation
        backbone** end to end (table, injection, anchor render, C3 self-check,
        prompt discipline) — fully graph-derived now (`cd272de`, `3d44bae`,
        −211 LOC).
  - [x] **B2 / A3** — `_AI_ACT_HIGH_RISK_BACKBONE_REFS` derived from the graph:
        completed Art 6's `TRIGGERS_OBLIGATION_CLUSTER` (Art 17-21 + Annex
        IV/VI/VII) so `retrieve_by_chain` reproduces the full list; hardcode
        deleted (`8213f88`). Retrieval net 20/20.
  - [x] **A6** — assessed and **kept**: `_ANCHOR_DEFINITION_TERMS` is genuine
        curation (foundational subject-matter definition per reg), not
        edge-derivable (IVDR's is point (2), not (1)). Fold into a
        `DefinitionExpander` in Phase 3, not a deletion.
  - Remaining hardcodes are KEEP: B3 `_GATE_ARTICLES`, B4
    `_IMPLICIT_PROVISION_REFS` (small seed/lexical config).
- **Phase 3 — agent spine**: fold detection into `scenario.py`; route the audit
  gap-fill and corrective pass through `RetrievalPlan`.
  - [x] **C1/C2/C4/C5 → `verify.py`** — the three scattered post-generation
        blocks in `ask_stream` (citation-scope, faithfulness/attribution,
        confidence) folded into one `verify_answer(...) -> VerificationResult`
        stage; behaviour-neutral relocation (underlying scorers untouched, their
        37 tests pass) + 8 new deterministic `test_verify.py` cases closing the
        post-gen orchestration's zero-coverage gap. C3 was already deleted with
        B1. C6 (ask-first gate) stays for `scenario.py` — a pre-retrieval phase,
        not verification. Open follow-up: confidence's faithfulness component is
        a constant 1.0 (recomputed post-redaction); compute-once fix deferred as
        an answer-affecting, user-approved step (see ledger).
- **Phase 4 — delete** subsumed patches, hardcoded tables, and dead env flags.
