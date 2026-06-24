# Patch ledger — read path

Every accumulated patch in `retrieval/` + `application/`, mapped to the failure
it fixed, whether it is load-bearing (preserve) or scar tissue (shed), and the
clean home that should own it after the rewrite.

**Disposition legend**

- **DERIVE** — delete the hardcode; the graph already holds this (traverse it).
- **FOLD** — behaviour is real but belongs inside one shared abstraction
  (`RetrievalPlan` / an expander), not a bespoke call site.
- **KEEP** — load-bearing and correct; relocate behind a clean interface, logic
  unchanged.
- **DELETE** — subsumed or redundant once the above land; verify with the net.

Verdict on "scar tissue vs domain knowledge" is **empirical**: remove it, run
the regression net, see if anything regresses. Rows below are the hypothesis.

---

## A. The five overlapping "pull more provisions" mechanisms

These are the heart of the debt: each independently fetches extra provisions and
dedups locally (`_merge_unique_provisions` is called from several sites). They
should collapse into **one** `RetrievalPlan` that selects seeds + an ordered set
of idempotent expanders under one budget and one dedup.

| # | Patch | Location | Failure it fixed | Disposition |
|---|---|---|---|---|
| A1 | `_retrieve_route_provisions` (route-branched retrieval) | `_retrieval.py:370` (~400 LOC of `if route.id == …`) | Different question types need different seeds/expansions | **FOLD** → routes become thin *policies* selecting expanders; the branching dissolves |
| A2 | Obligation-backbone force-retrieval (`_get_obligation_backbone_refs`) | `_retrieval.py:231` + `_OBLIGATION_MASTER_ARTICLES` `_config.py:203` | Role obligation questions missed the statutory anchor articles | **DERIVE** → `RoleExpander` traverses `OBLIGATION_OF` from the role node |
| A3 | AI-Act high-risk backbone (`_AI_ACT_HIGH_RISK_BACKBONE_REFS`) | `_retrieval.py:69` | Class-IIb-SaMD benchmark: obligation cluster not retrieved when `role_specs` empty | **DERIVED & DELETED** (`8213f88`) → completed Article 6's `TRIGGERS_OBLIGATION_CLUSTER` (added Art 17-21 + Annex IV/VI/VII to the curated chain); `retrieve_by_chain` now reproduces the full list, hardcode gone |
| A4 | Corrective retrieval pass (`_run_corrective_retrieval_pass`) | `_retrieval.py:1006` | Selected route retrieved insufficient evidence | **FOLD** → re-run `RetrievalPlan` with added seeds; not a parallel codepath |
| A5 | Audit gap-retrieve (`_gap_retrieve`) | `_audit.py:244` | Auditor names a missing provision/topic to close a backbone gap | **FOLD** → same `RetrievalPlan` re-run; auditor supplies seeds, not a private retriever path |
| A6 | Definition anchor force-injection (`_ANCHOR_DEFINITION_TERMS`) | `_definitions.py:28` | Short foundational terms ("ai system") lost the 15-term cap race → training-memory fallback | **KEEP** (was FOLD) — genuine curation, *not* edge-derivable (see verdict); relocate into a `DefinitionExpander` in Phase 3, don't delete |

> **A6 — empirical verdict (KEEP, not a delete target).** `_ANCHOR_DEFINITION_TERMS`
> is structurally unlike B1/B2: it duplicates no graph *edge*, it designates each
> regulation's foundational subject-matter definition (AI Act→"ai system" Art
> 3(1), MDR→"medical device" Art 2(1), GDPR→"personal data" Art 4(1), IVDR→"in
> vitro diagnostic medical device" Art 2(2)). The obvious derivation — "the first
> definition (point 1) in the definitions article" — reproduces 3 of 4 but
> **breaks on IVDR**, whose point (1) is "medical device" (cross-referencing the
> MDR) while its foundational term is point (2). No clean graph signal marks "the
> subject-matter definition," so deriving it would just relocate the same
> curation behind an IVDR special-case (net-neutral). It is small (4 entries),
> tested, and load-bearing (it implements the cap-race fix). Treat like B4:
> **keep**, and fold into a unified `DefinitionExpander` (priority ordering over
> the cap) as a Phase-3 refactor — not a Phase-2 deletion.

> Once A1–A6 share one plan + one dedup, the "redundant patches working against
> CRSS" the rewrite targets largely disappear. Net check: the
> Class-IIb-SaMD case must retrieve the same obligation cluster via A2/A3's
> graph-derived expanders as the hardcoded lists do today.

> **A2 — empirical proof (TC_011, post-reset baseline).** Question: *"baseline
> obligations for providers of general-purpose AI models"* (GPAI = Art 3(63)).
> The graph is complete: `provider` role `OBLIGATION_OF` → {Art 4, 16, 50, 51,
> **53**, 54, **55**, 88, 94}, and Art 3(63) exists as both Provision and
> DefinedTerm. Yet the captured retrieval bag is {50, 51, 54, 88, 94, Annex XI}
> — **Art 53 (the centerpiece GPAI-provider obligation) and Art 55 (systemic
> risk) are dropped**, and the Art 3(63) definition anchor is absent. The role
> traversal runs (50/51/54 are role obligations) but lets its `OBLIGATION_OF`
> targets compete in the k=8 vector bag, so 53/55 truncate out. This regressed
> from the pre-reset baseline (which had Art 53) purely because the richer
> rebuild — 813→1164 `OBLIGATION_OF` — surfaced more competitors at the k
> boundary. **Conclusion: TC_011 is a retrieval-path truncation, not a graph
> gap.** The A2 RoleExpander (role obligations are *guaranteed seeds*, never
> vector-truncated) + A6 DefinitionExpander (pin the in-scope defined term) fix
> it directly; the old hardcoded backbone was masking it by force-injecting 53.

> **A2 — LANDED (commit d94c01f), but deletion is not yet unlocked.**
> `retrieve_by_roles` now relevance-ranks the role's `OBLIGATION_OF` set
> (article-preferred, celex-scoped, cap 14) instead of taking an arbitrary
> first-k. Retrieval net 19/20→20/20 (TC_011 passes via traversal); on the
> A2-path quality cases fabrication fell −17 with combined fab+mis −2 (the full
> set's apparent +31 misattribution is generation noise — see REGRESSION_NET).
> **However**, attempting to delete the GPAI safety net (force-add Art 53/55)
> regressed **TC_020** — the `community_summary_search` route dropped Art 55.
> A2 only governs the *role* routes (`role_obligations` / `cross_regulation` /
> `legal_qualification` with roles); the community route has no roles, so its
> obligation completeness still rests on the safety net + backbone injection.
> **So A3/the safety net are NOT pure scar tissue — they are load-bearing for
> the community route.** Deleting `_OBLIGATION_MASTER_ARTICLES` /
> `_get_obligation_backbone_refs` / the GPAI safety net is blocked until the
> community route *also* derives its obligation cluster from the graph (its own
> Phase-2 task). The deterministic retrieval net caught this; the deletion was
> reverted.

> **A2/A3/B1 — RESOLVED & DELETED (commits b5e98c6 → 3d44bae).** The blocker
> above was cleared by giving the `community_summary_search` route the *same*
> graph traversal (`retrieve_by_roles`, k=40 for the complete obligation set).
> A bug surfaced and was fixed en route: a dual-role query
> (`[provider, deployer]`) returned only 2 in-scope rows because
> `EQUIVALENT_ROLE` expansion flooded 58 MDR/IVDR obligations and the Cypher's
> `ORDER BY article_id LIMIT 60` truncated the AI-Act rows (`32017…` sorts
> before `32024…`) *before* the Python celex filter ran — fixed by pushing the
> celex filter **into the Cypher, before the LIMIT**. With the community route
> at parity, the following were deleted, each gated on a green 20/20 retrieval
> net:
> - GPAI safety net + `_AI_ACT_GPAI_REFS` (`cceae41`)
> - community GPAI sub-question (`cd272de`)
> - **B1**: `_OBLIGATION_MASTER_ARTICLES`, `_get_obligation_backbone_refs`,
>   `_is_obligation_breadth_question`, the backbone-injection block, the
>   "OBLIGATIONS MASTER LIST" rendering, the **C3 backbone self-check**
>   (`CRSS_BACKBONE_SELFCHECK` flag gone) and its prompt discipline (`3d44bae`,
>   −211 LOC).
>
> **Caveat recorded:** B1 was *not* purely graph-derivable. The table designated
> each role's "obligation overview" article and encoded curated shared-master
> knowledge (GDPR Art 32 for both controller & processor) that has no
> OBLIGATION_OF / title signal. The deliberate trade (user decision) was to drop
> the explicit completeness *anchor + self-check* in favour of A2's ranked
> obligations being present in the main context. Answer-impact validated on the
> 8 `role_obligations` quality cases (the only ones the backbone ever fired on).

---

## B. Hardcoded knowledge that the graph already holds

The strongest signal that canonicalization was under-plugged. Each is a Python
table duplicating graph edges.

| # | Patch | Location | What it really is | Disposition |
|---|---|---|---|---|
| B1 | `_OBLIGATION_MASTER_ARTICLES` `(role,celex)→[articles]` | `_config.py:203` | `OBLIGATION_OF` edge data, hand-transcribed | **DERIVE** from `ActorRole` → `OBLIGATION_OF` |

> **B1 — empirical verdict (re-measured on the post-reset clean graph, commit
> `fcfe1f9`).** `OBLIGATION_OF` traversal reproduces *or exceeds* the hardcode for
> **11 of 13** `(role,celex)` pairs (provider → 9 articles incl. 16+53; MDR
> manufacturer → 26; GDPR controller → 129 provisions, processor → 61). The
> first measurement (stale graph, 813 edges) found 4 gaps; the clean rebuild
> (`role_linker` re-run, 1164 edges) **closed two of them** — the GDPR
> `controller`/`processor` `ActorRole` nodes now exist with full obligation
> edges, confirming they were *stale*, not genuinely absent. **One gap remains**:
> - MDR & IVDR `authorised representative`: node exists but `Article 11` (its
>   core duties) stays unlinked — a real `role_linker` heuristic miss
>   (`_detect_modality` requires a modal in the first sentence; AR Art 11 has a
>   role-named title but no leading modal). The drafted-then-reverted
>   `_title_is_role_named` helper is the fix.
> So B1 is now ~85% scar tissue, ~15% band-aid. The remaining graph-completion
> step (link AR Art 11) is a small, scoped `role_linker` change — do it as a
> deliberate Phase-2 task, then delete the table. (Caveat: GDPR controller→129 /
> processor→61 are paragraph-grained and likely over-linked — a RoleExpander
> precision concern, not a recall gap.)
| B2 | `_AI_ACT_HIGH_RISK_BACKBONE_REFS` flat article list | `_retrieval.py:69` | The high-risk obligation cluster | **DERIVED & DELETED** (`8213f88`) — completed Art 6's `TRIGGERS_OBLIGATION_CLUSTER`, `retrieve_by_chain` reproduces the full list |
| B3 | `_GATE_ARTICLES` (classification-chain gate refs) | `_retrieval.py:481` | Entry seeds for the classification traversal | **DERIVE / KEEP-as-seed-config** — small, may stay as a seed policy if not edge-backed |
| B4 | `_IMPLICIT_PROVISION_REFS` (keyword→canonical ref) | `_config.py:238` | Topic→provision shortcuts (lawful basis→Art 6, etc.) | **KEEP** (relocate) — genuine lexical shortcuts, not graph-derivable; small + tested |

> B1 + B2 are the prize: deleting them and proving the graph traversal
> reproduces the obligation set is the single most validating step of the
> rewrite, and it finally connects `reasoning_linker` / `role_linker` output to
> retrieval. **Both now DONE** (`3d44bae`, `8213f88`). Common pattern: each
> hardcode was scar tissue masking an *incomplete* curated graph — the fix was
> to complete the graph (AR Art 11 via `role_linker`; Art 17-21 + annexes in the
> Art 6 reasoning chain), prove `OBLIGATION_OF` / `TRIGGERS_OBLIGATION_CLUSTER`
> traversal reproduces the list on a green retrieval net, then delete. B3
> (`_GATE_ARTICLES`) and B4 (`_IMPLICIT_PROVISION_REFS`) remain KEEP — small
> seed/lexical config, not edge-derivable.

---

## C. Post-hoc validation self-checks (generation guardrails)

These run *after* generation and are healthy in principle, but each is a
separately bolted-on, env-flagged stage. They belong in one `verify.py` stage
over the typed `Evidence` set.

| # | Patch | Location / flag | Purpose | Disposition |
|---|---|---|---|---|
| C1 | Faithfulness check (fabricated/near) | `_faithfulness.py`, `CRSS_FAITHFULNESS_CHECK` | Redact ungrounded quotes | **KEEP** → into `verify.py` |
| C2 | Attribution guard (concatenation/misattribution) | `_faithfulness.py`, this session | Catch grounded-but-displaced text | **KEEP** → into `verify.py`; first brick of proof-carrying answers |
| C3 | Backbone self-check | `agent.py:803`, `CRSS_BACKBONE_SELFCHECK` | Validate the obligation backbone is present | **KEEP/REVIEW** — may be redundant once backbone is graph-derived (A2/A3) |
| C4 | Citation-scope check | `agent.py:848`, `CRSS_CITATION_SCOPE_CHECK` | Flag cited refs absent from context | **KEEP** → into `verify.py` |
| C5 | Confidence scoring | `_confidence.py` | 5-component composite | **KEEP** → reads `Evidence` provenance directly |
| C6 | Ask-first scope gate | `_scoping.py`, `CRSS_CLARIFY` | Clarify missing decisive actor role | **KEEP** → into `scenario.py` (it is a scenario-completeness check) |

---

## D. Config sprawl

| # | Patch | Location | Disposition |
|---|---|---|---|
| D1 | Per-line context-trim knobs (`CRSS_CHILD_CHARS`, `CRSS_CITE_CHARS`, `CRSS_INTERP_CHARS`, `CRSS_*_LINES`, `CRSS_TRIM_THRESHOLD`) | `_context.py` | **DELETE/CONSOLIDATE** → one context-budget policy on the `Evidence` renderer |
| D2 | Model overrides (`CRSS_HYDE_MODEL`, `CRSS_REWRITE_MODEL`, `CRSS_AUDIT_MODEL`, `CRSS_JUDGE_MODEL`) | various | **KEEP** — legitimate operational config |
| D3 | Channel toggles (`CRSS_LEXICAL`, `CRSS_RERANKER`) | `graph_retriever.py` | **KEEP** — belong on the retrieval core |

---

## Headline targets (highest validation value, do first)

1. **Collapse A1–A6 into one `RetrievalPlan` + one dedup.** Removes the
   redundancy directly named in the rewrite motivation.
2. **DERIVE B1 + B2 from the graph.** Deletes the biggest hardcoded tables and
   connects the under-used canonicalization edges — the "not properly plugged
   in" fix.
3. **Unify C1–C5 into `verify.py`** over a typed `Evidence` object — and make
   the `Provision` dict a real dataclass (the ad-hoc dict shape is what let the
   faithfulness corpus silently drift earlier).
