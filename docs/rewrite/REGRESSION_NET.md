# Regression net — status & gaps

For a strangler rewrite, the net is the safety line: snapshot current behaviour,
then diff every migration step against it and flip only when new ≥ old. Good
news — **two layers already exist**; this is *extend*, not *build*.

## What exists

| Layer | Script | Measures | Cases | LLM? | Deps |
|---|---|---|---|---|---|
| Retrieval recall | `scripts/eval_retrieval.py` | expected regs + `must_contain_refs` present in retrieved bag; expected route | `eval/golden_set.json` (20) | No (HyDE stubbed) | live Neo4j |
| Answer quality | `scripts/eval_answer_quality.py` | rubric LLM-judge score via `agent.ask` | `eval/quality_set.json` (2) | Yes | Neo4j + Mistral |
| Actor-role retrieval | `scripts/evaluate_actor_role_retrieval.py` | role-traversal coverage | — | — | Neo4j |
| Bisection | `scripts/bisect_quality.py` | quality regression bisect | — | — | — |

`eval/golden_set.json` schema: `id, label, question, expected_route,
expected_celexes, must_contain_refs, notes`.

## Gaps to close before migrating (Phase 0 remainder)

1. **Quality set is too small (2 cases).** Grow `quality_set.json` to cover every
   route × regulation × actor role, plus the known-hard cases: Class IIb SaMD +
   continuous learning, lex specialis (73(10)), in-house Art 5(5), GPAI, GDPR
   controller/processor, cross-reg interplay. Target ~25–40.

2. **No faithfulness / attribution metric in the net.** The
   fabricated / misattributed / near-verbatim counts (`check_faithfulness`) are
   not measured by either harness. Add them as first-class regression metrics —
   a migration that increases fabricated or misattributed quotes must fail the
   net even if the rubric score holds.

3. **No frozen baseline / diff mode.** `eval_retrieval.py` checks a *fixed*
   `must_contain_refs` expectation, not the *full retrieved set* vs a snapshot.
   Add: dump each case's retrieved provision IDs + answer to a baseline file on
   `main`, and a `--diff <baseline>` mode that reports added/dropped provisions
   and rubric/faithfulness deltas per case. This is what guards a route-by-route
   strangler migration.

4. **No norm-coverage / exception-recall metric** *(innovation, later).* Because
   the graph defines the obligation cluster for a (role, risk-tier) scenario,
   coverage becomes *checkable*: did the answer surface every `OBLIGATION_OF`
   target, and every `DEROGATES_FROM` exception? Build after contracts land.

## Acceptance bar for the rewrite

A migration step may flip from old→new path only when, on the full golden +
quality sets:

- retrieval recall (`must_contain_refs`) ≥ baseline, **and**
- no case loses an expected route/celex, **and**
- fabricated + misattributed quote counts ≤ baseline, **and**
- mean rubric score ≥ baseline (no single case drops > 1.0).

## Immediate next actions

- [x] Expand `quality_set.json` to the route × reg × role matrix — **32 cases**
      (was 2), tagged with a `tests` dimension per case.
- [x] Add faithfulness/attribution counts to `eval_answer_quality.py` —
      `_parse_faithfulness` reads the agent's flag blocks; per-case +
      aggregate (`fabricated_total` / `misattributed_total` / `near_verbatim_total`).
- [x] Add baseline-snapshot + `--diff` mode to `eval_retrieval.py`
      (`--snapshot PATH` / `--diff PATH`, stable provision-id identity).
- [x] **Capture the pre-Phase-2 baseline** (live Neo4j + Mistral). Captured on
      `read-path-rewrite@HEAD` — *not* literal `main` — so `--diff` isolates the
      Phase-2 retrieval deltas rather than folding in this session's
      faithfulness/scoping changes.
      - Retrieval (done): `python scripts/eval_retrieval.py --snapshot eval/baseline_retrieval.json`
        → 20/20 recall, 20/20 route.
      - Quality: `CRSS_CLARIFY=0 python scripts/eval_answer_quality.py --out eval/baseline_quality.json --quiet`
        (`CRSS_CLARIFY=0` so role-less questions are answered, not deflected to
        a clarification — the ask-first gate is orthogonal to the retrieval
        rewrite and would otherwise void the answer scores).

Every Phase 2+ migration step diffs against these: `eval_retrieval.py --diff
eval/baseline_retrieval.json`, and compare `eval/baseline_quality.json`
fabricated/misattributed totals.
