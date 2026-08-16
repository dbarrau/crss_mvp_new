# CRSS evaluation strategy — a cost-aware pyramid

CRSS's evals exist to expose **systemic** failure modes (retrieval misses, dropped
corpus text, fabricated/misattributed quotes, audit misgrounding, jurisdiction /
draft-numbering leakage, a net-negative revision loop) — cheaply and repeatably.
LLM-judge calls are the expensive and *noisiest* signal, so the strategy inverts
the naïve spend: **a broad free deterministic base, a thin paid judge tip.**

## The cost map

Only two evals actually burn API money — both are LLM-judge:

| Eval | LLM cost | Systemic flaw it exposes |
|---|---|---|
| `eval_answer_quality.py` (panel judge) | **High** — cases × judge-runs × panel-size judge calls **+** 1 generation/case | Holistic answer quality / reliability |
| `bisect_quality.py` | **High** — wraps the above, repeatedly | Regression localisation |
| `eval_graph_ablation.py` (answer-level) | Medium — **2× generation**, no judge | Graph's cite-recall contribution |
| `eval_graph_ablation.py --retrieval-only` | **Zero** | Same, unconfounded by generation |
| `eval_revision_delta.py` | **Zero** (judge overlay opt-in) | Does the audit/revision loop help or hurt? |
| `check_answer_keys.py` | **Zero** — grades a *results file* | Objective correctness (must_cite / must_state) |
| `eval_audit_grounding.py` | **Zero** | Audit gap-fill misgrounding (wrong def number) |
| `verify_completeness.py` | **Zero** | Corpus husks / dropped text |
| `eval_retrieval.py`, role/authority audits | **Zero** | Retrieval recall, graph integrity |

Two levers fall out of this table:

1. **`check_answer(answer, key)`** (`scripts/check_answer_keys.py`) is a fully
   deterministic, law-grounded grader (`must_cite` / `must_state` / `must_not_claim`,
   keys inline in `eval/quality_set.json`). It is reusable anywhere, for free.
2. The judge is both the **most expensive** *and* the **noisiest** signal (it has
   rewarded fabrication; the run-to-run noise floor is ~1.0/case). Most systemic
   flaws never need it.

## The three tiers

### Tier 0 — free deterministic gates · run on every change / in CI
`pytest tests/` · `verify_completeness.py` · `eval_audit_grounding.py` ·
`eval_graph_ablation.py --retrieval-only` · role/authority audits.
These catch retrieval misses, corpus husks, grounding-guard regressions and graph
integrity — **gate merges on them.** Zero API cost.

### Tier 1 — generation-only, no judge · run per feature
**Generate once, grade many.** `scripts/generate_eval_artifact.py` runs each case
through the pipeline **exactly once** and captures a rich artifact; every cheap
grader then reads that one artifact instead of regenerating:

```
python scripts/generate_eval_artifact.py --out artifact_vN.json          # one generation/case
python scripts/eval_revision_delta.py    --artifact artifact_vN.json      # draft vs final (free)
python scripts/check_answer_keys.py      --results  artifact_vN.json      # law-grounded correctness (free)
```

The artifact envelope is `{"meta": …, "results": [{"id", "answer", …}]}` — the
same shape `check_answer_keys` consumes, with `answer` = the final answer, so it
also feeds the judge's `answer_override` path with **no** regeneration.

### Tier 2 — LLM judge · milestone-only
`eval_answer_quality.py` full cross-family panel before a release; a single cheap
judge for routine spot-checks; `bisect_quality.py` **only after** a Tier-0/1
signal has already confirmed a regression. Reuse the Tier-1 artifact via
`answer_override` so the judge never pays for generation twice.

## Failure-mode → tier map

| Systemic failure mode | Caught by | Tier |
|---|---|---|
| Retrieval misses a decisive provision | `eval_graph_ablation --retrieval-only`, `eval_retrieval` | 0 |
| Corpus text dropped (amendment husks, headings) | `verify_completeness` | 0 |
| Audit gap-fill grounds on the wrong definition | `eval_audit_grounding` | 0 |
| Fabricated / misattributed quotes | faithfulness fab-count (in the artifact) | 1 |
| Revision loop is net-negative | `eval_revision_delta` (Δfab, Δcite, Δstate) | 1 |
| Objective correctness (cites, timelines, thresholds) | `check_answer_keys` | 1 |
| Holistic legal-reasoning quality (the residual) | `eval_answer_quality` panel | 2 |

## The draft-vs-final capture (how Tier 1 sees two answers for one generation)

`ask_stream(..., capture=dict)` is an **opt-in** hook (default `None` → the
production path is byte-for-byte unchanged). When a dict is passed it records the
pre-audit **draft** and post-audit **final**, each finalised through the
*identical* deterministic tail (pointer-resolution → verification → bolding →
post-processing), so the only difference between them is whether the
audit/revision loop ran. It also records each side's faithfulness fabrication
counts (`unverified` + `misattributed`), confidence and retrieved provisions.
That is what lets `eval_revision_delta.py` measure the second pass's contribution
— **including whether it adds fabrication** (the failure the loop was caught doing:
2 → 11 quotes) — for the cost of a single generation and zero judge calls.

## Cadence, in one line

Run **Tier 0 always**, **Tier 1 when you touch generation/retrieval/the audit
loop**, and **Tier 2 only at milestones or to confirm a regression Tier 0/1
already flagged.** Never let the judge pay for a generation a Tier-1 artifact
already holds.
