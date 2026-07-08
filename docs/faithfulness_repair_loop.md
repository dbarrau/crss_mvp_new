# Faithfulness repair loop (design sketch)

Status: **proposal**. This is the never-built `CRSS_FAITHFULNESS_CHECK=2`
("strict") mode that `_faithfulness.py` and `verify.py` already leave a slot for.

## Problem (root cause, evidence-backed)

The `[…]` holes in answers are **not** a retrieval gap or a markdown-matching
bug. Traced on the canonical case ("How does MDCG 2019-11 classify standalone
software as a medical device?"):

- MDCG 2019-11 **was** retrieved — 20 of 32 nodes, the dominant source; the key
  phrases are literal substrings of the built corpus.
- The normalizer strips balanced `**bold**` mid-quote cleanly; markdown is not
  defeating the match.
- The model **re-authored** MDCG prose in its own register and wrapped it in `>`
  quotation syntax. It therefore fails the verbatim/near threshold
  (`_NEAR_VERBATIM_RECALL=0.90`, `_NEAR_VERBATIM_BLOCK=0.60`). In at least one
  case the rewrite distorted the law ("…is still MDSW" vs the source's "does not
  have or perform a medical purpose on its own").

So the defect is at **generation**: the model cannot tell quoting from
paraphrasing, emits paraphrase-as-quotation, and the pipeline's only recourse
today (`remove_unverified_quotes`) is deletion — which leaves a misleading empty
blockquote. The check is measuring the right thing; the pipeline has no way to
*act* on it except with a scalpel.

## Principle

Verification should **repair**, not just **redact**. A faithful paraphrase and a
fabrication are not the same failure and must not get the same guillotine.
Prefer deterministic repair; use at most one bounded LLM call, and only for the
case that genuinely needs generation.

## Repair strategies, by flag type

The `FaithfulnessReport` already carries typed buckets and each `Quote` has
`start`/`end`/`text`. `_build_sources` yields `source_map` (ref → normalized
text); `_nearest_citation_ref(answer, quote_start)` finds the label a quote sits
under; `_resolve_cited_source` resolves a ref (with parent-chain fallback) to its
text. That is enough to repair most cases deterministically.

### 1. Misattributed (ATTRIBUTION FLAG) — deterministic, no LLM

The text is real and grounded; only the citation is wrong. Locate the source
that *actually* contains the (normalized) quote span by scanning `source_map`
values, and rewrite the nearest `[Article X]` label to the correct ref. If no
single source contains it (concatenation dump), fall through to strategy 3.

- **Cost:** zero LLM. **Risk:** low. **Value:** high — fixes a wrong legal cite
  rather than deleting a correct obligation.

### 2. Fabricated-but-grounded-concept (FAITHFULNESS FLAG, reworded) — deterministic first

This is the MDSW case: the *phrase* is in the corpus but the model's full
sentence is a reword. Repair = **demote the quotation to an attributed
paraphrase**: strip the `>`/quote marks for that span and prefix the claim with
its grounded source ("Per MDCG 2019-11, …"), keeping the model's wording as
prose (a paraphrase does not need to match the corpus). This removes the *false
promise of verbatim fidelity* without erasing the point.

- Optionally verify the demoted claim's cite exists in context; if not, add the
  non-binding / unverified caveat instead of asserting it.

### 3. Genuinely fabricated (no grounding anywhere) — LLM repair or drop

Only here is generation warranted. Batch **all** strategy-3 spans into **one**
bounded, temperature-0 LLM call that receives (a) each offending span and (b) the
retrieved source excerpts, and returns either a grounded verbatim replacement or
an attributed paraphrase — never a new quote it can't ground. Re-verify the
result **once**; anything still `absent` falls back to today's redaction. One
pass guarantees termination and bounded cost.

## Integration points

- `application/verify.py::_apply_faithfulness` — today calls
  `remove_unverified_quotes(answer, report)`. Under `faith_mode == 2`, route
  through a new `repair_quotes(answer, report, provisions, definitions, client)`
  that applies strategies 1→2→3 (deterministic first), re-verifies once, and only
  then falls back to `remove_unverified_quotes` for residual `absent` spans.
- `application/_faithfulness.py` — add `repair_quotes` beside
  `remove_unverified_quotes`; reuse `_build_sources`, `_nearest_citation_ref`,
  `_resolve_cited_source`. Span edits must apply **right-to-left** by `start`
  offset so earlier offsets stay valid (same discipline as the current redactor).
- Confidence still reads the **pre-repair** report, so a repaired answer does not
  get artificially inflated confidence — repair fixes presentation, not the fact
  that the model originally mis-quoted.

## The one hard decision: streaming vs. correctness

`ask_stream` streams the draft to the user **before** `verify_answer` runs. You
cannot repair a span the user has already seen. Options:

- **(A) Buffer the answer** — stream progress steps live, compute the full draft,
  repair, then emit the final text. Correct by construction (a compliance officer
  never sees an unverified quote, even transiently). Costs perceived TTFT — see
  `answer-latency-context-bloat`.
- **(B) Stream draft, then emit a "revised" correction** — keeps TTFT, but shows
  the user a wrong quote first. Unacceptable for a compliance product.
- **(C) Stream prose live, hold only blockquote spans until verified** — best UX,
  most complex; needs the streamer to detect and buffer `>` spans.

Recommendation: **(A)** for correctness now; revisit **(C)** if TTFT regresses
materially. This is the real systemic choice — everything above is mechanical
once it's made.

## Non-goals

- No retrieval-coverage changes (retrieval was not the problem here).
- No threshold tuning of `_NEAR_VERBATIM_*` (the thresholds correctly rejected a
  reworded, partly-wrong quote).
- No new prompt exhortation ("quote only exact words") — the model cannot
  perceive the context/memory boundary, so prose instructions cannot enforce it.