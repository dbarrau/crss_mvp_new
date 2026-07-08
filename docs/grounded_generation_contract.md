# Grounded generation contract (design)

Status: **proposal**. Supersedes the earlier "faithfulness repair loop" framing,
which solved the problem one layer too low.

## The mistake we are correcting

The `[…]` holes in answers were diagnosed (correctly) as **paraphrase-as-
quotation at generation** — not a retrieval gap, not a markdown-matching bug.
Evidence, from tracing "How does MDCG 2019-11 classify standalone software as a
medical device?":

- MDCG 2019-11 **was** retrieved — 20 of 32 nodes, the dominant source; its key
  phrases are literal substrings of the built corpus.
- The normalizer strips balanced `**bold**` mid-quote cleanly; markdown is not
  the cause.
- The model **re-authored** MDCG prose in its own register and wrapped it in `>`,
  so it failed the verbatim/near threshold — and in ≥1 case the reword distorted
  the law ("…is still MDSW" vs the source's "does not have or perform a medical
  purpose on its own").

The first fix attempt was a **repair loop**: let the model type quotes, verify
them, and repair/redact the bad ones. That treats the symptom. The disease is
that **the model is authoring verbatim quotes at all**. If it never authored
them, there would be nothing to verify, nothing to repair, and no `[…]`.

## Principle: take verbatim authorship away from the model

The model emits **claims + pointers**, never verbatim source text. The system
renders quotes deterministically from the pointed source.

- Fabricated quotation becomes **impossible by construction** — the model never
  writes quote text, so it cannot invent it.
- A paraphrase is just prose that makes no claim to be verbatim.
- Enforcement moves from *the model's self-discipline* (a boundary it cannot
  perceive) to *the system's rendering step* (deterministic).

This is **not** the "quote only exact words" prompt exhortation, which fails
because the model cannot tell context from memory. It is a **contract/schema
change**: the model is simply never handed the pen for verbatim text. It follows
the direction `verify.py` already names — "the substrate for proof-carrying
answers."

## The contract

The model produces answer prose in which every intended quotation is a
**pointer**, not text:

- A pointer references a retrieved provision by node ID / ref, optionally with the
  operative phrase the claim leans on (`[cite: MDCG_2019_11_§3.1 | "drives or
  influences the use of a device"]`).
- The renderer resolves each pointer against the retrieved corpus and inserts the
  **verbatim** span (inline for operative words, block for longer passages).
- Non-quoted assertions are plain prose carrying a bare `[cite: …]` — a claim,
  not a quotation.

Rendering owns: inline vs. block vs. partial ("operative words only") quotes,
the `[BINDING]` / `[NON-BINDING GUIDANCE]` caveat attachment, and dropping a
pointer whose target was not retrieved (with an explicit note) rather than
emitting an unsupported quote.

## What this does and does not solve

Solves, completely:

- **Fabricated quotation** — the entire failure class in this thread. Gone by
  construction.

Does **not** solve (so a thin net survives, demoted to last resort):

- **Wrong pointer** — the model points at the wrong node. Strictly better than
  today: structural and checkable ("does the pointed node support this?"), and
  the words themselves are not also fabricated. This is what the old attribution
  check becomes — a pointer-resolution check.
- **Wrong paraphrase in prose** — a false statement with a correct citation, no
  quote involved. The current faithfulness check never caught this anyway (it
  only inspects verbatim quotes). Remains a semantic risk; out of scope here.

## Layering

1. **Primary — generation contract (this doc).** Claims + pointers; deterministic
   quote rendering. Kills fabricated quotation.
2. **Net — verification.** The existing `_faithfulness` / `verify.py` machinery is
   retained only for the residual wrong-pointer / wrong-paraphrase cases, and as
   a guard during migration. It is a safety net, **not** the load-bearing wall.
   The repair loop (deterministic cite-fix, demote-to-paraphrase, one bounded LLM
   repass) is a fallback for pointers that still fail to resolve — not the primary
   mechanism.

## Integration points

- `_prompts.py` — replace "quote verbatim where you can copy exact words" with the
  pointer contract; the model is instructed to emit `[cite: …]` pointers and to
  **never** place source text in quotation marks itself.
- `_context.py` — provisions already carry stable IDs/refs; expose them as the
  pointer vocabulary the model is told to use, and keep the verbatim text keyed by
  that ID for the renderer.
- new render step (in `_postprocessing.py`) — resolve pointers → insert verbatim
  spans (inline/block/partial), attach binding caveats, drop-with-note on
  unresolved pointers. Apply edits right-to-left by offset.
- `verify.py` — verification stays, reframed as the net over the *rendered*
  answer; strict-mode repair becomes the fallback for unresolved pointers.

## The open decision (unchanged): streaming vs. correctness

`ask_stream` streams the draft before rendering/verification can run. With the
pointer contract this is *less* fraught — the streamed draft contains pointers,
not fabricated quotes, so what the user sees mid-stream is never a false
quotation. Options: (A) buffer and render before emit; (B) stream prose live and
resolve pointers to quotes in a final pass; (C) resolve pointers inline as they
stream. (B)/(C) become viable precisely because the un-rendered form is already
safe. Decide when building.

## Pointer syntax (decided)

The pointer **key is the node ID**, never the display ref. Retrieved provisions
carry a stable `article_id` (`32017R0745_art_10`) and each child a stable `id`
(`32017R0745_010.014`); `display_ref` is `None`/non-unique on many nodes — using
it as the key would reinherit the exact citation-ambiguity failure documented
elsewhere. Two pointer forms the model may emit:

- `[cite: <node_id>]` — attach a claim to a provision. Renders to the
  human-readable ref ("Article 10 MDR 2017/745"). No verbatim text.
- `[quote: <node_id>]` — request the verbatim text of that node. The renderer
  substitutes the node's exact stored text (block for a passage). The model never
  types the quoted words.

An unresolved pointer (id not in the retrieved bag) is dropped with a note — never
rendered as an unsupported quote.

## Implementation stages

1. **Resolver (this commit) — deterministic, LLM-free, fully unit-tested.**
   `build_pointer_index(provisions, definitions)` → `{node_id: {text, ref,
   regulation, binding_force}}`; `resolve_pointers(answer, index)` → rewrites
   `[cite:]`/`[quote:]` and reports unresolved ids. This is the inverse of the
   faithfulness check and can be validated with plain dict fixtures, before any
   prompt change.
2. **Context vocabulary.** Render `id: <node_id>` in each `_format_one_provision`
   header so the model can see the key to point at.
3. **Prompt contract.** Replace the "quote verbatim where you can" instruction in
   `_prompts.py` with the pointer contract: emit `[cite:]`/`[quote:]`, never place
   source text in quotation marks.
4. **Wire the render step** into `_postprocessing.py` (resolve before the answer
   is finalised); demote `verify.py` faithfulness to the net over the rendered
   answer.
5. **Streaming decision** (buffer vs. resolve-inline) — deferred; the un-rendered
   stream is already safe (pointers, not fabricated quotes).

## Non-goals

- No retrieval-coverage changes (retrieval was not the problem).
- No `_NEAR_VERBATIM_*` threshold tuning (the thresholds correctly rejected a
  reworded, partly-wrong quote).
- No prompt exhortation about honesty — enforcement is structural, not hortatory.
