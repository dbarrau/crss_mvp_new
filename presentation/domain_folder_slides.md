# `domain/` — Presentation Slides (infra walkthrough for Raman)

*Audience: Raman (physicist / senior data scientist). Presenter: Diego. Framing kept in plain software-architecture terms — no jargon you'd have to defend. One optional physics analogy is in the notes only.*

*Two slides: Slide 1 is the thesis and stands alone. Slide 2 is optional — it shows the single-source-of-truth claim is real, and signals engineering discipline (good for getting a senior DS on board). Delivery: content is Keynote-ready; say the word if you'd rather I generate a `.pptx` of these two.*

---

## SLIDE 1 — `domain/`: What CRSS Knows (Knowledge vs. Mechanism)

**Big thesis line:**
> CRSS separates **what it knows about the law** (`domain/` — declarative data) from **how it processes it** (everything else — generic code).

**Two columns:**

| Knowledge — `domain/` (reg-specific data) | Mechanism — everything else (reg-agnostic code) |
|---|---|
| **Catalogs** — which regulations exist, their CELEX identity, consolidated-version handling (MDR · IVDR · AI Act · GDPR + MDCG guidance) | `ingestion/` — parse text → nodes |
| **Schema** — the node & edge types of the graph (`graph_schema.json`) | `canonicalization/` — link nodes into a reasoning graph |
| **Ontology** — the legal knowledge: actor roles (+ cross-regulation equivalences), defined terms, provision-role taxonomy, cross-reference & citation grammar | `retrieval/` — hybrid search over the graph |
| **Curated legal-reasoning chains** — hand-built rules *(our IP)* | `application/` — grounded answer + guards |

**Payoff line (bottom, emphasised):**
> To add a regulation or a new legal rule, we edit **data in `domain/`** — not code. One engine, the whole EU regulatory space.

**Speaker notes (your voice):**
- "The whole system turns on one separation: the knowledge lives in one folder — `domain/`. Everything else is a generic engine that reads it."
- "`domain/` is the *spec*; the four pipeline stages are the *machine* that runs the spec. Add a regulation, add a rule — the machine doesn't change."
- *(optional, only if Raman leans in)* "Same idea as separating a model's parameters from the solver — fixed solver, you swap the parameters."
- "The crown jewel is the curated legal-reasoning chains — hand-built, can't be auto-generated. That's the defensible core."

---

## SLIDE 2 (optional) — `domain/` as the Single Source of Truth

**Header line:**
> One auditable place for the law. Refactor the knowledge freely — the tests pin the behaviour.

**Content:**
- Every curated knowledge table now lives in `domain/`: detection vocabularies, topic→provision maps, citation grammar, role priorities, framework short-names.
- `application/` keeps only *mechanism* — matchers and builders that **import** the data from `domain/`. No legal knowledge is hard-coded in the engine anymore.
- **Why we can trust it:** the knowledge was reorganised in verified behaviour-neutral steps — a deterministic fingerprint over 32 evaluation questions stayed byte-identical before/after each step, and the full 271-test suite stayed green.

**Visual (easy in Keynote):** left = several small boxes scattered inside an `application/` frame; arrow → ; right = those boxes collapsed into one `domain/` box, with thin `import` arrows pointing back from `application/` to `domain/`.

**Speaker notes (your voice):**
- "This matters for audit and trust. When a regulatory expert asks *where does the system get X* — the answer is always one folder."
- "And because the knowledge is cleanly separated, we can keep improving it without fear: we have a deterministic check that proves an answer didn't change."

---

### If you want a tighter single slide
Use **Slide 1** only, and fold one line from Slide 2 into its payoff:
> *…edit data in `domain/`, not code — one auditable place for the law, safe to evolve.*