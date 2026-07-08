# CRSS Infrastructure — Core-Tech Onboarding

*Audience: strong math/ML background (embeddings, vectors, RAG are assumed known); **no** assumptions about EU regulatory law. Goal: enough mental model + map to start contributing to the graph + retrieval core within a day.*

---

## 0. What CRSS is, in one paragraph
CRSS ingests EU regulations (MDR, IVDR, AI Act, GDPR) and official guidance into a **knowledge graph**, then answers cross-regulation compliance questions by retrieving over that graph and generating answers that are **provably traceable to the source legal text**. The thesis in one line: *generic LLMs hallucinate law; a graph + grounded retrieval + a bounded generator does not.*

## 1. Mental model (in your vocabulary)
Three stages, each a familiar class of problem:

1. **Build a structured state space** — ingestion + canonicalization. Messy "observations" (EUR-Lex HTML, PDFs) → a labelled graph. Nodes = provisions, edges = legal relationships. This is state-space construction / feature engineering: impose the latent structure that's implicit in the text.
2. **Retrieval = search over that space** — a *learned metric* (e5 embeddings) gives semantic nearest-neighbours; *graph edges* give logical/structural neighbours. You need both.
3. **Bounded generation = a constrained operator** — the LLM writes the answer under a hard constraint: every quoted claim must be *conserved* from retrieved source text. Think conservation law: nothing in the output that wasn't in the input. Violations are detected and stripped post-hoc.

### The one idea that justifies the whole architecture
**Why a graph, not just vector RAG?** Vector RAG retrieves *semantically similar* chunks. But law is a *connected logical system*: an obligation in Article 10 triggers requirements in Annex II, which uses a term defined in Article 2, which is constrained by the AI Act. Those provisions are **not** semantically similar — they're *logically linked*. Cosine similarity can't traverse them; explicit graph edges can. That traversal is the moat — it's exactly what an embeddings-only competitor cannot do.

### Translation table
| CRSS term | What you already know |
|---|---|
| Knowledge graph | Labelled network / discrete state space |
| Provision (`:Provision` node) | A node/state (one article, paragraph, point…) |
| `HAS_PART` edge | Containment / tree structure |
| `CITES`, `INTERPRETS`, reasoning-chain edges | Typed relations = the system's "interaction terms" |
| Embedding (`multilingual-e5-base`, 768-d) | Learned metric on semantic space (you know this) |
| Hybrid retrieval + RRF | Rank aggregation / ensemble of two retrievers |
| Cross-encoder reranker | Pairwise relevance model, expensive but accurate |
| Canonicalization | Feature engineering: deriving structure from raw nodes |
| Faithfulness/attribution guard | Conservation constraint on the generator's output |
| Actor role | A symmetry label that selects which obligations apply |

## 2. Domain primer — EU regulation in five minutes
*(the only genuinely unfamiliar part — read this twice)*

- A **Regulation** (e.g. MDR = Regulation (EU) 2017/745) is directly binding EU law. It's hierarchical: **Article → Paragraph → Sub-paragraph → Point (a)(b) → item (i)(ii)**, plus **Annexes** (detailed technical requirements). Every one of these is a "provision" = a node.
- Provisions **cross-reference** each other ("as referred to in Article 47") and reference **other regulations** (MDR → GDPR). These links are not in the text as data — they're prose we have to *extract*.
- **Defined terms**: each regulation defines its own vocabulary ("'manufacturer' means…"). The *same word can mean different things in different regulations* — a core source of compliance error.
- **Actor roles** (manufacturer, importer, deployer, provider…): obligations attach to a role. **The single decisive variable in any compliance answer is "which role are you?"** This is why the agent refuses to answer obligation questions until it knows the role (the "ask-first scope gate").
- **Guidance** (MDCG documents): non-binding interpretation of the binding law. Lower authority than a Regulation — the system tracks this.
- **The business problem**: MDR and the AI Act **overlap** — an AI-enabled medical device must satisfy both at once, and the two regulations don't cross-reference each other cleanly. Doing that mapping by hand is the pain CRSS removes.

**One concrete example.** MDR *Article 10 — "General obligations of manufacturers."* In the PDF it's a heading + a dozen paragraphs. After ingestion + canonicalization it's a node `32017R0745_article_10` with paragraph children, linked **up** to its chapter, **across** to the GDPR/AI-Act provisions it interacts with, attached to the **manufacturer** actor-role, tagged **binding**. The machine now knows what it is, where it sits, and what it touches.

## 3. The four pillars (and where *you* plug in)

**① Ingestion — `ingestion/`** — text → structured nodes. Deterministic, ID-driven parsing of EUR-Lex HTML (a normalization pre-pass folds the two EUR-Lex HTML dialects into one so there's a single parser); LLM-assisted parsing only for guidance PDFs. Output: `parsed.json` → Neo4j → embeddings.
*Key decision:* binding law is parsed deterministically — **never** by an LLM — so the law is never hallucinated at ingest. *Mostly stable; not where you'll spend time.*

**② Canonicalization — `canonicalization/` — YOUR TURF.** Seven ordered stages (`python -m canonicalization`) that turn flat nodes into a *reasoning* graph: resolve cross-references (`crosslinker`), materialize delegation edges (`delegation_linker`), link defined terms (`term_linker`), create actor-role nodes + obligation edges (`role_linker`), classify each provision's role (`provision_role_classifier`), load curated **legal-reasoning chains** (`reasoning_linker`), detect communities via Louvain (`community_linker`).
*Key decision / the IP:* the curated reasoning edges in `domain/ontology/legal_reasoning_chains.py` are **hand-built and can't be auto-generated** — this is the defensible core. The intellectual question "which edges encode real legal logic" lives here.

**③ Retrieval — `retrieval/graph_retriever.py` — YOUR TURF, pure data science.** Hybrid: dense (e5, in-memory numpy cosine — no external vector DB) + lexical (Neo4j BM25 full-text), fused via **Reciprocal Rank Fusion**, then an optional **cross-encoder reranker** (`BAAI/bge-reranker-v2-m3`). Plus graph-aware modes: `retrieve_by_refs`, `retrieve_by_roles` (traverse from actor-role nodes), `retrieve_by_communities_hierarchical`, `retrieve_by_chain`.
*Key decision:* embeddings are loaded into memory at startup; cosine is plain numpy. Retrieval quality is won/lost on the **RRF weights, reranker, and traversal-expansion** knobs — all tunable, all under-optimized.

**④ Bounded agent — `application/`** — orchestration + grounded generation + guards. Deterministic question routing (`_routing`, no LLM), the ask-first scope gate (`_scoping`), retrieval orchestration (`_retrieval`), context assembly (`_context`), prompt build (`_prompts`), then **faithfulness + attribution verification** (`_faithfulness`): every verbatim quote is checked against the retrieved corpus; ungrounded quotes are stripped and flagged, displaced "real but misattributed" quotes get a separate flag. Entry points `ask()` / `ask_stream()` in `agent.py`.
*Key decision:* the "don't hallucinate law" guarantee is enforced **deterministically, after generation** — not trusted to the LLM.

## 4. Code map + conventions
- **`domain/`** — read this *first*. The ontology + schema = the "physics" of the system: which node/edge types exist (`schema/graph_schema.json`), the legal-reasoning chains, actor roles, provision roles, cross-reference patterns.
- `ingestion/` · `canonicalization/` · `retrieval/` · `application/` — the four pillars.
- `infrastructure/` — Neo4j loader, batch embedder.
- `scripts/` — runnable entry points (`build_all.py`, `load_neo4j.py`, `embed_provisions.py`, `chat.py`).
- `tests/` — unit tests mock Neo4j + the LLM, so most run with no live services.
- Conventions: provision IDs are `<CELEX>_<kind>_<ref>` (e.g. `32017R0745_article_10`); `application/` private symbols are `_`-prefixed and re-exported from `agent.py` — import from `agent`, not the sub-modules.

## 5. Get hands-on (fastest path to intuition)
1. Python 3.12 venv `crss_mvp` → `pip install -r requirements.txt`; Neo4j via Docker; `.env` with Neo4j + `MISTRAL_API_KEY`.
2. Full build: `python scripts/build_all.py`.
3. `python scripts/chat.py`, then type `debug` to see the retrieval trace. **Watching the retrieval trace on one real MDR×AI-Act question is the single fastest way to grok the system** — you see the dense hits, the lexical hits, the fusion, the rerank, and the graph expansion in order.
4. First task to build intuition: pick one compliance question, run it with `debug`, and trace *why* each provision was retrieved — then try changing one knob (e.g. toggle `CRSS_RERANKER=0`) and watch the ranking move.

## 6. Open problems (where you add the most value)
- **The eval is a noisy LLM-as-judge** (mean ≈8.5/10 over 32 cases). A rigorous, lower-variance evaluation is open and high-leverage — squarely a data-scientist problem.
- **Retrieval tuning**: RRF weighting, reranker choice/threshold, and how aggressively to expand along graph edges are all under-explored.
- **Canonicalization coverage**: the reasoning chains are hand-curated; how to scale them (semi-automatically, with validation) is the central IP question.
- **Cross-regulation reasoning** (MDR × AI Act) is the hardest *and* most valuable retrieval problem — the overlap the whole product is sold on.

---
*Maintainer notes for keeping this honest: the eval number and retrieval architecture are summarised from `CLAUDE.md` and the code as of this writing; verify against `retrieval/graph_retriever.py` and `eval/` before quoting externally.*