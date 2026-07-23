# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRSS (Compliance Readiness Support System) is a GraphRAG system for EU regulatory compliance analysis. It ingests EU regulations (MDR, IVDR, AI Act, GDPR) and MDCG guidance PDFs, builds a knowledge graph in Neo4j with cross-references and vector embeddings, and provides an AI agent that answers cross-regulation compliance questions grounded in actual legal text.

## Environment Setup

Python 3.12 is required. The virtualenv is named `crss_mvp`:

```bash
pyenv virtualenv 3.12.9 crss_mvp
pyenv local crss_mvp
pip install -r requirements.txt   # pinned single source of truth for all deps
playwright install chromium
```

Required `.env` at project root:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
MISTRAL_API_KEY=<key>
LLAMA_CLOUD_API_KEY=<key>   # only to ingest/re-parse MDCG guidance PDFs (llama-cloud); not needed for CELEX regs or query-only
```

Optional env vars (can go in `.env` or shell):
```
MISTRAL_MODEL=mistral-large-latest     # user-facing answer + revision model
CRSS_HYDE_MODEL=mistral-small-latest   # HyDE passage (embedded, never shown) — fast by default
CRSS_REWRITE_MODEL=mistral-small-latest # standalone-question rewrite (internal) — fast by default
CRSS_AUDIT_MODEL=mistral-medium-latest # audit verdict (structured) — mid-tier by default
CRSS_CONTEXT_CHAR_BUDGET=140000        # cap on rendered provision context (~35K tokens); trims low-priority tail to bound LLM time-to-first-token
CRSS_LEXICAL=1                         # set to 0 to disable BM25 lexical channel + RRF fusion
CRSS_RERANKER=1                        # set to 0 to disable cross-encoder reranker
CRSS_RERANKER_MODEL=BAAI/bge-reranker-v2-m3  # default reranker; override if needed
CRSS_FAITHFULNESS_CHECK=1              # ON by default: verify every verbatim quote against the
                                       # retrieved corpus (provisions + definitions + guidance);
                                       # unverified quotes are deterministically repaired where
                                       # possible, else redacted + flagged. Set to 0 to disable.
                                       # 2 (strict) adds an LLM repair tier for the residuals the
                                       # deterministic repair can't fix (own-prose-in-quote-marks,
                                       # memory quotes): replace-by-exact-copy / demote / delete,
                                       # each gated on deterministic re-verification — worst case
                                       # identical to mode 1 (application/_faithfulness_repair.py).
CRSS_REPAIR_MODEL=mistral-medium-latest # strict-tier (mode 2) quote-repair model
CRSS_CLARIFY=1                         # ON by default: ask-first scope gate. When an obligation
                                       # question omits the decisive actor role, CRSS asks which
                                       # role before answering instead of silently assuming one.
                                       # Set to 0 to disable (always answer single-shot).
CRSS_GRAPH_EXPANSION=1                 # ON by default. Set to 0 for the graph-ablation baseline:
                                       # the retriever + agent fall back to a flat dense+lexical
                                       # RAG (article body kept, but every graph-reasoning edge —
                                       # CITES/INTERPRETS, reverse cross-reg, role/chain/community
                                       # traversal, curated backbones, corrective pass — removed).
                                       # Only scripts/eval_graph_ablation.py should flip this;
                                       # unset/"1" is byte-for-byte the production path.
```

**Ask-first scope gate** (`application/_scoping.py`, deterministic, no LLM):
actor role is the backbone of every EU compliance answer (and the audit loop's
first check), yet users rarely state it because they do not know it is the
decisive variable. After routing, if an *obligation-focused* question has **no
detected actor role** and at least one role-partitioned regulation in scope,
CRSS emits a `clarify` event (slot + framework-scoped candidate roles + a legal
rationale) and a `done` event carrying the rendered question, then stops — it
does **not** retrieve or generate. The gate stays silent for definition/
provision-lookup/community-overview routes, when a role is already present, or
when no real options exist. The loop closes via the existing standalone-question
rewrite: the user's reply ("I'm the deployer") is folded back into the original
question on the next turn, the role is detected, and the role-aware answer flows
— so chat clients must append the clarification turn to `history`.

Faithfulness verification is deterministic (no LLM call). Its corpus mirrors the
full REGULATORY CONTEXT — provisions, the definitions block, and interpretive
guidance lines. Matching is graduated into three tiers so a trivial rewording is
not treated like a fabrication: **exact** (verbatim substring) → kept silently;
**near-verbatim** (grounded with minor wording differences, e.g. a dropped "the";
high character recall + a long contiguous matching span) → kept with a light
"verify exact wording" note; **absent** (cannot be grounded) → stripped from the
answer with a loud flag. Normalisation folds the hyphen/dash family and
apostrophe variants so orthographic edits ("machine-readable" vs "machine
readable") never cause a false flag.

Grounding alone (does this text exist *somewhere* in context?) is not enough
once the retriever force-loads a whole obligation cluster: the LLM can then dump
a wall of real provision text concatenated under a single citation, and every
fragment verifies individually. Two structural guards run on already-grounded
quotes and **remove** offenders under a distinct **ATTRIBUTION FLAG** (kept
separate from the fabrication flag, since the text is real but displaced):
**concatenation** — a single quote drawing a long contiguous span from more than
two distinct provisions is a dump, not a quotation; **misattribution** — a quote
whose text is absent from the specific provision its nearest `[Article X]` label
cites (parent articles count, e.g. Article 43 grounds an `Article 43(4)` cite).
Misattribution stays silent when the cited provision was never retrieved, so it
cannot adjudicate and never false-flags.

For a full walkthrough of the runtime verify stage — the fixed order of the
citation guards, the faithfulness check's classify→repair→redact flow, the
strict-tier LLM repair, and the pre-repair-confidence rationale — see
[`docs/faithfulness_check.md`](docs/faithfulness_check.md).

Retrieval combines a dense (cosine) channel with a lexical (Neo4j BM25 full-text)
channel fused via Reciprocal Rank Fusion; an optional cross-encoder reranker runs
downstream. The BM25 index is created idempotently on first `GraphRetriever()` init.

**One-time reranker download** (run once, then cached in `~/.cache/huggingface`):
```python
from sentence_transformers import CrossEncoder
CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
```

Neo4j via Docker Compose (`docker-compose.neo4j.yml` at the repo root):
```bash
docker compose -f docker-compose.neo4j.yml up -d
```
The container is named `crss_neo4j` and listens on Bolt `7687` / browser `7474`.
Data persists in `./neo4j/` (gitignored — rebuild via the ingest pipeline, never
from git). The default password is `testpassword`; match it in `.env`'s
`NEO4J_PASSWORD`, and set `NEO4J_URI=bolt://localhost:7687` (Bolt scheme, not the
`7474` browser port).

## Commands

### Run tests
```bash
# All tests
python -m pytest tests/ -q

# Single test file
python -m pytest tests/test_agent_routing.py -q

# Single test by name
python -m pytest tests/test_agent_routing.py::test_select_question_route_prefers_direct_provision_lookup -q
```

### Run the pipeline (full ingest from scratch)

One command runs the whole DAG in the only correct order (preflight → scrape +
parse every catalog doc → load → embed → canonicalize → community summaries).
The doc set is derived from the catalogs, so it can't drift:
```bash
python scripts/build_all.py          # full build (wipes); add --check for preflight only
```
MDCG guidance follows the catalog's `tier` upload-priority — by default only
tier 1 (the curated core, matching the README list) is ingested. Useful flags:
`--mdcg-all` / `--mdcg-tier N` (widen guidance), `--no-mdcg` (regulations only),
`--docs <id...>` (subset), `--no-wipe` (incremental), `--no-summaries`,
`--strict`, `-y` (skip wipe prompt).

To run the stages manually instead (this is exactly what `build_all.py` does),
scrape + parse **each** document first — repeat the first line per CELEX / MDCG
id; the README lists the full set:
```bash
python -m ingestion.run_pipeline --doc 32017R0745 --lang EN   # repeat per doc (MDR, IVDR, AI Act, GDPR, CIR, MDCG_*)
python scripts/load_neo4j.py --wipe
python scripts/embed_provisions.py
python -m canonicalization --cleanup
python scripts/generate_community_summaries.py   # requires MISTRAL_API_KEY
```

### Run the demo server
```bash
python demo/server.py         # http://localhost:5050
python demo/server.py --port 8080
```

### Interactive chat
```bash
python scripts/chat.py        # type `debug` to toggle retrieval trace, `k=N` to change top-k
```

### Smoke test (requires Neo4j + embeddings)
```bash
python scripts/test_agent.py
```

### Answer-quality eval (LLM judge; requires Neo4j + MISTRAL_API_KEY)
```bash
python scripts/eval_answer_quality.py --judge-runs 3 --out quality_vN.json
# Cross-family panel judge (defuses same-model self-preference bias). Providers
# with no SDK/key are skipped; runs Mistral-only until a second key is added.
# openai + google-genai SDKs are installed: set GEMINI_API_KEY (or GOOGLE_API_KEY)
# for the Gemini judge, OPENAI_API_KEY for GPT, or `pip install anthropic` + key
# for Claude. Any ONE cross-family judge already defuses the bias:
python scripts/eval_answer_quality.py \
  --judge-panel "mistral:mistral-large-latest,openai:gpt-4.1" \
  --judge-runs 3 --out quality_vN.json    # or set CRSS_JUDGE_PANEL
# mistral + gpt-4.1 are the two frontier-grade judges (measured Jul 2026: they agree
# within 0.30, so the feared Mistral self-preference is small). Use gpt-4.1 not
# gpt-4o (later cutoff, knows the final AI Act). For a 3rd family use gemini-2.5-pro
# (needs the PAID Gemini tier); gemini-flash-latest is NOT judge-grade — it scored
# ~1.3 low and, via tie-break-to-worse, vetoed clean answers. New Gemini keys 404 on
# pinned gemini-2.5-*/2.0-* ids; only -latest aliases resolve. Sanity-check any new
# judge model with ONE call first — a bad id / unsupported temperature errors every case.
```
The panel medians across the pooled judge calls and prints a per-judge
breakdown; a large spread between the Mistral judge and the cross-family judges
is the self-preference bias made visible. Run role-less single-shot cases with
`CRSS_CLARIFY=0` so the scope gate does not stub them.

Result files are archived under `eval/runs/` automatically: a bare `--out`
filename (e.g. `--out quality_vN.json`) resolves there, while an explicit path
is used as-is. Only the three inputs — `quality_set.json`, `golden_set.json`,
`rubric_prompt.txt` — live directly in `eval/`.

### Graph-ablation eval (isolates the graph's contribution; no LLM judge)
```bash
python scripts/eval_graph_ablation.py --retrieval-only --out ablation_retrieval.json
python scripts/eval_graph_ablation.py --out ablation.json    # answer-level (2× generation/case)
python scripts/eval_graph_ablation.py --case HQ_001 HQ_005 --limit 6
```
Runs each keyed case twice against the same retriever/model — `CRSS_GRAPH_EXPANSION`
on (full GraphRAG) vs off (flat dense+lexical RAG) — and diffs deterministic
`must_cite` recall. **Prefer `--retrieval-only`** (checks the retrieved context,
no generation, runs in minutes): answer-level recall is confounded by the LLM
citing provisions from parametric memory that were never retrieved (measured
Jul 2026: retrieval-level Δ +9pp, answer-level Δ +1.4pp — the gap is flat-RAG
answers citing decisive provisions with no retrieved text behind them).
Forces `CRSS_CLARIFY=0` itself.

### Re-parse without re-scraping (HTML cached)
```bash
python -m ingestion.run_pipeline --doc 32024R1689
python scripts/load_neo4j.py --doc 32024R1689 --wipe
python scripts/embed_provisions.py
python -m canonicalization --cleanup
```

### Force re-scrape from EUR-Lex
```bash
rm data/legislation/<CELEX>/EN/raw/raw.html
python -m ingestion.run_pipeline --doc <CELEX>
```

### Canonicalization options
```bash
python -m canonicalization --dry-run          # preview without writing
python -m canonicalization --cleanup          # remove stale ExternalAct nodes
python -m canonicalization --cleanup --no-communities   # skip Louvain (faster re-runs)
```

## Architecture

### Data flow
```
EUR-Lex HTML / MDCG PDFs
  → ingestion/run_pipeline.py       scrape + parse → data/legislation/<celex>/EN/parsed.json
  → scripts/load_neo4j.py           load JSON into Neo4j (:Provision / :Guidance nodes)
  → scripts/embed_provisions.py     vector embeddings (multilingual-e5-base, 768d, in-memory numpy)
  → canonicalization/               post-load graph enrichment (5 linker stages)
  → scripts/generate_community_summaries.py  LLM summaries for Community nodes
```

### Agent pipeline (`application/`)

The agent is decomposed across private sub-modules, all re-exported from `application/agent.py`:

| Module | Responsibility |
|---|---|
| `_routing.py` | Deterministic question classification into `_QuestionRoute` (no LLM) |
| `_scoping.py` | Ask-first scope gate: detect a missing decisive actor role and clarify before answering (no LLM) |
| `_definitions.py` | Detect defined terms in the question; expand via provision graph |
| `_retrieval.py` | Orchestrate vector + graph retrieval; HyDE query expansion; corrective passes |
| `_context.py` | Assemble structured context string (definitions, provisions, cross-refs) |
| `_prompts.py` | Build system prompt + user message for Mistral |
| `_postprocessing.py` | Safety formatting, uncertainty banners, backbone validation |
| `_faithfulness.py` | Verify verbatim quotes appear in retrieved source text |
| `_confidence.py` | Five-component composite confidence score (no LLM; from retrieval metadata) |
| `_config.py` | Shared constants, regex patterns, regulation name→CELEX mappings |

**Key entry points:** `ask(question, retriever, history)` for single-shot; `ask_stream(...)` for SSE streaming (used by the demo server).

### Retrieval (`retrieval/graph_retriever.py`)

`GraphRetriever` loads all `:Provision` and `:Guidance` embeddings into memory at startup (numpy cosine similarity, no Neo4j vector plugin needed). It exposes:
- `retrieve_hybrid(query, k)` — vector top-k + `HAS_PART` child expansion + `CITES` cross-reference expansion
- `retrieve_by_refs(refs, celex_filter)` — direct lookup by provision ID / display_ref
- `retrieve_by_roles(role_specs, k)` — graph traversal from `ActorRole` nodes via `OBLIGATION_OF` / `INSTANTIATES`

Query prefix: `"query: "` (asymmetric E5 encoding). Passages stored with `"passage: "`.

### Graph schema (`domain/schema/graph_schema.json`)

Node labels:
- `:Provision` — all regulation nodes (Article, Paragraph, Annex, etc.)
- `:Guidance` — MDCG guidance nodes (GuidanceSection, GuidanceParagraph, etc.)
- `:DefinedTerm`, `:ActorRole`, `:Community`, `:ExternalAct`

Key edge types: `HAS_PART` (containment, ordered), `CITES` (internal cross-refs), `INTERPRETS` (guidance→regulation), `DELEGATES_TO` (enacting→annex), `USES_TERM` (provision→DefinedTerm), `MEMBER_OF` (provision→Community), `OBLIGATION_OF`/`INCLUDES_ROLE`/`EQUIVALENT_ROLE` (actor-role edges).

### Canonicalization pipeline (`canonicalization/`)

Seven stages run in order via `python -m canonicalization`:
1. **crosslinker** — resolves `CITES_EXTERNAL` into `CITES`/`INTERPRETS` edges
2. **delegation_linker** — materializes `DELEGATES_TO` from enacting provisions to annexes
3. **term_linker** — materializes `USES_TERM` edges to `DefinedTerm` nodes
4. **role_linker** — creates `ActorRole` nodes and their obligation/role edges
5. **provision_role_classifier** — assigns a `provision_role` (closed taxonomy) to every `:Provision` via deterministic rules
6. **reasoning_linker** — loads the curated legal-reasoning edges (`TRIGGERS_OBLIGATION_CLUSTER`, `IS_PREREQUISITE_FOR`, `REQUIRES_PRIOR_CHECK`, `DEROGATES_FROM`) and `OBLIGATION_OF` patches from `domain/ontology/legal_reasoning_chains.py`. Wraps `scripts/load_legal_reasoning_chains.py` (still runnable standalone). **Without this stage the retriever's reasoning-chain traversals return nothing**, so it is part of the default pipeline.
7. **community_linker** — Louvain community detection; creates `Community` nodes + `MEMBER_OF` edges

Stages 5 and 6 read the flattened `text_for_analysis` body (prefix stripped via `strip_context_prefix`), not the heading-only `p.text`, so article-level obligations classify and link correctly.

### Adding a new regulation

1. Add an entry to `domain/legislation_catalog.py` (`LEGISLATION` dict) with CELEX, name, type, and optionally `source_celex` for consolidated versions.
2. Run `python -m ingestion.run_pipeline --doc <NEW_CELEX>` to scrape and parse.
3. Run `scripts/load_neo4j.py --doc <NEW_CELEX>` (omit `--wipe` to keep existing data).
4. Run `scripts/embed_provisions.py` and `python -m canonicalization --cleanup`.

### MDR/IVDR consolidated versions

MDR (`32017R0745`) and IVDR (`32017R0746`) use consolidated versions (all amendments applied). The `source_celex` field in `domain/legislation_catalog.py` controls which EUR-Lex URL is scraped. The node IDs always use the canonical CELEX key. To update to a newer consolidation, change `source_celex`, delete the cached `raw.html`, and re-run the pipeline.

## Key conventions

- All `application/` private symbols are prefixed with `_` and re-exported from `application/agent.py` — import from there, not from sub-modules.
- Provision IDs follow the pattern `<CELEX>_<kind>_<ref>` (e.g., `32017R0745_article_2`).
- The LLM model is `mistral-large-latest` by default; override with `MISTRAL_MODEL` env var.
- `data/legislation/` and `data/guidance/` are gitignored — they hold scraped/parsed data.
- Tests in `tests/` are unit tests that mock Neo4j and the LLM; no live services required (except `test_communities_integration.py` and `test_retriever_interprets.py` which need Neo4j).
