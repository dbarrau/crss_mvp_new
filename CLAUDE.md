# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRSS (Compliance Readiness Support System) is a GraphRAG system for EU regulatory compliance analysis. It ingests EU regulations (MDR, IVDR, AI Act, GDPR) and MDCG guidance PDFs, builds a knowledge graph in Neo4j with cross-references and vector embeddings, and provides an AI agent that answers cross-regulation compliance questions grounded in actual legal text.

## Environment Setup

Python 3.12 is required. The virtualenv is named `crss_mvp`:

```bash
pyenv virtualenv 3.12.9 crss_mvp
pyenv local crss_mvp
pip install beautifulsoup4 lxml neo4j python-dotenv mistralai sentence-transformers torch playwright requests flask
playwright install chromium
```

Required `.env` at project root:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
MISTRAL_API_KEY=<key>
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
```

Retrieval combines a dense (cosine) channel with a lexical (Neo4j BM25 full-text)
channel fused via Reciprocal Rank Fusion; an optional cross-encoder reranker runs
downstream. The BM25 index is created idempotently on first `GraphRetriever()` init.

**One-time reranker download** (run once, then cached in `~/.cache/huggingface`):
```python
from sentence_transformers import CrossEncoder
CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
```

Neo4j via Docker:
```bash
docker run -d --name crss-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password -v crss_neo4j_data:/data neo4j:5-community
```

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
```bash
python -m ingestion.run_pipeline --doc 32017R0745 --lang EN
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
