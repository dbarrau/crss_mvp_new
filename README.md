# CRSS — Compliance Readiness Support System

A GraphRAG system for EU regulatory compliance analysis. It ingests EU regulations (MDR, IVDR, AI Act, GDPR) from EUR-Lex and MDCG guidance documents from PDF, builds a knowledge graph in Neo4j with cross-references and embeddings, and provides an AI-powered agent that answers cross-regulation compliance questions grounded in the actual legal text.

## Architecture

```
EUR-Lex HTML ──▶ Parse ──▶ parsed.json ──▶ Neo4j Graph ──▶ Embeddings
                                                │
               Canonicalization Pipeline
   (crosslinker → delegation → term → role →
    provision-role → reasoning → community)
                                                │
                                          Retriever (vector + graph)
                                                │
                                          Mistral LLM Agent ──▶ Answer
```

**Supported Regulations:**

| CELEX | Regulation | Source |
|---|---|---|
| `32017R0745` | MDR 2017/745 | Consolidated (`02017R0745-20260101`) |
| `32017R0746` | IVDR 2017/746 | Consolidated (`02017R0746-20250110`) |
| `32024R1689` | EU AI Act 2024/1689 | Legal-basis act |
| `32016R0679` | GDPR 2016/679 | Consolidated (`02016R0679-20160504`) |
| `32026R0977` | CIR (EU) 2026/977 † | Implementing act |

<sub>† The Commission Implementing Regulation is ingested and queryable, but its
actor-role coverage is still partial — some obligations are not yet role-linked.</sub>

**Supported MDCG Guidance Documents:**

| ID | Title | Source |
|---|---|---|
| `MDCG_2019_5` | MDCG 2019-5 — Technical Documentation for medical devices (MDR) | PDF |
| `MDCG_2019_11` | MDCG 2019-11 — Qualification and classification of software (MDR/IVDR) | PDF |
| `MDCG_2020_3` | MDCG 2020-3 Rev.1 — Significant changes under Article 120 MDR | PDF |
| `MDCG_2020_5` | MDCG 2020-5 — Clinical Evaluation: Equivalence | PDF |
| `MDCG_2020_6` | MDCG 2020-6 — Clinical evidence for devices previously CE marked under Directives | PDF |
| `MDCG_2020_13` | MDCG 2020-13 — Clinical Evaluation Assessment Report template | PDF |
| `MDCG_2022_18` | MDCG 2022-18 — Article 97 MDR: legacy devices (position paper) | PDF |
| `MDCG_2023_3` | MDCG 2023-3 Rev.2 — Vigilance terms and concepts (MDR/IVDR) | PDF |
| `MDCG_2025_6` | MDCG 2025-6 — Interplay between MDR/IVDR and the AI Act | PDF |

---

## Setup

Clone the repository and `cd` into it before anything else:

```bash
git clone <repo-url> crss
cd crss
```

The steps below assume you are at the repo root. End-to-end, a fresh setup is:
**Python env → dependencies → Neo4j → `.env` → Mistral key → `build_all.py`.**

### 1. Python Environment

Requires **Python 3.12**. We use pyenv with a virtualenv named `crss_mvp`:

```bash
pyenv virtualenv 3.12.9 crss_mvp
pyenv local crss_mvp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` is the pinned single source of truth for all runtime and
test dependencies. Then install the Playwright browser:

```bash
playwright install chromium
```

### 3. Neo4j

CRSS talks to Neo4j over **Bolt (7687)**; the browser UI is on **7474**. No
plugins required — embeddings live in in-memory numpy, not the Neo4j Vector Index.

**Option A — Docker Compose (bundled, recommended):**

The repo ships [`docker-compose.neo4j.yml`](docker-compose.neo4j.yml), so one
command brings up a correctly-configured container (named `crss_neo4j`):

```bash
docker compose -f docker-compose.neo4j.yml up -d
```

- Default credentials: **`neo4j` / `testpassword`** — use these in your `.env` (next step).
- Data persists in `./neo4j/` at the repo root (gitignored — rebuild it via the
  ingest pipeline, never from git).
- Verify it's up at [http://localhost:7474](http://localhost:7474).

Manage it later:

```bash
docker compose -f docker-compose.neo4j.yml stop    # pause
docker compose -f docker-compose.neo4j.yml start   # resume
docker compose -f docker-compose.neo4j.yml down    # remove the container (keeps ./neo4j data)
rm -rf ./neo4j                                      # wipe the database entirely
```

**Option B — Neo4j Desktop:**

Download from [neo4j.com/download](https://neo4j.com/download/), create a local
DBMS (Bolt on 7687), start it, and use its password in your `.env`.

### 4. Environment Variables

Create a `.env` file at the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password        # use "testpassword" if you started Neo4j via the bundled Compose file
MISTRAL_API_KEY=your_mistral_api_key

# Only required to ingest/re-parse MDCG guidance PDFs (the `MDCG_*` docs).
# Not needed for regulation-only setups or for querying an already-loaded graph.
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key
```

> The loader auto-converts `http://localhost:7474` → `bolt://localhost:7687`, so either URI format works.

**MDCG guidance dependency.** The `MDCG_*` documents are parsed from PDF via
[LlamaParse](https://cloud.llamaindex.ai/) (the `llama-cloud` package, pinned in
`requirements.txt`). Parsing each guidance PDF therefore requires a
`LLAMA_CLOUD_API_KEY` **and** the source PDF — supplied either by a `download_url`
in [`domain/mdcg_catalog.py`](domain/mdcg_catalog.py) or placed manually under
`data/guidance/<DOC_ID>/EN/raw/`. Already-parsed `*_clean.md` artifacts are
cached and reused, so the key is only needed for the first parse (or after you
delete the cache to force a re-parse). The EUR-Lex regulations (CELEX docs) do
**not** use LlamaParse and need no key.

### 5. Mistral API Key

Get one at [console.mistral.ai](https://console.mistral.ai/). The agent uses `mistral-large-latest` by default (configurable via `MISTRAL_MODEL` env var).

---

## Quick Start — Full Pipeline

**One command** runs the entire DAG in the correct order, with a preflight check
that fails fast on missing dependencies, env vars, or an unreachable Neo4j:

```bash
python scripts/build_all.py            # full from-scratch build (wipes)
python scripts/build_all.py --check    # preflight only — verify setup, do no work
```

It derives the document set from the catalogs (so it can't drift). MDCG guidance
follows the catalog's `tier` upload-priority — by default only **tier 1** (the
curated core, i.e. the 9 docs listed below) is ingested; use `--mdcg-all` (every
tier) or `--mdcg-tier N` to widen, `--no-mdcg` to skip guidance entirely. Other
flags: `--docs <id...>`, `--no-wipe`, `--no-summaries`, `--strict`.

<details>
<summary>Or run the stages by hand (what <code>build_all.py</code> orchestrates)</summary>

```bash
# 1. Scrape & parse regulations
python -m ingestion.run_pipeline --doc 32017R0745 --lang EN
python -m ingestion.run_pipeline --doc 32017R0746 --lang EN
python -m ingestion.run_pipeline --doc 32024R1689 --lang EN
python -m ingestion.run_pipeline --doc 32016R0679 --lang EN
python -m ingestion.run_pipeline --doc 32026R0977 --lang EN

# 1b. Parse MDCG guidance documents (PDF)
python -m ingestion.run_pipeline --doc MDCG_2019_5 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2019_11 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2020_3 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2020_5 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2020_6 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2020_13 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2022_18 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2023_3 --lang EN
python -m ingestion.run_pipeline --doc MDCG_2025_6 --lang EN

# 2. Load into Neo4j
python scripts/load_neo4j.py --wipe

# 3. Embed provisions (vector search)
python scripts/embed_provisions.py

# 4. Refresh canonical relationships, semantic edges, and communities
python -m canonicalization --cleanup

# 5. (Optional) Generate LLM summaries for Community nodes
python scripts/generate_community_summaries.py

# 6. Chat
python scripts/chat.py
```

</details>

---

## Step-by-Step Details

### 1. Scrape & Parse — `ingestion/run_pipeline.py`

Scrapes HTML from EUR-Lex (via Playwright headless Chromium) and parses it into `parsed.json`.

```bash
python -m ingestion.run_pipeline --doc <DOC_ID> --lang <LANG>
```

| Arg | Default | Description |
|---|---|---|
| `--doc` | `32017R0745` | Document identifier (CELEX ID or MDCG ID) |
| `--lang` | `EN` | Language code |

**Output:**

```
data/legislation/<celex>/<lang>/
├── raw/raw.html       ← cached; skipped on re-run if present
└── parsed.json        ← provisions + relations + defined_terms
```

If `raw/raw.html` already exists, scraping is skipped. To force a re-scrape, delete the HTML file first.

> **Note:** MDR and IVDR use consolidated versions (current law with all amendments applied). The canonical CELEX key (`32017R0745`, `32017R0746`) stays the same for node IDs, cross-references, and queries. The `source_celex` field in `domain/legislation_catalog.py` controls which EUR-Lex URL is scraped. To update to a newer consolidation, change `source_celex`, delete `raw/raw.html`, and re-run.

### 2. Load into Neo4j — `scripts/load_neo4j.py`

```bash
# Load all discovered parsed.json files
python scripts/load_neo4j.py

# Load specific document(s), wiping first
python scripts/load_neo4j.py --doc 32024R1689 --wipe
python scripts/load_neo4j.py --doc 32024R1689 32017R0745 --wipe
```

| Arg | Default | Description |
|---|---|---|
| `--doc` | *(all)* | One or more document identifiers to load (CELEX ID or MDCG ID) |
| `--lang` | `EN` | Language subdirectory |
| `--wipe` | off | Delete existing graph for the target document(s) before loading |

**What gets created in Neo4j:**

- **Regulation nodes (base label `:Provision`):** `Document`, `Citation`, `Recital`, `Chapter`, `Section`, `Article`, `Paragraph`, `Subparagraph`, `Point`, `Annex`, `AnnexChapter`, `AnnexPart`, `AnnexSection`, `AnnexSubsection`, `AnnexPoint`, `AnnexSubpoint`, `AnnexBullet`
- **Guidance nodes (base label `:Guidance`):** `GuidanceDocument`, `GuidanceSection`, `GuidanceSubsection`, `GuidanceParagraph`, `GuidanceChart`
- **Other nodes:** `DefinedTerm`, `ExternalAct`, `ActorRole`, `Community`
- **Structural edges:** `HAS_PART` (ordered containment)
- **Semantic edges (added by canonicalization):** `CITES` (internal cross-refs), `INTERPRETS` (crosslinker), `CITES_EXTERNAL` / `AMENDS` (external acts), `DEFINED_BY` (term → definition provision), `DELEGATES_TO` (enacting → annex), `USES_TERM` (provision → `DefinedTerm`), `MEMBER_OF` (provision → `Community`), `INSTANTIATES` / `INCLUDES_ROLE` / `OBLIGATION_OF` / `EQUIVALENT_ROLE` (actor-role edges), `TRIGGERS_OBLIGATION_CLUSTER` / `IS_PREREQUISITE_FOR` / `REQUIRES_PRIOR_CHECK` / `DEROGATES_FROM` (curated legal-reasoning edges)
- **Indexes:** Unique constraints on `Provision.id`, `Guidance.id`, `DefinedTerm.id`, `ActorRole.id`, `Community.id`; lookup indexes on `.celex`, `.kind`, `.community_id` for `:Provision`; `.celex`, `.kind` for `:Guidance`; `DefinedTerm.term_normalized`, `DefinedTerm.category`; `ActorRole.term_normalized`, `ActorRole.source_type`, `ActorRole.celex`; `Community.level`

### 3. Embed Provisions — `scripts/embed_provisions.py`

```bash
python scripts/embed_provisions.py
```

No arguments. Reads provisions and guidance nodes from Neo4j, embeds with `intfloat/multilingual-e5-base` (768 dims), writes the `embedding` property back to each `:Provision` / `:Guidance` node.

- Prefix: `"passage: "` (asymmetric E5 encoding)
- Batch: 64 texts per encode call, 500 nodes per Neo4j write

### 4. Canonicalize Relationships — `canonicalization`

Runs the full post-load canonicalization pipeline in a safe execution order:

1. `crosslinker` — resolves `CITES_EXTERNAL` references into concrete `CITES` and `INTERPRETS` edges
2. `delegation_linker` — materializes `DELEGATES_TO` edges from enacting provisions to annex provisions
3. `term_linker` — materializes `USES_TERM` edges from provisions and guidance nodes to `DefinedTerm`
4. `role_linker` — materializes `ActorRole`, `INSTANTIATES`, `INCLUDES_ROLE`, `OBLIGATION_OF`, and `EQUIVALENT_ROLE`
5. `provision_role_classifier` — assigns a `provision_role` (closed taxonomy) to every `:Provision` via deterministic rules
6. `reasoning_linker` — loads the curated legal-reasoning edges (`TRIGGERS_OBLIGATION_CLUSTER`, `IS_PREREQUISITE_FOR`, `REQUIRES_PRIOR_CHECK`, `DEROGATES_FROM`) and `OBLIGATION_OF` patches. **Without this stage the retriever's reasoning-chain traversals return nothing**, so it runs by default.
7. `community_linker` — runs per-regulation Louvain community detection, writes `Community` nodes (Level 0 and Level 1), `MEMBER_OF` edges, and `community_id` on each `:Provision`

```bash
# Preview all stages without writing
python -m canonicalization --dry-run

# Run all stages and clean up stale ExternalAct nodes in the crosslinker stage
python -m canonicalization --cleanup

# Skip community detection (faster re-runs when only legal text changed)
python -m canonicalization --cleanup --no-communities
```

| Flag | Effect |
|---|---|
| `--dry-run` | Preview all canonicalization stages without writing to Neo4j |
| `--cleanup` | Remove stale `ExternalAct` nodes and resolved `CITES_EXTERNAL` edges in the crosslinker stage |
| `--no-communities` | Skip stage 7 (community detection); useful for quick re-runs after text-only changes |
| `--community-seed N` | Deterministic random seed for Louvain community detection (default: 42) |

Run this after loading documents into Neo4j so cross-document relationships and derived semantic edges stay current.

### 5. Generate Community Summaries — `scripts/generate_community_summaries.py`

Generates LLM summaries for each `Community` node and stores a summary embedding alongside the text. Run once after canonicalization; re-run with `--rescan` when community structure is rebuilt.

```bash
python scripts/generate_community_summaries.py
```

| Flag | Effect |
|---|---|
| `--rescan` | Re-generate summaries for communities that already have one |
| `--batch-size N` | Provisions sampled per community for summarization (default: 12) |
| `--dry-run` | Print summaries without writing to Neo4j |

Requires `MISTRAL_API_KEY`. Uses `mistral-large-latest` (configurable via `MISTRAL_MODEL`).

### 6. Interactive Chat — `scripts/chat.py`

```bash
python scripts/chat.py
```

| Command | Effect |
|---|---|
| `quit` | Exit |
| `debug` | Toggle showing retrieved provisions before the answer |
| `k=N` | Change number of retrieved provisions (default 5) |

### 7. Smoke Test

```bash
python scripts/test_agent.py
```

Runs a retriever test + a full agent test. Requires Neo4j (with embeddings) and `MISTRAL_API_KEY`.

---

## Common Operations

### Re-ingest everything from scratch

```bash
python -m ingestion.run_pipeline --doc 32017R0745
python -m ingestion.run_pipeline --doc 32017R0746
python -m ingestion.run_pipeline --doc 32024R1689
python -m ingestion.run_pipeline --doc 32016R0679
python -m ingestion.run_pipeline --doc 32026R0977
python -m ingestion.run_pipeline --doc MDCG_2019_5
python -m ingestion.run_pipeline --doc MDCG_2019_11
python -m ingestion.run_pipeline --doc MDCG_2020_3
python -m ingestion.run_pipeline --doc MDCG_2020_5
python -m ingestion.run_pipeline --doc MDCG_2020_6
python -m ingestion.run_pipeline --doc MDCG_2020_13
python -m ingestion.run_pipeline --doc MDCG_2022_18
python -m ingestion.run_pipeline --doc MDCG_2023_3
python -m ingestion.run_pipeline --doc MDCG_2025_6
python scripts/load_neo4j.py --wipe
python scripts/embed_provisions.py
python -m canonicalization --cleanup
python scripts/generate_community_summaries.py
```

### Re-parse only (HTML already cached)

```bash
python -m ingestion.run_pipeline --doc 32024R1689
python scripts/load_neo4j.py --doc 32024R1689 --wipe
python scripts/embed_provisions.py
python -m canonicalization --cleanup
```

### Force re-scrape from EUR-Lex

```bash
rm data/legislation/32024R1689/EN/raw/raw.html
python -m ingestion.run_pipeline --doc 32024R1689
```

### Validate parsed output

```bash
python scripts/analyze_graphrag.py data/legislation/32024R1689/EN/parsed.json
```

### Verify Neo4j graph integrity

```bash
python scripts/_verify_neo4j.py
```

---

## Project Structure

```
application/          Agent (Mistral LLM + retrieval orchestration)
canonicalization/     Post-load graph enrichment pipeline and individual linkers
data/legislation/     Scraped HTML + parsed JSON (gitignored)
data/guidance/        MDCG guidance PDFs + parsed JSON (gitignored)
domain/               Legislation catalog, MDCG catalog, ontology, graph schema
  ontology/           Cross-reference patterns, defined terms, EUR-Lex HTML patterns
  schema/             Graph schema (JSON)
infrastructure/       Embeddings (sentence-transformers) + Neo4j driver
ingestion/            Full pipeline: scrape → normalize → parse
  parse/              Universal EUR-Lex parser
    structural_layer/ Preamble, enacting terms, annexes, final provisions
    semantic_layer/   Cross-references, definitions, normative modalities
  scrape/             Playwright-based EUR-Lex scraper
retrieval/            GraphRetriever (vector similarity + graph traversal)
scripts/              CLI entry points (load, embed, build communities, generate summaries, chat, test, verify)
```
