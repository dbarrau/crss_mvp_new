# CRSS — Compliance Readiness Support System

A GraphRAG system for EU regulatory compliance analysis. It ingests EU regulations (MDR, IVDR, AI Act) from EUR-Lex, builds a knowledge graph in Neo4j with cross-references and embeddings, and provides an AI-powered agent that answers cross-regulation compliance questions grounded in the actual legal text.

## Architecture

```
EUR-Lex HTML ──▶ Parse ──▶ parsed.json ──▶ Neo4j Graph ──▶ Embeddings
                                                │
                                          Crosslinker
                                          (inter-regulation CITES)
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

---

## Setup

### 1. Python Environment

Requires **Python 3.12**. We use pyenv with a virtualenv named `crss_mvp`:

```bash
pyenv virtualenv 3.12.9 crss_mvp
pyenv local crss_mvp
```

### 2. Install Dependencies

```bash
pip install \
  beautifulsoup4 \
  lxml \
  neo4j \
  python-dotenv \
  mistralai \
  sentence-transformers \
  torch \
  playwright \
  requests
```

Then install the Playwright browser:

```bash
playwright install chromium
```

### 3. Neo4j

Install and run Neo4j locally. Default ports: **Bolt 7687** / **Browser 7474**. No plugins required (embeddings use in-memory numpy, not Neo4j Vector Index).

**Option A — Docker (recommended):**

```bash
docker run -d \
  --name crss-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v crss_neo4j_data:/data \
  neo4j:5-community
```

The `-v crss_neo4j_data:/data` flag persists the database across container restarts. To verify it's running, open [http://localhost:7474](http://localhost:7474) in a browser.

To stop/start later:

```bash
docker stop crss-neo4j
docker start crss-neo4j
```

To wipe and start fresh:

```bash
docker rm -f crss-neo4j
docker volume rm crss_neo4j_data
# Then re-run the docker run command above
```

**Option B — Neo4j Desktop:**

Download from [neo4j.com/download](https://neo4j.com/download/), create a local DBMS, and start it.

### 4. Environment Variables

Create a `.env` file at the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
MISTRAL_API_KEY=your_mistral_api_key
```

> The loader auto-converts `http://localhost:7474` → `bolt://localhost:7687`, so either URI format works.

### 5. Mistral API Key

Get one at [console.mistral.ai](https://console.mistral.ai/). The agent uses `mistral-large-latest` by default (configurable via `MISTRAL_MODEL` env var).

---

## Quick Start — Full Pipeline

```bash
# 1. Scrape & parse regulations
python -m ingestion.run_pipeline --celex 32017R0745 --lang EN
python -m ingestion.run_pipeline --celex 32017R0746 --lang EN
python -m ingestion.run_pipeline --celex 32024R1689 --lang EN

# 2. Load into Neo4j
python scripts/load_neo4j.py --wipe

# 3. Embed provisions (vector search)
python scripts/embed_provisions.py

# 4. Crosslink regulations (resolve inter-regulation references)
python -m canonicalization.crosslinker --cleanup

# 5. Chat
python scripts/chat.py
```

---

## Step-by-Step Details

### 1. Scrape & Parse — `ingestion/run_pipeline.py`

Scrapes HTML from EUR-Lex (via Playwright headless Chromium) and parses it into `parsed.json`.

```bash
python -m ingestion.run_pipeline --celex <CELEX_ID> --lang <LANG>
```

| Arg | Default | Description |
|---|---|---|
| `--celex` | `32017R0745` | CELEX identifier |
| `--lang` | `EN` | Language code |

**Output:**

```
data/regulations/<celex>/<lang>/
├── raw/raw.html       ← cached; skipped on re-run if present
└── parsed.json        ← provisions + relations + defined_terms
```

If `raw/raw.html` already exists, scraping is skipped. To force a re-scrape, delete the HTML file first.

> **Note:** MDR and IVDR use consolidated versions (current law with all amendments applied). The canonical CELEX key (`32017R0745`, `32017R0746`) stays the same for node IDs, cross-references, and queries. The `source_celex` field in `domain/regulations_catalog.py` controls which EUR-Lex URL is scraped. To update to a newer consolidation, change `source_celex`, delete `raw/raw.html`, and re-run.

### 2. Load into Neo4j — `scripts/load_neo4j.py`

```bash
# Load all discovered parsed.json files
python scripts/load_neo4j.py

# Load specific regulation(s), wiping first
python scripts/load_neo4j.py --celex 32024R1689 --wipe
python scripts/load_neo4j.py --celex 32024R1689 32017R0745 --wipe
```

| Arg | Default | Description |
|---|---|---|
| `--celex` | *(all)* | One or more CELEX IDs to load |
| `--lang` | `EN` | Language subdirectory |
| `--wipe` | off | Delete existing graph for target regulation(s) before loading |

**What gets created in Neo4j:**

- **Nodes:** `Document`, `Citation`, `Recital`, `Chapter`, `Section`, `Article`, `Paragraph`, `Subparagraph`, `Point`, `Annex`, `AnnexChapter`, `AnnexSection`, `AnnexPoint`, `AnnexSubpoint`, `AnnexBullet`, `DefinedTerm`
- **Edges:** `HAS_PART` (structural), `CITES` (internal cross-refs), `CITES_EXTERNAL`/`AMENDS` (external acts), `DEFINED_BY` (term → definition provision)
- **Indexes:** Unique constraints on `Provision.id` and `DefinedTerm.id`; lookup indexes on `Provision.celex`, `Provision.kind`, `DefinedTerm.term_normalized`, `DefinedTerm.category`

### 3. Embed Provisions — `scripts/embed_provisions.py`

```bash
python scripts/embed_provisions.py
```

No arguments. Reads provisions from Neo4j, embeds with `intfloat/multilingual-e5-small` (384 dims), writes the `embedding` property back to each `:Provision` node.

- Prefix: `"passage: "` (asymmetric E5 encoding)
- Batch: 64 texts per encode call, 500 nodes per Neo4j write

### 4. Crosslink Regulations — `canonicalization/crosslinker.py`

Resolves `CITES_EXTERNAL` references between loaded regulations into concrete `CITES` edges between existing Provision nodes.

```bash
# Preview without writing
python -m canonicalization.crosslinker --dry-run

# Run and clean up stale ExternalAct nodes
python -m canonicalization.crosslinker --cleanup
```

| Flag | Effect |
|---|---|
| `--dry-run` | Preview matches without writing to Neo4j |
| `--cleanup` | Remove stale `ExternalAct` nodes and resolved `CITES_EXTERNAL` edges |

**Run this after loading all regulations**, so cross-references between MDR, IVDR, and AI Act are resolved.

### 5. Interactive Chat — `scripts/chat.py`

```bash
python scripts/chat.py
```

| Command | Effect |
|---|---|
| `quit` | Exit |
| `debug` | Toggle showing retrieved provisions before the answer |
| `k=N` | Change number of retrieved provisions (default 5) |

### 6. Smoke Test

```bash
python scripts/test_agent.py
```

Runs a retriever test + a full agent test. Requires Neo4j (with embeddings) and `MISTRAL_API_KEY`.

---

## Common Operations

### Re-ingest everything from scratch

```bash
python -m ingestion.run_pipeline --celex 32017R0745
python -m ingestion.run_pipeline --celex 32017R0746
python -m ingestion.run_pipeline --celex 32024R1689
python scripts/load_neo4j.py --wipe
python scripts/embed_provisions.py
python -m canonicalization.crosslinker --cleanup
```

### Re-parse only (HTML already cached)

```bash
python -m ingestion.run_pipeline --celex 32024R1689
python scripts/load_neo4j.py --celex 32024R1689 --wipe
python scripts/embed_provisions.py
python -m canonicalization.crosslinker --cleanup
```

### Force re-scrape from EUR-Lex

```bash
rm data/regulations/32024R1689/EN/raw/raw.html
python -m ingestion.run_pipeline --celex 32024R1689
```

### Validate parsed output

```bash
python scripts/analyze_graphrag.py data/regulations/32024R1689/EN/parsed.json
```

### Verify Neo4j graph integrity

```bash
python scripts/_verify_neo4j.py
```

---

## Project Structure

```
application/          Agent (Mistral LLM + retrieval orchestration)
canonicalization/     Crosslinker (inter-regulation CITES resolution)
data/regulations/     Scraped HTML + parsed JSON (gitignored)
domain/               Regulations catalog, ontology, graph schema
  ontology/           Cross-reference patterns, defined terms, EUR-Lex HTML patterns
  schema/             Graph schema (JSON)
infrastructure/       Embeddings (sentence-transformers) + Neo4j driver
ingestion/            Full pipeline: scrape → normalize → parse
  parse/              Universal EUR-Lex parser
    structural_layer/ Preamble, enacting terms, annexes, final provisions
    semantic_layer/   Cross-references, definitions, normative modalities
  scrape/             Playwright-based EUR-Lex scraper
retrieval/            GraphRetriever (vector similarity + graph traversal)
scripts/              CLI entry points (load, embed, chat, test, verify)
```
