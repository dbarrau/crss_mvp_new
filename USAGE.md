# CRSS — Developer Operations Guide

## Prerequisites

| Component | Notes |
|---|---|
| **Python** | `crss_mvp` pyenv virtualenv (see `.python-version`) |
| **Neo4j** | Running locally on default ports (Bolt 7687 / Browser 7474) |
| **`.env`** | Must exist at project root with `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `MISTRAL_API_KEY` |

The loader auto-converts `http://localhost:7474` → `bolt://localhost:7687`, so either URI format works in `.env`.

---

## Quick Reference — Full Pipeline

```bash
# 1. Parse regulations (reuses cached HTML if present)
python -m ingestion.run_pipeline --celex 32024R1689 --lang EN
python -m ingestion.run_pipeline --celex 32017R0745 --lang EN
python -m ingestion.run_pipeline --celex 32017R0746 --lang EN

# 2. Load everything into Neo4j (wipe for clean re-import)
python scripts/load_neo4j.py --wipe

# 3. Embed provisions
python scripts/embed_provisions.py

# 4. Chat
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

**What it produces:**

```
data/regulations/<celex>/<lang>/
├── raw/raw.html       ← cached; pipeline reuses if present
└── parsed.json        ← provisions + relations + defined_terms
```

If `raw/raw.html` already exists, scraping is skipped. To force a re-scrape, delete the HTML file first.

**Supported regulations:**

| CELEX | Name |
|---|---|
| `32017R0745` | MDR 2017/745 |
| `32024R1689` | EU AI Act |
| `32017R0746` | IVDR 2017/746 |

**Quick re-parse of MDR + AI Act:**

```bash
python scripts/_reparse.py
# Results written to /tmp/crss_reparse_results.txt
```

### 2. Validate Parsed Output (optional)

```bash
python scripts/validate_parsed.py \
  data/regulations/32024R1689/EN/parsed.json \
  domain/schema/graph_schema.json
```

For a deeper quality analysis (provision counts, orphans, depth distribution):

```bash
python scripts/analyze_graphrag.py data/regulations/32024R1689/EN/parsed.json
# Saves <filename>_analysis_report.json alongside the input
```

### 3. Load into Neo4j — `scripts/load_neo4j.py`

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
| `--uri` | from `.env` | Neo4j bolt URI |
| `--user` | from `.env` | Neo4j username |
| `--password` | from `.env` | Neo4j password |
| `--database` | from `.env` | Neo4j database name |

**What gets created in Neo4j:**

Nodes (with structural labels):
`Document`, `Citation`, `Recital`, `Chapter`, `Section`, `Article`, `Paragraph`, `Subparagraph`, `Point`, `Annex`, `AnnexChapter`, `AnnexSection`, `AnnexPoint`, `AnnexSubpoint`, `AnnexBullet`, `DefinedTerm`

Edges:
- `HAS_PART` — ordered structural containment
- `CITES` — internal cross-references between provisions
- `CITES_EXTERNAL` / `AMENDS` — references to external EU acts
- `DEFINED_BY` — links `:DefinedTerm` → source `:Provision` (definition point)

Indexes created automatically:
- Unique constraint on `Provision.id` and `DefinedTerm.id`
- Lookup indexes on `Provision.celex`, `Provision.kind`, `DefinedTerm.term_normalized`, `DefinedTerm.category`

### 4. Embed Provisions — `scripts/embed_provisions.py`

```bash
python scripts/embed_provisions.py
```

No arguments. Reads provisions from Neo4j, embeds with `intfloat/multilingual-e5-small` (384 dims), writes `embedding` property back to each `:Provision` node.

- Prefix: `"passage: "` (asymmetric E5 encoding)
- Batch: 64 texts per encode call, 500 nodes per Neo4j write
- Eligible kinds: `article`, `paragraph`, `subparagraph`, `point`, `roman_item`, `recital`, `section`, `annex`, `annex_section`, `annex_point`, `annex_subpoint`, `annex_bullet`

### 5. Smoke Test

```bash
python scripts/test_agent.py
```

Runs a retriever test + a full agent test. Requires Neo4j (with embeddings) and `MISTRAL_API_KEY`.

### 6. Interactive Chat — `scripts/chat.py`

```bash
python scripts/chat.py
```

In-chat commands:

| Command | Effect |
|---|---|
| `quit` | Exit |
| `debug` | Toggle showing retrieved provisions before the answer |
| `k=N` | Change number of retrieved provisions (default 5) |

---

## Common Operations

### Re-ingest everything from scratch

```bash
# Re-parse all three regulations
python -m ingestion.run_pipeline --celex 32017R0745
python -m ingestion.run_pipeline --celex 32024R1689
python -m ingestion.run_pipeline --celex 32017R0746

# Wipe Neo4j and reload
python scripts/load_neo4j.py --wipe

# Re-embed
python scripts/embed_provisions.py
```

### Re-parse only (HTML already cached)

```bash
python -m ingestion.run_pipeline --celex 32024R1689
# Then reload + re-embed:
python scripts/load_neo4j.py --celex 32024R1689 --wipe
python scripts/embed_provisions.py
```

### Force re-scrape from EUR-Lex

```bash
rm data/regulations/32024R1689/EN/raw/raw.html
python -m ingestion.run_pipeline --celex 32024R1689
```

### Verify Neo4j graph integrity

```bash
python scripts/_verify_neo4j.py
# Results in /tmp/crss_neo4j_verify.txt
```

### Verify parsed.json hierarchy integrity

```bash
python scripts/_validate_hierarchy.py
# Results in /tmp/crss_validate.txt
```

### Query defined terms in Neo4j (Cypher examples)

```cypher
-- All actors across all regulations
MATCH (d:DefinedTerm {category: "actor"}) RETURN d.term, d.regulation

-- Exact term lookup
MATCH (d:DefinedTerm {term_normalized: "provider"})-[:DEFINED_BY]->(p)
RETURN d.term, d.category, p.text

-- All defined terms for one regulation
MATCH (d:DefinedTerm {celex: "32024R1689"}) RETURN d.term, d.category ORDER BY d.category, d.term
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j user (`NEO4J_USER` also accepted) |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database |
| `MISTRAL_API_KEY` | *(required for chat/agent)* | Mistral API key |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral model to use |

---

## Data Directory Layout

```
data/regulations/
├── 32017R0745/EN/          ← MDR
│   ├── raw/raw.html
│   └── parsed.json
├── 32017R0746/EN/          ← IVDR
│   ├── raw/raw.html
│   └── parsed.json
└── 32024R1689/EN/          ← AI Act
    ├── raw/raw.html
    └── parsed.json
```

`data/` is gitignored. Raw HTML and parsed JSON are local-only.

---

## Architecture at a Glance

```
EUR-Lex HTML
    │
    ▼  ingestion/run_pipeline.py
parsed.json  (provisions + relations + defined_terms)
    │
    ▼  scripts/load_neo4j.py
Neo4j  (:Provision nodes, :DefinedTerm nodes, HAS_PART/CITES/DEFINED_BY edges)
    │
    ▼  scripts/embed_provisions.py
Neo4j  (embedding property added to Provision nodes)
    │
    ▼  retrieval/graph_retriever.py
Hybrid retrieval  (in-memory cosine similarity → Cypher graph expansion)
    │
    ▼  application/agent.py → scripts/chat.py
Mistral-grounded regulatory Q&A
```
