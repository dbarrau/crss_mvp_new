# CRSS Project Structure Documentation

**Purpose**: Developer alignment guide for building the Compliance Readiness Support System (CRSS)
**Audience**: Development team working on MDR/IVDR/EU AI Act compliance tooling
**Last Updated**: March 26, 2026

---

## Project Overview

CRSS is a compliance readiness system that helps medtech startups navigate EU regulatory requirements — specifically the MDR (Regulation 2017/745), IVDR (Regulation 2017/746), and EU AI Act (Regulation 2024/1689). It ingests regulations from EUR-Lex, builds a structural knowledge graph in Neo4j, embeds provisions for vector search, and answers regulatory compliance questions via an LLM-powered agent grounded in retrieved legal text.

**Core Principle**: The system is a decision-support tool, NOT a certification system. It surfaces issues for human review rather than making regulatory judgments.

## Folder Structure (Actual)

```
crss/
│
├── domain/                        # Pure regulatory metadata & ontology
│   ├── regulations_catalog.py     # CELEX → metadata registry (single source of truth)
│   ├── ontology/
│   │   ├── cross_reference_patterns.py  # Regex patterns for legal cross-references
│   │   ├── defined_terms.py             # Patterns for extracting defined terms
│   │   └── eurlex_html.py               # EUR-Lex HTML element IDs & CSS classes
│   └── schema/
│       └── graph_schema.json      # JSON Schema for provisions & relations
│
├── ingestion/                     # Scrape + parse EUR-Lex HTML → parsed.json
│   ├── pipeline.py                # Backward-compatible wrapper
│   ├── run_pipeline.py            # Main pipeline orchestrator
│   ├── scrape/
│   │   └── scrape.py              # Playwright-based EUR-Lex retrieval
│   ├── parse/
│   │   ├── dispatcher.py          # Routes CELEX → parser, writes parsed.json
│   │   ├── normalizer.py          # Consolidated HTML → OJ format normalizer
│   │   ├── universal_eurlex_parser.py  # Universal two-pass parser (structural → semantic)
│   │   ├── base/
│   │   │   ├── registry.py        # CELEX → parser function registry
│   │   │   └── utils.py           # ParserContext class (tracks state during parsing)
│   │   ├── structural_layer/
│   │   │   ├── preamble_parser.py       # Citations & recitals
│   │   │   ├── enacting_terms_parser.py # Chapters, sections, articles, paragraphs, points
│   │   │   ├── annex_parser.py          # Annex hierarchies (chapters, sections, points)
│   │   │   └── final_provisions_parser.py
│   │   └── semantic_layer/
│   │       ├── cross_references.py      # Extract & resolve cross-refs → CITES relations
│   │       ├── definitions.py           # Extract defined terms → DefinedTerm nodes
│   │       └── normative_modalities.py  # Classify obligation/prohibition/permission (EN/DE/FR)
│   └── sandbox/                   # Development notebooks (not production)
│
├── canonicalization/              # Post-parse enrichment & cross-regulation linking
│   ├── crosslinker.py             # Resolve CITES_EXTERNAL → CITES in Neo4j
│   └── text_enrichment.py         # Context-prefix + bottom-up flattening for embeddings
│
├── infrastructure/                # External systems adapters
│   ├── embeddings/
│   │   └── batch_embedder.py      # Batch-embed provisions → Neo4j node properties
│   └── graphdb/
│       └── neo4j/
│           └── loader.py          # Load parsed.json → Neo4j structural graph
│
├── retrieval/
│   └── graph_retriever.py         # Hybrid in-memory vector + graph retriever
│
├── application/
│   └── agent.py                   # LLM-powered regulatory Q&A agent (Mistral)
│
├── scripts/                       # CLI entry-points
│   ├── chat.py                    # Interactive REPL for regulatory Q&A
│   ├── load_neo4j.py              # Load parsed.json into Neo4j (--wipe, --celex)
│   ├── embed_provisions.py        # Batch-embed all provisions
│   ├── analyze_graphrag.py        # Graph analytics
│   ├── diagnose_html.py           # HTML structure diagnostics
│   ├── _validate_hierarchy.py     # Validate provision hierarchy
│   ├── _verify_neo4j.py           # Verify Neo4j data integrity
│   └── test_agent.py              # Agent smoke test
│
├── data/
│   └── regulations/
│       ├── 32017R0745/EN/         # MDR
│       ├── 32017R0746/EN/         # IVDR
│       └── 32024R1689/EN/         # EU AI Act
│
└── docs/
    └── embeddings_and_agent.md    # Architecture notes on embedding + agent design
```

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│   APPLICATION (agent.py)                            │  ← LLM-powered regulatory Q&A
├─────────────────────────────────────────────────────┤
│   RETRIEVAL (graph_retriever.py)                    │  ← Hybrid vector + graph search
├─────────────────────────────────────────────────────┤
│   CANONICALIZATION (crosslinker, text_enrichment)   │  ← Post-parse enrichment & linking
├─────────────────────────────────────────────────────┤
│   INGESTION (scrape → parse → parsed.json)          │  ← EUR-Lex HTML acquisition & parsing
├─────────────────────────────────────────────────────┤
│   DOMAIN (regulations_catalog, ontology, schema)    │  ← Pure regulatory knowledge
└─────────────────────────────────────────────────────┘

Cross-cutting: INFRASTRUCTURE (Neo4j loader, batch embedder)
```

---

## Data Flow (End-to-End)

```
1. SCRAPE        EUR-Lex HTML (Playwright)
                    → data/regulations/{celex}/{lang}/raw/raw.html

2. PARSE         Structural + semantic parsing
                    → data/regulations/{celex}/{lang}/parsed.json

3. LOAD          parsed.json → Neo4j (:Provision nodes + :HAS_PART edges)
                    Script: python scripts/load_neo4j.py [--wipe]

4. EMBED         Batch-embed provisions → Neo4j embedding properties
                    Script: python scripts/embed_provisions.py

5. CROSSLINK     Resolve CITES_EXTERNAL → concrete CITES edges
                    Script: python -m canonicalization.crosslinker

6. QUERY         User question → vector search → graph expansion → LLM answer
                    Script: python scripts/chat.py
```

---

## Directory Structure & Responsibilities

### `domain/` — Pure Regulatory Knowledge

**Purpose**: Metadata, ontology patterns, and schemas. No I/O, no database code.

**Key Principle**: Domain layer NEVER imports from `infrastructure/` or `application/`.

#### `domain/regulations_catalog.py`

Central registry mapping CELEX IDs to regulation metadata. Single source of truth for regulation identity.

**Currently tracked regulations**:

| CELEX | Name | Type |
|-------|------|------|
| `32017R0745` | MDR (Regulation 2017/745) | Regulation |
| `32017R0746` | IVDR (Regulation 2017/746) | Regulation |
| `32024R1689` | EU AI Act (Regulation 2024/1689) | Regulation |

Supports consolidated versions via `source_celex` field.

#### `domain/ontology/`

Pattern libraries for extracting structured information from legal text.

- **`cross_reference_patterns.py`** — Four compiled regex patterns:
  - `EXPLICIT_REF`: Article/Annex/Point with "of this Regulation"
  - `RELATIVE_REF`: Bare "Article 10", "paragraph 2" in context
  - `RANGE_REF`: "Articles 102 to 109", "8, 9, 10 and 11"
  - `EXTERNAL_REF`: "Regulation (EU) 2017/745" with optional provision qualifiers
  - `FOOTNOTE_MARKER`: Strips `(1)` footnote noise before pattern matching

- **`defined_terms.py`** — Extracts quoted terms followed by "means" keyword. Classifies into categories: `actor`, `body`, `other`. Supports typographic quotes.

- **`eurlex_html.py`** — Constants for EUR-Lex HTML structure: element IDs (`tit_1`, `pbl_1`, `art_N`), CSS classes (`oj-normal`, `eli-main-title`), and regex patterns for repeating elements (articles, recitals, paragraphs).

#### `domain/schema/graph_schema.json`

JSON Schema (draft-07) defining provision and relation structures. Key provision fields: `id`, `celex`, `kind`, `level`, `text`, `text_for_analysis`, `path`, `hierarchy_depth`, `semantic_role`, `internal_refs`, `external_refs`.

---

### `ingestion/` — Data Acquisition & Parsing

**Purpose**: Retrieve EUR-Lex HTML and parse it into structured `parsed.json` files.

**Data Flow**: `EUR-Lex → Playwright scrape → HTML normalize → structural parse → semantic enrichment → parsed.json`

#### `ingestion/run_pipeline.py`

Main orchestrator. Validates CELEX against `regulations_catalog`, creates directory structure, sequences scraper → parser.

```python
from ingestion.run_pipeline import run
run("32024R1689", "EN")  # Scrape + parse AI Act
```

#### `ingestion/scrape/scrape.py`

Playwright-based scraping. Launches headless Chromium, navigates to EUR-Lex, waits for network idle, saves raw HTML.

Output: `data/regulations/{celex}/{lang}/raw/raw.html`

#### `ingestion/parse/dispatcher.py`

Routes each CELEX to its parser (via `base/registry.py`), normalizes consolidated HTML if needed, runs definition extraction, text enrichment, and writes `parsed.json`.

#### `ingestion/parse/normalizer.py`

Transforms consolidated EUR-Lex HTML (CELEX starting with `0`) to match original OJ format. Remaps CSS classes, reconstructs paragraph IDs, converts CSS-grid layouts to tables, strips amendment markers.

#### `ingestion/parse/universal_eurlex_parser.py`

Universal two-pass parser:
1. **Structural pass**: Preamble → enacting terms → final provisions → annexes
2. **Semantic pass**: Cross-reference resolution over the flat provisions list

Returns `{"provisions": [...], "relations": [...]}`.

#### `ingestion/parse/base/`

- **`registry.py`** — Maps CELEX IDs to parser functions. Currently all use `parse_eurlex_html`.
- **`utils.py`** — `ParserContext` class: tracks CELEX, language, provisions list, nodes dict. Factory methods: `make_node()`, `add_node()`. Generates IDs as `{celex}_{html_id}`.

#### `ingestion/parse/structural_layer/`

HTML-to-provision parsers for each document section:

| Parser | Handles | Output node kinds |
|--------|---------|-------------------|
| `preamble_parser.py` | Title, citations, recitals | `preamble`, `citation`, `recital` |
| `enacting_terms_parser.py` | Chapters, sections, articles, paragraphs, points | `chapter`, `section`, `article`, `paragraph`, `subparagraph`, `point`, `roman_item` |
| `annex_parser.py` | Annex hierarchies | `annex`, `annex_chapter`, `annex_part`, `annex_section`, `annex_subsection`, `annex_point`, `annex_subpoint`, `annex_bullet` |
| `final_provisions_parser.py` | Entry-into-force, repeals | `final_provisions` |

#### `ingestion/parse/semantic_layer/`

Post-structural semantic extraction:

- **`cross_references.py`** — `CrossReferenceResolver` class. Walks provisions, applies regex patterns, resolves to concrete provision IDs. Handles relative refs (bare "Article 10"), explicit refs ("of this Regulation"), ranges, external refs. Outputs `CITES` and `CITES_EXTERNAL` relation dicts.

- **`definitions.py`** — `extract_defined_terms()`. Scans provisions for quoted-term-means patterns. Outputs `DefinedTerm` node dicts and `DEFINED_BY` edge dicts. IDs: `{celex}_defterm_{normalized_term}`.

- **`normative_modalities.py`** — `classify_requirement_type()`. Pattern-based detection of obligation ("shall"), prohibition ("shall not"), permission ("may"), definition ("means") in EN/DE/FR.

---

### `canonicalization/` — Post-Parse Enrichment & Linking

#### `canonicalization/text_enrichment.py`

Called during parsing (by dispatcher). Two-phase text enrichment:
1. **Bottom-up flattening**: Parent nodes get concatenated children text (capped at 1500 chars for annexes)
2. **Context prefixing**: Each node's `text_for_analysis` is prefixed with ancestor titles (e.g., "Chapter III > Section 2 > Article 15: ...")

This solves the context-loss problem where a paragraph alone is meaningless without knowing which article/chapter it belongs to.

#### `canonicalization/crosslinker.py`

Post-Neo4j-load pass. Reads `parsed.json` files, finds `CITES_EXTERNAL` relations whose target regulation is also loaded in the graph, and creates concrete `CITES` edges in Neo4j.

```bash
python -m canonicalization.crosslinker           # Create edges
python -m canonicalization.crosslinker --dry-run  # Preview only
python -m canonicalization.crosslinker --cleanup  # Remove stale ExternalAct stubs
```

Cross-regulation links discovered: AI Act → MDR (2 edges), IVDR → MDR (21 edges).

---

### `infrastructure/` — External Systems Adapters

#### `infrastructure/graphdb/neo4j/loader.py`

`RegulationGraphLoader` class. Loads `parsed.json` into Neo4j.

**Node model**: Every provision gets a `:Provision` label plus a structural label.

15 structural labels:

```
:Document
:Citation
:Recital
:Chapter
:Section
:Article
:Paragraph
:Subparagraph
:Point
:Annex
:AnnexChapter
:AnnexSection
:AnnexPoint
:AnnexSubpoint
:AnnexBullet
```

Additional labels: `:AnnexPart`, `:AnnexSubsection`

**Node properties**:

| Property | Description |
|----------|-------------|
| `id` | Unique provision ID (e.g., `32024R1689_art_43`) |
| `celex` | CELEX identifier |
| `regulation_id` | Human name (e.g., "EU AI Act") |
| `kind` | Raw structural kind |
| `text` | Full text content |
| `text_for_analysis` | Context-prefixed text for embeddings |
| `number` | Item number ("1", "I", "a") |
| `title` | Optional heading |
| `display_ref` | Human-friendly label (e.g., "Article 43", "Chapter IX") |
| `display_path` | Full path (e.g., "Chapter IX / Section 1 / Article 72") |
| `name` | Visual caption for Neo4j Browser (= `display_ref`) |
| `hierarchy_depth` | Integer depth from root |
| `path_string` | "/"-joined ancestor IDs |
| `embedding` | 384-dim float vector (added by embed step) |


**Edges**:
- `(parent)-[:HAS_PART {order}]->(child)` — Ordered structural containment
- `(source)-[:CITES]->(target)` — Cross-reference (intra- and inter-regulation)
- `(source)-[:CITES_EXTERNAL]->(ExternalAct)` — Unresolved external references
- `(provision)-[:DEFINED_BY]->(DefinedTerm)` — Term definition links

**Editorial container handling**: `preamble`, `enacting_terms`, `final_provisions`, `annexes` containers are kept as real nodes (not flattened) for navigable hierarchy.

**Usage**:
```bash
python scripts/load_neo4j.py                           # Load all regulations
python scripts/load_neo4j.py --celex 32024R1689 --wipe # Wipe + reload AI Act
```

#### `infrastructure/embeddings/batch_embedder.py`

Batch-embeds provisions using `intfloat/multilingual-e5-small` (384 dimensions). Stores vectors as Neo4j node properties (no separate vector DB).

**Key details**:
- Model: `intfloat/multilingual-e5-small` (384-dim)
- Prefix: `"passage: "` for documents, `"query: "` for queries
- Embeddable kinds: `article`, `paragraph`, `recital`, `section`, `annex_section`, `annex_point`, `point`, `subparagraph`, `annex_subsection`
- Only embeds nodes with non-empty `text_for_analysis`
- Batch size: 64 (encoding), 500 (Neo4j write)
- Hardware: Detects MPS (Apple Silicon) for acceleration

```bash
python scripts/embed_provisions.py  # ~6745 provisions, ~2min on MPS
```

**Important**: After `load_neo4j.py --wipe`, embeddings are lost and must be regenerated.

---

### `retrieval/` — Hybrid Vector + Graph Search

#### `retrieval/graph_retriever.py`

`GraphRetriever` class. Loads all embeddings from Neo4j into an in-memory numpy matrix at startup (~1ms cosine similarity for 6k vectors).

**Retrieval pipeline**:
1. Encode query with `"query: "` prefix
2. Cosine similarity against in-memory matrix
3. Per-CELEX top-k allocation (supports multi-regulation queries)
4. Graph expansion via Cypher:
   - Children (up to 25 via `:HAS_PART`)
   - Internal citations (up to 5 via `:CITES`)
   - Cross-regulation citations (up to 8 via `:CITES`)
5. Reverse cross-reference expansion (find provisions in other regulations that cite retrieved ones)
6. Direct structural reference lookup (if query mentions "Article 26" explicitly)

**Key features**: Multi-regulation support, deduplication, cross-regulation edge traversal.

---

### `application/` — LLM-Powered Agent

#### `application/agent.py`

`ask(question, retriever, k=5) → str` — Main entry point.

**Pipeline**:
1. Detect mentioned regulations via regex patterns
2. Extract explicit provision references ("Article 43", "Annex VII")
3. HyDE query augmentation (generate hypothetical regulatory excerpt via Mistral)
4. Retrieve provisions (vector + graph)
5. Fetch relevant defined terms
6. Assemble context with definitions block + provisions
7. Call Mistral LLM with strict textual grounding system prompt

**LLM**: Mistral API (key in `.env` as `MISTRAL_API_KEY`)

**Grounding rules** (enforced in system prompt):
- No training memory for specific details — cite only retrieved text
- Preserve exact structural labels and ordinals from source
- No reordering of sub-items
- Preserve qualifiers

---

### `scripts/` — CLI Entry-Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `chat.py` | Interactive REPL | `python scripts/chat.py` |
| `load_neo4j.py` | Load data to Neo4j | `python scripts/load_neo4j.py [--wipe] [--celex X]` |
| `embed_provisions.py` | Batch-embed provisions | `python scripts/embed_provisions.py` |
| `analyze_graphrag.py` | Graph analytics | `python scripts/analyze_graphrag.py` |
| `diagnose_html.py` | HTML structure diagnostics | `python scripts/diagnose_html.py` |
| `_validate_hierarchy.py` | Validate provision hierarchy | Internal validation |
| `_verify_neo4j.py` | Verify Neo4j data integrity | Internal verification |
| `test_agent.py` | Agent smoke test | `python scripts/test_agent.py` |

---

## Neo4j Graph Model

### Current Stats
- ~5,923 provision nodes across 3 regulations
- ~6,745 embedded provisions
- Cross-regulation CITES edges: AI Act→MDR (2), IVDR→MDR (21)

### Environment Variables
```
NEO4J_URI=http://localhost:7474       # Auto-converted to bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j                  # (default)
MISTRAL_API_KEY=<key>
```

### Node Caption in Neo4j Browser
Nodes display the `name` property as their visual caption. This shows structural references like "Chapter IX", "Article 43", "Annex II" instead of raw IDs. The `name` property is set equal to `display_ref` during loading.

---

## Dependency Rules

```
┌───────────────┐
│  application  │  ← imports: retrieval, domain
└──────┬────────┘
       ↓
┌───────────────┐
│   retrieval   │  ← imports: infrastructure (Neo4j driver), domain
└──────┬────────┘
       ↓
┌───────────────────┐
│ canonicalization   │  ← imports: domain, neo4j driver (crosslinker)
└──────┬────────────┘
       ↓
┌───────────────┐
│   ingestion   │  ← imports: domain, canonicalization.text_enrichment
└──────┬────────┘
       ↓
┌───────────────┐
│    domain     │  ← imports: NOTHING (pure stdlib)
└───────────────┘

Cross-cutting:
└── infrastructure/  ← imports: domain (catalog), neo4j driver
```

---

## Data Directory Structure (Actual)

```
data/
└── regulations/
    ├── 32017R0745/          # MDR
    │   └── EN/
    │       ├── raw/
    │       │   ├── raw.html
    │       │   └── raw_legal_basis.html
    │       └── parsed.json
    ├── 32017R0746/          # IVDR
    │   └── EN/
    │       ├── raw/
    │       │   └── raw.html
    │       └── parsed.json
    └── 32024R1689/          # EU AI Act
        └── EN/
            ├── raw/
            │   └── raw.html
            └── parsed.json
```

---

## Provision ID Format

- Articles: `{celex}_art_{N}` (e.g., `32024R1689_art_43`)
- Paragraphs: `{celex}_{art:03d}.{para:03d}` (e.g., `32024R1689_043.003`)
- Other structural: `{celex}_{html_element_id}`
- Defined terms: `{celex}_defterm_{normalized_term}`

IDs are deterministic and stable across re-parses.

---

## Key Takeaways for Developers

1. **Domain is pure data**: `regulations_catalog.py` and `ontology/` contain no I/O — just metadata and regex patterns.

2. **Parsing is two-pass**: Structural (HTML → provision nodes) then semantic (cross-reference resolution, definition extraction).

3. **Text enrichment is critical**: `text_for_analysis` adds ancestor context to each provision, making embeddings much more useful than raw text.

4. **Embeddings live in Neo4j**: No separate vector DB. In-memory numpy arrays at query time (~1ms search over 6k vectors).

5. **Data flows one way**: `scrape → parse → load_neo4j → embed → crosslink → query`. After `--wipe`, you must re-embed.

6. **Cross-regulation linking**: The crosslinker resolves external references post-load. It's a separate step because it requires all regulations to be loaded first.

7. **Neo4j Browser captions**: Nodes use the `name` property for visual display. This equals `display_ref` (e.g., "Chapter IX", "Article 43").

8. **LLM grounding**: The agent enforces strict textual grounding — answers must cite only retrieved provision text, never LLM training data.

---

**Last Updated**: March 26, 2026
**Version**: 2.0
**Maintainers**: Diego Barra
