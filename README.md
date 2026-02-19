# CRSS Project Structure Documentation

**Purpose**: Developer alignment guide for building the Compliance Readiness Support System (CRSS)
**Audience**: Development team working on MDR/EU AI Act compliance tooling
**Last Updated**: February 18, 2026

---

## Project Overview

CRSS is a compliance readiness system that helps medical device manufacturers prepare MDR and EU AI Act technical documentation by detecting inconsistencies, mapping evidence, and surfacing gaps before submission to Notified Bodies.

**Core Principle**: The system is a decision-support tool, NOT a certification system. It surfaces issues for human review rather than making regulatory judgments.

## Folder Structure Reference

```
crss/
│
├── domain/                    # pure regulatory intelligence model
│   ├── models/
│   ├── ontology/
│   ├── identifiers/
│
├── language/                  # cross-cutting multilingual logic
│   ├── detection.py
│   ├── normalization.py
│   ├── mappings/
│
├── ingestion/
│   ├── scrape/
│   ├── parse/
│   │   ├── base/
│   │   └── registry.py
│   └── pipeline.py
│
├── canonicalization/
│   ├── canonicalizer.py
│   ├── crosslinker.py
│   └── versioning.py
│
├── infrastructure/            # external systems adapters
│   ├── graphdb/
│   │   └── neo4j/
│   ├── vectordb/
│   │   └── faiss/
│   └── embeddings/
│
├── retrieval/
│   ├── vector/
│   ├── graph/
│   ├── hybrid/
│   └── api.py
│
├── application/               # user-facing intelligence layer
│   ├── agents/
│   ├── services/
│   └── orchestration/
│
├── evaluation/
│
└── utils/
```
---

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                  APPLICATION                        │  ← User-facing workflows, agents, orchestration
├─────────────────────────────────────────────────────┤
│                    RETRIEVAL                        │  ← Query interfaces (vector, graph, hybrid)
├─────────────────────────────────────────────────────┤
│                CANONICALIZATION                     │  ← Normalize parsed data into canonical graph
├─────────────────────────────────────────────────────┤
│                   INGESTION                         │  ← Scrape + parse EUR-Lex regulations
├─────────────────────────────────────────────────────┤
│                     DOMAIN                          │  ← Pure regulatory business logic
└─────────────────────────────────────────────────────┘

Cross-cutting: LANGUAGE (multilingual support), INFRASTRUCTURE (DB/vector adapters)
```

---

## Directory Structure & Responsibilities

### `domain/` - Pure Regulatory Intelligence Model

**Purpose**: Contains all business logic about regulations, provisions, obligations. This is the heart of the system.

**Key Principle**: Domain layer NEVER imports from `infrastructure/` or `application/`. It defines interfaces that infrastructure implements.

```
domain/
├── models/          # Core entities representing regulatory concepts
├── ontology/        # Value objects defining regulatory types/categories
└── identifiers/     # Canonical ID system for cross-regulation references
```

#### `domain/models/`

**What Lives Here**: Core business entities (Provision, Obligation, Requirement, Regulation, TechnicalDocument)

**Example Files**:
- `provision.py` - A unit of regulatory text (article, paragraph, point)
- `obligation.py` - A legal requirement extracted from a provision
- `requirement.py` - Specific testable criteria from obligations
- `regulation.py` - Metadata about regulations (MDR, AI Act)
- `technical_document.py` - Manufacturer's technical documentation

**Key Properties**:
- Dataclasses or Pydantic models (NO SQLAlchemy/ORM)
- Pure business logic methods only
- No I/O operations (no file reading, no DB queries)
- Immutable where possible


**Developer Guidelines**:
- Models should be rich (contain business logic), not anemic (just getters/setters)
- No database concerns here (that's in `infrastructure/`)
- Think: "What does it mean to be a Provision?" not "How do we store a Provision?"

---

#### `domain/ontology/`

**What Lives Here**: Value objects and enums defining the regulatory type system

**Example Files**:
- `provision_types.py` - Enum: ARTICLE, PARAGRAPH, POINT, ANNEX, SECTION
- `obligation_types.py` - Enum: SAFETY, PERFORMANCE, DATA_GOVERNANCE, TRANSPARENCY
- `requirement_types.py` - Enum: MANDATORY, CONDITIONAL, RECOMMENDED
- `actor_roles.py` - Enum: PROVIDER, DEPLOYER, USER, NOTIFIED_BODY

**Key Properties**:
- Immutable value objects
- No business logic (just type definitions)
- Used for type safety across the system

---

#### `domain/identifiers/`

**What Lives Here**: Canonical ID system for uniquely identifying provisions across regulations and languages

**Key Properties**:
- Deterministic ID generation (same provision always gets same ID)
- Language-agnostic (MDR Article 10 has one ID regardless of EN/DE/FR)
- Supports cross-regulation references (AI Act → MDR)


**Developer Guidelines**:
- IDs must be stable across system updates
- Always frozen (immutable)
- Include validation in constructors
- Document ID format clearly (used in Neo4j, FAISS metadata, API responses)

---

### `language/` - Cross-Cutting Multilingual Logic

**Purpose**: Handle all language-related concerns (detection, normalization, translation alignment)

**Key Principle**: This layer is imported by ALL other layers. Keep it lightweight and dependency-free.

```
language/
├── detection.py       # Detect language of text
├── normalization.py   # Unicode normalization, whitespace cleanup
└── mappings/          # Language-specific rules and translations
```

#### `language/detection.py`

**What Lives Here**: Language detection for unlabeled text

**Developer Guidelines**:
- Use for unlabeled documents only (EUR-Lex URLs specify language)
- Minimum 20 characters for reliability
- Default to "en" on uncertainty

---

#### `language/normalization.py`

**What Lives Here**: Text normalization (Unicode, whitespace, punctuation)


**Developer Guidelines**:
- Always normalize before parsing or storage
- NFKD is critical for consistent EU text processing
- Don't normalize user queries (they may intentionally use special chars)

---

#### `language/mappings/`

**What Lives Here**: Language-specific rules and translations


**Developer Guidelines**:
- YAML for configuration, Python for logic
- Add new languages by editing YAML (no code changes)
- Keep patterns simple (complex parsing in parsers)

---

### `ingestion/` - Data Acquisition Layer

**Purpose**: Retrieve and parse regulatory documents from EUR-Lex into structured provisions

**Programming paradigm**: Functional - pure functions that take documentss and output chunks or provisions

**Data Flow**: `EUR-Lex HTML → scraping/ → parsing/ → List[Provision]`

```
ingestion/
├── scraping/        # Playwright-based EUR-Lex retrieval
├── parsing/         # Regulation-specific HTML parsers
│   ├── base/        # BaseParser + shared utilities
│   └── registry.py  # Map CELEX ID → Parser
└── pipeline.py      # Orchestrate scraping + parsing
```

#### `ingestion/scraping/`

**What Lives Here**: EUR-Lex document retrieval

**Developer Guidelines**:
- Use Playwright for dynamic content (EUR-Lex uses async loading)
- Cache HTML to disk (avoid re-scraping during development)
- Handle rate limiting (EUR-Lex has no official limit, but be respectful)
- Save raw HTML to `data/regulations/{celex_id}/{version}/raw/{lang}.html`

---

#### `ingestion/parsing/base/`

**What Lives Here**: Shared parsing utilities and base parser class

---

#### `ingestion/parsing/registry.py`

**What Lives Here**: Map CELEX IDs to parser implementations

**Developer Guidelines**:
- Register all parsers here
- Parser instances are reusable (stateless)
- Raise clear errors for unregistered CELEX IDs

---

#### `ingestion/pipeline.py`

**What Lives Here**: Orchestrate scraping + parsing

**Developer Guidelines**:
- Always save raw HTML (enables re-parsing without re-scraping)
- Save parsed provisions as JSON (enables skipping to canonicalization)
- Handle errors gracefully (scraping can fail)

---

### `canonicalization/` - Normalize to Canonical Graph

**Purpose**: Take parsed provisions (in multiple languages) and create a single canonical knowledge graph with multilingual properties

**Programming paradigm**: Functional with light OOP - fuynctions for normalization, but nodes/relations can be dataclasses (immutable) for convenience

**Data Flow**: `List[Provision] (EN, DE, FR) → Canonical Graph (multilingual)`

```
canonicalization/
├── canonicalizer.py  # Merge language versions into canonical graph
├── crosslinker.py    # Resolve cross-regulation references
└── versioning.py     # Track regulation amendments over time
```

#### `canonicalization/canonicalizer.py`

**What Lives Here**: Merge parsed provisions across languages into canonical graph

**Key Logic**:
1. Group provisions by canonical ID (same article in EN/DE/FR)
2. Merge text fields into multilingual dictionary
3. Verify consistency (are EN/DE/FR saying the same thing?)
4. Create canonical graph nodes


**Developer Guidelines**:
- Parse references from English text only (most reliable)
- Store references as graph edges in Neo4j
- Handle ambiguous references gracefully (log warnings, don't fail)

---

#### `canonicalization/versioning.py`

**What Lives Here**: Track regulation amendments and maintain historical versions

**Key Logic**:
1. Store each regulation version with effective date
2. Track which provisions changed between versions
3. Enable time-travel queries ("What did Article 10 say in 2020?")


**Developer Guidelines**:
- Store versions by effective date (not amendment CELEX ID)
- Always preserve old versions (never overwrite)
- Enable "What changed?" queries (critical for change impact analysis)

---

### `infrastructure/` - External Systems Adapters

**Purpose**: Isolate external dependencies (databases, vector stores, embedding models). Domain layer never imports from here.

**Key Principle**: Infrastructure implements interfaces defined in domain layer.

```
infrastructure/
├── graphdb/         # Graph database adapters
│   └── neo4j/
├── vectordb/        # Vector database adapters
│   └── faiss/
└── embeddings/      # Embedding model wrappers
```

#### `infrastructure/graphdb/neo4j/`

**What Lives Here**: Neo4j adapter and repository implementations


**Developer Guidelines**:
- Never import domain models into infrastructure (domain imports should be at top level)
- Use interfaces from `domain/repositories/` (keeps domain layer clean)
- Handle connection pooling, retries, errors here
- Log all database operations (audit trail)

---

#### `infrastructure/vectordb/faiss/`

**What Lives Here**: FAISS vector store adapter


**Developer Guidelines**:
- Use `IndexFlatIP` for cosine similarity (L2-normalize embeddings first)
- Save ID map alongside FAISS index (FAISS only stores vectors, not IDs)
- For large indexes (>1M provisions), use `IndexIVFFlat` with clustering

---

#### `infrastructure/embeddings/`

**What Lives Here**: Embedding model wrappers

**Example Files**:
- `multilingual_e5.py` - intfloat/multilingual-e5-large wrapper
- `batch_embedder.py` - Efficient batch embedding generation


**Developer Guidelines**:
- Use multilingual models (multilingual-e5, paraphrase-multilingual-mpnet)
- Always L2-normalize embeddings before storing in FAISS
- Use correct instruction prefixes (E5 models require "passage:" and "query:")
- Batch embed for efficiency (do NOT embed one at a time)

---

### `retrieval/` - Query Interfaces

**Purpose**: Provide runtime query capabilities over indexed data

**Data Flow**: `User Query → Retrieval → List[Provision]`

**Programming paradigm**:
- Indexing / Embeddings: Functional - building vectors, exporting to FAISS/Neo4j
- Retrieval API: LIght OOP - can use classes for retrievers (vector, graph, hybrid) which encapsulate config/state

****

```
retrieval/
├── vector/          # Vector similarity search
├── graph/           # Graph traversal search
├── hybrid/          # Combined vector + graph
└── api.py           # FastAPI endpoints
```

#### `retrieval/vector/`

**What Lives Here**: Vector similarity retrieval

**Developer Guidelines**:
- Separate concerns: FAISS returns IDs, repository returns Provisions
- Don't return raw FAISS results to users (always fetch full provisions)
- Support multilingual queries (model handles this automatically)

---

#### `retrieval/graph/`

**What Lives Here**: Graph traversal-based retrieval

**Developer Guidelines**:
- Limit traversal depth (infinite depth can return entire graph)
- Use DISTINCT to avoid duplicates in complex graphs
- Cache common hierarchy queries (they don't change often)

---

#### `retrieval/hybrid/`

**What Lives Here**: Combined vector + graph retrieval


**Developer Guidelines**:
- Vector search first (fast, good recall)
- Graph expansion second (adds context, improves precision)
- Re-ranking is crucial (research RRF, weighted fusion)
- Balance parameters: too much graph expansion → irrelevant results

---

#### `retrieval/api.py`

**What Lives Here**: FastAPI endpoints for retrieval

**Developer Guidelines**:
- Use Pydantic for request/response validation
- Return provision dicts (JSON serializable)
- Handle errors gracefully (400 for bad requests, 404 for not found)
- Add authentication in production

---

### `application/` - User-Facing Intelligence Layer

**Purpose**: High-level workflows, agent orchestration, business use cases

**Key Principle**: This is where the "intelligence" lives. Application layer coordinates domain + infrastructure to solve user problems.

**Programming Paradigm**: OOP - agents are stateful, orchestrate multiple calls, track content.

```
application/
├── agents/          # Multi-turn agentic workflows
├── services/        # Single-concern use cases
└── orchestration/   # Complex multi-service coordination
```

#### `application/agents/`

**What Lives Here**: LLM-powered agents for complex tasks


**Developer Guidelines**:
- Agents are stateful (maintain conversation history)
- Use async/await for LLM calls (they're slow)
- Always validate LLM outputs (they can hallucinate)
- Log all LLM interactions (audit trail)

---

#### `application/services/`

**What Lives Here**: Single-concern business use cases

**Developer Guidelines**:
- Services are stateless (pure functions with dependencies)
- Single responsibility (one service = one use case)
- Return structured data (not strings)
- Document assumptions clearly

---

#### `application/orchestration/`

**What Lives Here**: Complex workflows coordinating multiple services


**Developer Guidelines**:
- Orchestrators are high-level (coordinate services/agents)
- Handle errors gracefully (one service failure shouldn't crash workflow)
- Generate actionable outputs (not just data dumps)
- Log progress (these workflows can take minutes)

---

### `evaluation/` - Quality Assurance

**Purpose**: Test pipeline quality, retrieval accuracy, compliance detection

```
evaluation/
├── ingestion_tests.py   # Verify parsing accuracy
├── retrieval_tests.py   # Measure retrieval quality (precision, recall)
└── benchmarks.py        # Standard test queries
```

**Developer Guidelines**:
- Automate evaluation (run in CI)
- Track metrics over time (detect regressions)
- Use real user queries as benchmarks
- Set thresholds (e.g., precision > 0.8)

---

### `utils/` - Shared Utilities

**Purpose**: Truly generic helpers used across all layers

```
utils/
├── logging.py       # Structured logging
├── text.py          # Text processing utilities
└── dates.py         # Date handling
```

**Developer Guidelines**:
- Keep utils minimal (most logic belongs in domain/language)
- No business logic here (that's domain layer)
- Only include if used by 3+ modules

---

## Dependency Rules (CRITICAL)

```
┌─────────────┐
│ application │  ← Can import: domain, retrieval, infrastructure
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  retrieval  │  ← Can import: domain, infrastructure
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ canonicaliz-│  ← Can import: domain, language, ingestion
│   ation     │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  ingestion  │  ← Can import: domain, language
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   domain    │  ← Can import: NOTHING (except standard library)
└─────────────┘

Cross-cutting:
├── language/       ← Can be imported by ALL layers
└── infrastructure/ ← Can import: domain (interfaces only)
```

**Enforce with**:
```bash
# Install import-linter
pip install import-linter

# Create .import-linter
# layers:
#   - domain
#   - ingestion
#   - canonicalization
#   - retrieval
#   - application

# Run check
lint-imports
```

---

## Testing Strategy

### Unit Tests
- Test individual functions/classes
- Mock external dependencies
- Fast (< 1s per test)


### Evaluation Tests
- Test retrieval quality
- Benchmark queries with known answers
- Track metrics over time

---

## Data Directory Structure

```
data/
├── regulations/
│   └── {celex_id}/              # e.g., 32017R0745
│       └── {version}/           # e.g., 2020-04-24
│           ├── raw/             # Scraped HTML
│           │   ├── EN.html
│           │   ├── DE.html
│           │   └── FR.html
│           ├── parsed/          # Parsed provisions (per language)
│           │   ├── EN.json
│           │   ├── DE.json
│           │   └── FR.json
│           └── canonical/       # Canonical graph (multilingual)
│               ├── graph.json
│               └── meta.json
│
└── indexes/
    ├── embeddings/
    │   ├── multilingual_e5.faiss
    │   └── multilingual_e5.json  # ID map
    └── neo4j/
        ├── nodes.csv
        └── relationships.csv
```

---

## Key Takeaways for Developers

1. **Domain is king**: All business logic lives in `domain/`. No database code, no HTTP code.

2. **Infrastructure implements interfaces**: Domain defines `ProvisionRepository` interface, infrastructure implements `Neo4jProvisionRepository`.

3. **Language is cross-cutting**: Import `language/` from anywhere. Keep it lightweight.

4. **Data flows one way**: `scrape → parse → canonicalize → index → retrieve`. Never skip stages.

5. **Multilingual by design**: Every provision has text in multiple languages. Always use `provision.get_text(lang)`.

6. **Canonical IDs everywhere**: Use `CanonicalID` for all provision references. Never use raw strings.

7. **Test at all levels**: Unit tests for logic, integration tests for workflows, evaluation tests for quality.

8. **Configuration over code**: Add languages via YAML, not Python files.

9. **Audit everything**: Log all operations (scraping, parsing, indexing, queries). Critical for regulatory compliance.

10. **Think in layers**: Each layer has a clear job. Don't mix concerns.

---

**Last Updated**: February 18, 2026
**Version**: 1.0
**Maintainers**: Diego Barra
