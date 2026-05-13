# Embeddings & Agent Architecture in CRSS

This document explains what text is embedded, how embeddings are computed and stored, and how the agent uses them together with the Neo4j knowledge graph to answer regulatory questions.

---

## 1. What Text Gets Embedded: `text_for_analysis`

Not every raw text field in the graph is embedded. Instead, a dedicated enrichment step (`canonicalization/text_enrichment.py`) produces a specially prepared field called **`text_for_analysis`** on each provision. This field is designed to be the optimal input for embedding models.

### 1.1 Two-Phase Enrichment

#### Phase 1 — Bottom-Up Flattening

Many nodes in the regulation tree are structural containers (chapters, sections, articles) whose own `text` field is either empty or just a heading. These nodes would produce poor embeddings if embedded as-is.

The flattening phase walks the provision tree **bottom-up** and gives each parent node the concatenated text of all its descendants:

```
Article 11 (text: "Technical documentation")     ← heading-only
  ├── Paragraph 1 (text: "Before placing on the market…")
  ├── Paragraph 2 (text: "The technical documentation shall…")
  └── Paragraph 3 (text: "…")

After flattening:
  Article 11.text_for_analysis = "Before placing on the market… The technical documentation shall… …"
```

Leaf nodes (paragraphs, points, bullets) keep their own text unchanged.

#### Phase 2 — Context Prefixing

After flattening, each node's `text_for_analysis` is prefixed with its **structural ancestry** — the chapter, section, and article titles leading to it. This gives the embedding model positional awareness so that semantically similar text in different parts of the regulation can be distinguished.

The format is:

```
{Ancestor Chain} | {Body Text}
```

**Real example** from the EU AI Act:

```
Chapter III — Requirements for High-Risk AI Systems > Section 2 — … > Article 11 — Technical documentation | Before placing on the market or putting into service a high-risk AI system, the provider shall draw up the technical documentation…
```

### 1.2 Which Node Types Are Embedded

Not all nodes in the graph get an embedding. The batch embedder filters to provision and guidance kinds that carry meaningful normative or descriptive text:

| Embedded (`text_for_analysis ≠ null`) | Skipped (`text_for_analysis = null`) |
|---------------------------------------|---------------------------------------|
| `article`, `paragraph`, `subparagraph` | `document` |
| `point`, `roman_item` | `preamble`, `enacting_terms` |
| `recital`, `section`, `chapter` | `final_provisions`, `annexes` |
| `annex`, `annex_section`, `annex_subsection` | |
| `annex_part`, `annex_point`, `annex_subpoint`, `annex_bullet` | |
| **Guidance (MDCG):** `guidance_section`, `guidance_subsection` | |
| `guidance_paragraph`, `guidance_chart` | |

Both `:Provision` nodes (legislation) and `:Guidance` nodes (MDCG documents) are embedded using the same pipeline.

### 1.3 Why This Matters

The `text_for_analysis` field solves two problems at once:

1. **Empty parent nodes** — An article with 5 paragraphs gets a flattened body containing all its children's text, so it becomes searchable as a whole.
2. **Ambiguous leaf nodes** — A paragraph saying "shall comply with the requirements" is meaningless without context. The prefix `Chapter III > Section 2 > Article 11 — Technical documentation` tells the model _which_ requirements are meant.

---

## 2. How Embeddings Are Computed

### 2.1 The Embedding Model

| Property | Value |
|----------|-------|
| Model | `intfloat/multilingual-e5-base` |
| Dimensions | 768 |
| Size | ~279 MB |
| Family | E5 (EmbEddings from bidirEctional Encoder rEpresentations) |
| Multilingual | Yes — supports 100+ languages including all EU official languages |

The E5 model family requires **instruction prefixes** to distinguish between documents being indexed and queries being searched:

- **Indexing** (passages stored in the database): `"passage: "` + text
- **Querying** (user questions at runtime): `"query: "` + text

This asymmetric prefix design is critical — it tells the model whether the input is a document to be found or a question doing the finding.

### 2.2 The Batch Embedding Process

The batch embedder (`infrastructure/embeddings/batch_embedder.py`) runs as a one-time offline step:

```
┌──────────────────────────────────────────────────────────────┐
│  1. Query Neo4j for all :Provision AND :Guidance nodes with  │
│     text_for_analysis IS NOT NULL AND kind IN [embed_kinds]  │
│     (UNION ALL — both legislation and MDCG documents)        │
│                                                              │
│  2. Prepend "passage: " + display_path + ": " to each text   │
│     (structural ancestry prefix for positional awareness)    │
│                                                              │
│  3. Encode all texts with multilingual-e5-base               │
│     • batch_size = 64                                        │
│     • normalize_embeddings = True  (L2-norm → unit vectors)  │
│     • Uses MPS (Apple Silicon GPU) if available               │
│                                                              │
│  4. Write embedding vectors back to Neo4j as                 │
│     LIST<FLOAT> properties on each Provision/Guidance node   │
│     • Stored in batches of 500 via UNWIND/SET                │
└──────────────────────────────────────────────────────────────┘
```

**Run with:**

```bash
python scripts/embed_provisions.py
```

### 2.3 Where Embeddings Are Stored

Each embedded node in Neo4j gains a property:

```
(:Provision {
    id: "32024R1689_art_11_par_1",
    text_for_analysis: "Chapter III — … | Before placing on the market…",
    embedding: [0.0234, -0.0891, 0.0456, …]    ← 768 floats
})

(:Guidance {
    id: "MDCG_2020_3_sec_3_1",
    text_for_analysis: "Section 3.1 — … | Significant changes include…",
    embedding: [0.0312, -0.0745, 0.0521, …]    ← 768 floats
})
```

The embeddings live _on the nodes themselves_ as a `LIST<FLOAT>` property — there is no separate vector index or external vector database. This is sufficient because:

- The dataset is small (~6,300 vectors)
- Similarity search is done in-memory with numpy (see Section 3)
- It keeps the infrastructure simple — one database, one system

---

## 3. How the Retriever Uses Embeddings + Graph

The `GraphRetriever` (`retrieval/graph_retriever.py`) implements a **hybrid retrieval** strategy that combines vector similarity search with graph traversal.

### 3.1 Architecture Overview

```
User Question
     │
     ▼
┌─────────────────────────────┐
│  Encode with "query: " +   │   ← SentenceTransformer
│  multilingual-e5-small      │
└────────────┬────────────────┘
             │ query vector (384d)
             ▼
┌─────────────────────────────┐
│  Cosine Similarity          │   ← numpy matrix multiplication
│  against 6,301 embeddings   │      ~1ms on CPU
│  in memory                  │
└────────────┬────────────────┘
             │ top-k article/section IDs
             ▼
┌─────────────────────────────┐
│  Graph Expansion (Cypher)   │   ← Neo4j
│  • HAS_PART → children      │
│  • CITES → cross-references │
└────────────┬────────────────┘
             │ enriched provisions
             ▼
        Agent / LLM
```

### 3.2 Step-by-Step Retrieval

#### Step 1 — Load Index (once, at startup)

When the `GraphRetriever` is instantiated, it loads **all** embedded provisions and guidance nodes from Neo4j into a numpy matrix:

```python
# Cypher: fetch all embedded vectors (legislation + guidance)
MATCH (n:Provision)
WHERE n.embedding IS NOT NULL
RETURN n.id AS id, n.kind AS kind, n.embedding AS emb
UNION ALL
MATCH (n:Guidance)
WHERE n.embedding IS NOT NULL
RETURN n.id AS id, n.kind AS kind, n.embedding AS emb
```

This produces a matrix of shape `(N, 768)` — each row is one node's embedding. N covers both legislation provisions and MDCG guidance nodes.

#### Step 2 — Encode the User's Question

The user's natural-language question is encoded with the `"query: "` prefix. This is the E5 model's way of marking asymmetric retrieval — the prefix signals that this text is a _query_ looking for relevant _passages_.

```python
q_vec = model.encode("query: " + question, normalize_embeddings=True)
```

#### Step 3 — Cosine Similarity (In-Memory)

Because all embeddings are L2-normalized (unit vectors), cosine similarity reduces to a simple dot product:

```python
scores = matrix @ q_vec    # (N,) vector of similarity scores
```

This operation completes in ~1ms for the full index — far faster than any database round-trip.

#### Step 4 — Filter to Parent-Level Kinds

The retriever filters results to **parent-level provisions** only:

- `article` — the main structural unit of regulatory text
- `annex_section`, `annex_subsection`, `annex_point`, `annex_part` — numbered units within annexes
- `recital` — explanatory preamble provisions
- `section` — structural groupings within chapters
- `guidance_section`, `guidance_subsection` — sections within MDCG guidance documents

Each leaf node (paragraph, point, roman item) is mapped to its nearest parent-level ancestor using the stored `path_string` property. This prevents the top-k from being consumed by, say, five paragraphs from the same article — instead we get the five most relevant *articles or sections*, then expand their children in the next step.

#### Step 5 — Graph Expansion via Cypher

For each top-k article, the retriever runs a Cypher query that traverses the graph structure:

```cypher
-- Get children (paragraphs, points) up to 5 levels deep
OPTIONAL MATCH (art)-[:HAS_PART*1..5]->(leaf)
WHERE leaf.text_for_analysis IS NOT NULL

-- Internal citations (same regulation), up to 5 levels deep
OPTIONAL MATCH (art)-[:HAS_PART*1..5]->()-[:CITES]->(cited:Provision)
WHERE cited.text_for_analysis IS NOT NULL AND cited.celex = art.celex

-- Cross-regulation citations (different regulation)
OPTIONAL MATCH (art)-[:HAS_PART*1..5]->()-[:CITES]->(xref:Provision)
WHERE xref.text_for_analysis IS NOT NULL AND xref.celex <> art.celex

-- Sibling expansion for Guidance nodes: pulls adjacent sections
-- from the same parent so that related examples appear together
OPTIONAL MATCH (parent:Guidance)-[:HAS_PART]->(art)
OPTIONAL MATCH (parent)-[:HAS_PART]->(sibling:Guidance)
```

This yields a rich context block per article:

```
Article 11 (score: 0.847)
  ├── Paragraph 1: "Before placing on the market…"
  ├── Paragraph 2: "The technical documentation shall…"
  ├── Point (a): "a general description of the AI system…"
  └── Cross-references:
       → Article 9 (Risk management system)          [internal]
       → Annex IV (Technical documentation)          [internal]
       → Article 43 MDR (Conformity assessment)      [cross-reg]
```

For cited nodes that are high-level containers with short text (e.g. "Annex XIV"), the retriever runs a second Cypher pass to drill into their direct children, so the LLM sees the actual substantive content rather than just a heading.

### 3.3 Why Hybrid (Vector + Graph)?

| Vector-only | Graph-only | Hybrid (what CRSS does) |
|-------------|-----------|------------------------|
| Finds semantically similar text | Finds structurally related text | Both |
| Misses children/siblings | Requires exact IDs upfront | Vector narrows, graph expands |
| No cross-references | Hard to rank by relevance | Ranked results with full context |

The graph expansion is what makes this a **GraphRAG** system rather than plain RAG. A standard RAG system would return isolated text chunks. CRSS returns articles _with their full structural hierarchy and cross-references_, giving the LLM the regulatory context needed for accurate answers.

---

## 4. How the Agent Uses Retrieved Context

The agent (`application/agent.py`) orchestrates retrieval and LLM generation.

### 4.1 Agent Pipeline

```
User Question
     │
     ▼
┌──────────────────────────┐
│  GraphRetriever.retrieve │  → top-5 provisions/guidance nodes
└────────────┬─────────────┘     with children and cross-references
             │
             ▼
┌──────────────────────────┐
│  _format_context()       │  → structured text block with:
│                          │    [1] Article ref (Regulation)
│                          │        Path: Chapter > Section > Article
│                          │        Body text
│                          │        - Paragraph 1: …
│                          │        - Point (a): …
│                          │    Cross-references:
│                          │        → Cited provision text
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Mistral LLM             │  ← EU-hosted (mistral.ai, Paris)
│  model: mistral-large    │
│  temperature: 0.1        │     Low temperature → deterministic,
│                          │     less creative answers
│  System prompt:          │
│  "You are an EU regulatory│
│   compliance expert…"    │
│  "Answer strictly based   │
│   on provided context"   │
└────────────┬─────────────┘
             │
             ▼
        Answer with citations
```

### 4.2 Context Formatting

The retrieved provisions are formatted into a structured text block that the LLM receives as context. Each provision block includes:

1. **Header** — Article reference and regulation name (e.g., `[1] Article 11 (EU AI Act)`)
2. **Path** — Structural location (e.g., `Chapter III / Section 2 / Article 11`)
3. **Body** — The article's `text_for_analysis`
4. **Children** — Each child paragraph/point with its text (truncated to 500 chars)
5. **Cross-references** — Text of provisions cited by this article (truncated to 300 chars)

### 4.3 System Prompt Design

The system prompt constrains the LLM to:

- Answer **only** from the provided regulatory context
- **Cite specific articles** and paragraphs
- **Explicitly say** when context is insufficient (rather than hallucinating)
- Structure answers with: direct answer → regulatory references → cross-references

### 4.4 The LLM: Mistral (EU-Hosted)

| Property | Value |
|----------|-------|
| Provider | Mistral AI (Paris, France) |
| Model | `mistral-large-latest` (configurable via `MISTRAL_MODEL` env var) |
| Hosting | EU-based servers |
| API | `mistralai` Python SDK v2.x |
| GDPR | Data processed in EU |

Mistral was chosen because it is the only major LLM provider that is both **European-headquartered** and **hosts data in the EU** — important for a system handling regulatory compliance data.

---

## 5. Data Flow (End to End)

```
EUR-Lex HTML
     │
     ▼  ingestion/parse/
┌─────────────────────┐
│  Parse into          │
│  provisions tree     │
│  (parsed.json)       │
└────────┬────────────┘
         │
         ▼  canonicalization/text_enrichment.py
┌─────────────────────┐
│  Enrich each node:   │
│  1. Flatten children │
│  2. Prefix ancestry  │
│  → text_for_analysis │
└────────┬────────────┘
         │
         ▼  infrastructure/graphdb/neo4j/loader.py
┌─────────────────────┐
│  Load into Neo4j     │
│  :Provision nodes    │
│  :HAS_PART edges     │
│  :CITES edges        │
└────────┬────────────┘
         │
         ▼  infrastructure/embeddings/batch_embedder.py
┌─────────────────────┐
│  Embed with          │
│  multilingual-e5     │
│  "passage: " + text  │
│  → n.embedding       │
└────────┬────────────┘
         │
         ▼  (at query time)
┌─────────────────────┐
│  retrieval/          │  "query: " + question → cosine similarity
│  graph_retriever.py  │  → top-k articles → graph expansion
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  application/        │  format context → Mistral → grounded answer
│  agent.py            │
└─────────────────────┘
```

---

## 6. Commands Reference

```bash
# One-time: embed all provisions (~5 min on CPU, ~1 min on MPS)
python scripts/embed_provisions.py

# Interactive chat with the knowledge graph
python scripts/chat.py

# Inside chat.py:
#   debug  — show retrieved provisions and similarity scores
#   k=N   — change number of retrieved provisions (default 5)
#   quit  — exit
```

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| In-memory similarity instead of Neo4j Vector Index | Neo4j 5 Community lacks vector index support; numpy is faster for <10k vectors anyway |
| `multilingual-e5-base` (768d) | Good balance of quality and speed; multilingual coverage for all EU official languages |
| `"passage:"` / `"query:"` asymmetric prefixes | Required by the E5 model family; improves retrieval quality by ~15% vs. no prefix |
| `display_path` included in passage text | Provides structural ancestry context during embedding, reducing ambiguity for identical text in different locations |
| Filter top-k to parent-level kinds | Prevents duplicate context from siblings; graph expansion fills in child detail |
| Store embeddings as node properties | No extra infrastructure; single-system simplicity for a small dataset |
| Both `:Provision` and `:Guidance` nodes embedded | MDCG guidance documents are first-class retrieval targets alongside legislation |
| Mistral (EU company, EU hosting) | GDPR alignment; European data sovereignty for regulatory compliance work |
| Temperature 0.1 for answers, 0.0 for HyDE | Deterministic HyDE hypothesis; near-deterministic final answer generation |
