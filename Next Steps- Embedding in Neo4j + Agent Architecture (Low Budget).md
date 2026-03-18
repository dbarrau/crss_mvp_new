# Next Steps: Embedding in Neo4j + Agent Architecture (Low Budget)

---

## First: Drop FAISS from Your Plan

Your `project_infrastructure.md:344` planned a separate FAISS adapter, but **Neo4j's native vector type makes that unnecessary** for your scale.

- You have ~6,400 nodes across 3 regulations → *tiny dataset*
- Neo4j supports:
  - `LIST<FLOAT>` vector property
  - `VECTOR INDEX`

### Benefits
- One system instead of two
- Simpler to operate
- Simpler queries
- Zero extra infrastructure cost

---

## Step 1 — Choose Your Embedding Model

Your docs mention `intfloat/multilingual-e5-large` (1024 dims).

### Options

| Model                      | Dims | Size  | Speed (CPU) | Quality      |
|---------------------------|------|-------|-------------|--------------|
| multilingual-e5-large     | 1024 | 560MB | Slow        | Best         |
| multilingual-e5-base      | 768  | 278MB | OK          | Good         |
| multilingual-e5-small     | 384  | 118MB | Fast        | Good enough  |

### Recommendation

**`multilingual-e5-small`**

- Only ~3–5% worse than `base` on legal benchmarks
- Much faster (5 min vs 25 min batch jobs on CPU)
- Easy to upgrade later (just re-embed)

---

## Step 2 — Create the Neo4j Vector Index

Run this once in Neo4j Browser:

```cypher
CREATE VECTOR INDEX provision_embedding IF NOT EXISTS
FOR (n:Provision) ON n.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

Notes

- Single index for all Provision nodes
- Filter by kind or celex at query time
- No need for multiple indexes

## Step 3 - Build the batch Embedder

File: `infrastructure/embeddings/batch_embedder.py`

```python title="batch_embedder.py"
"""Batch-embed all provisions with text_for_analysis and store in Neo4j."""
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os, logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX   = "query: "

EMBED_KINDS = {
    "article", "paragraph", "subparagraph", "point", "roman_item",
    "recital", "section",
    "annex", "annex_section", "annex_point", "annex_subpoint", "annex_bullet",
}

def run(model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 64):
    model = SentenceTransformer(model_name)
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "password")),
    )

    with driver.session() as s:
        rows = s.run(
            "MATCH (n:Provision) "
            "WHERE n.text_for_analysis IS NOT NULL AND n.kind IN $kinds "
            "RETURN n.id AS id, n.text_for_analysis AS text",
            kinds=list(EMBED_KINDS),
        ).data()

    logger.info("Embedding %d provisions…", len(rows))

    ids   = [r["id"] for r in rows]
    texts = [PASSAGE_PREFIX + r["text"] for r in rows]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    batch = [
        {"id": id_, "emb": emb.tolist()}
        for id_, emb in zip(ids, embeddings)
    ]

    BATCH = 500
    with driver.session() as s:
        for i in range(0, len(batch), BATCH):
            s.run(
                "UNWIND $batch AS row "
                "MATCH (n:Provision {id: row.id}) "
                "SET n.embedding = row.emb",
                batch=batch[i : i + BATCH],
            )
            logger.info("Stored %d / %d", min(i + BATCH, len(batch)), len(batch))

    driver.close()
    logger.info("Done. %d nodes embedded.", len(ids))
```

**Install dependency**
> `pip install sentence-transformers`

## Step 4 - Build the Retrieval Layer

**Pipeline**

```
Query
  ↓ embed with "query: "
  ↓ vector search → top-k Articles
  ↓ HAS_PART traversal → children
  ↓ follow CITES edges
  ↓ assemble context
  ↓ LLM
```

**Hybrid Retrieval cypher**

```
CALL db.index.vector.queryNodes('provision_embedding', $k, $queryEmbedding)
YIELD node AS art, score
WHERE art.kind IN ['article', 'annex_section']

WITH art, score
MATCH (art)-[:HAS_PART*1..2]->(leaf)
WHERE leaf.text_for_analysis IS NOT NULL

RETURN 
  art.id AS article_id,
  art.display_ref AS article_ref,
  art.text_for_analysis AS article_text,
  collect(DISTINCT {id: leaf.id, text: leaf.text_for_analysis}) AS children,
  score
ORDER BY score DESC
LIMIT $k
```

**Add Cross-Reference Expansion**	

```
OPTIONAL MATCH (leaf)-[:CITES]->(cited:Provision)
WITH ..., collect(cited.text_for_analysis) AS cited_texts
```

## Step 5 - LLM Choice (Low Budget for MVP)

**Minimal Interface**

```
def ask(question: str) -> str:
    query_vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    provisions = graph_search(query_vec, k=5)
    context = assemble_context(provisions)
    return llm.complete(SYSTEM_PROMPT + context + question)
```

**LLM Options**

| Option                     | Cost                  | Quality     |
|---------------------------|-----------------------|-------------|
| Ollama + mistral:7b       | Free (local)          | Good        |
| Groq (llama3.3-70b)       | Free (rate-limited)   | Excellent   |
| Gemini 2.0 Flash          | ~$0.075 / 1M tokens   | Best value  |
| Claude Haiku 3.5          | ~$0.25 / 1M tokens    | Very good   |

