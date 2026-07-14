"""Rank community embeddings against a query — diagnostic only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

from retrieval._config import QUERY_PREFIX

KEYWORDS =["article 5", "prohibit", "article 6", "classif", "systemic risk", "article 55"]

question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
    "What are all obligations of providers under the EU AI Act?"
)

model = SentenceTransformer("intfloat/multilingual-e5-base")
q_vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).astype(np.float32)

import os
from dotenv import load_dotenv
load_dotenv()
_pw = os.environ.get("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", _pw))
with driver.session() as s:
    rows = s.run(
        "MATCH (c:Community) WHERE c.summary_embedding IS NOT NULL "
        "RETURN c.community_id AS cid, c.level AS level, "
        "c.summary_text AS summary, c.summary_embedding AS emb"
    ).data()
driver.close()

scored = []
for r in rows:
    emb = np.array(r["emb"], dtype=np.float32)
    summary = r["summary"] or ""
    scored.append((float(np.dot(q_vec, emb)), r["cid"], r["level"], summary))

scored.sort(reverse=True)
print(f"Total communities: {len(scored)}\n")
for rank, (score, cid, level, summary) in enumerate(scored, 1):
    snippet = summary[:110]
    marker = " <--" if any(kw in summary.lower() for kw in KEYWORDS) else ""
    print(f"{rank:3d}. [L{level}] {score:.4f}  {cid}  {snippet}{marker}")
