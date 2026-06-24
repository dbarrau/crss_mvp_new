"""Graph-aware vector retrieval over Neo4j provision embeddings.

Performs hybrid retrieval:
1. In-memory cosine similarity → top-k parent provisions (numpy, fast for <10k nodes)
2. Graph traversal → expand children via HAS_PART
3. Cross-reference expansion → follow CITES edges

No Neo4j Vector Index plugin required — embeddings are loaded into
memory at startup and similarity is computed with numpy.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Cross-encoder reranking: widen the cosine candidate pool before reranking.
# When a reranker is active, retrieve _CANDIDATE_MULTIPLIER × k candidates
# from cosine similarity, then rerank to keep the final k.  The multiplier
# is capped at _CANDIDATE_CAP to bound cross-encoder latency (each candidate
# is one forward pass through a ~568M-param model).
_CANDIDATE_MULTIPLIER = 5
_CANDIDATE_CAP = 48

# Cross-encoder truncation length.  Legal provisions front-load their key
# content, so 320 tokens captures the discriminative text at roughly half the
# inference cost of the model's 512 maximum.
_RERANK_MAX_LEN = 320

# Blend weight between the (normalised) cross-encoder score and the
# (normalised) cosine score.  0.0 = cosine only, 1.0 = cross-encoder only.
# Kept below 1.0 so a dominant cosine match (e.g. an explicitly-named annex)
# cannot be buried by a vocabulary-similar neighbour the cross-encoder prefers.
_RERANK_WEIGHT = 0.6

# Reciprocal Rank Fusion constant.  RRF score for a document is
# sum over channels of 1/(K + rank).  K=60 is the value from the original
# Cormack et al. RRF paper and is the de-facto standard; larger K flattens
# the contribution of top ranks, smaller K sharpens it.
_RRF_K = 60

# Weight of the lexical (BM25) channel in the fusion, relative to dense=1.0.
# Kept below 1.0 so the lexical channel *breaks ties* and rescues exact
# heading/term matches the dense model smears, without letting a
# mediocre-dense-but-lexically-dense node override a near-perfect dense match
# (e.g. the verbatim "'AI system' means …" definition point).
_LEXICAL_WEIGHT = 0.5

# Name of the Neo4j full-text (Lucene/BM25) index over provision text.
_FULLTEXT_INDEX = "provision_fulltext"

# Lucene query-syntax special characters.  These are stripped from the user
# question before it is passed to the full-text index so that punctuation
# (e.g. "?", "/", parentheses in "Article 6(3)") cannot raise a query-parse
# error or be misinterpreted as Lucene operators.
_LUCENE_SPECIALS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _sanitise_lucene(text: str) -> str:
    """Strip Lucene operator characters and collapse whitespace.

    The full-text index uses OR semantics across the remaining terms, so a
    cleaned bag-of-words query is sufficient for BM25 scoring.
    """
    cleaned = _LUCENE_SPECIALS.sub(" ", text)
    return " ".join(cleaned.split())

# Parent-level kinds we want the vector search to return.
# annex_point is included so that deep annex leaves (subpoints, bullets)
# anchor to a specific numbered point rather than a broad subsection.
_PARENT_KINDS = frozenset({
    "article", "annex_section", "annex_subsection", "annex_point",
    "annex_part",
    "recital", "section",
    # Guidance (MDCG) — guidance_paragraph and guidance_chart are anchor-worthy
    # so a retrieved deep guidance leaf resolves to its own paragraph/chart
    # rather than collapsing up to the enclosing section.
    "guidance_section", "guidance_subsection",
    "guidance_paragraph", "guidance_chart",
})

# Graph expansion: given top-k article IDs, fetch children + cross-refs.
# Cross-regulation citations are returned separately with higher budget
# since they are the most valuable for multi-regulation questions.
_EXPAND_CYPHER = """\
UNWIND $ids AS aid
OPTIONAL MATCH (p1:Provision {id: aid})
OPTIONAL MATCH (p2:Guidance  {id: aid})
WITH coalesce(p1, p2) AS art
WHERE art IS NOT NULL

// Parent expansion: get the higher-level context (e.g., if art is a point, get its Annex/Section)
OPTIONAL MATCH (art)<-[:HAS_PART*1..3]-(parent:Provision)
WHERE parent.text_for_analysis IS NOT NULL

WITH art,
     collect(DISTINCT {
       id: parent.id,
       kind: parent.kind,
       text: parent.text_for_analysis,
       ref: parent.display_ref,
       binding_force: parent.binding_force
     })[..3] AS parents

OPTIONAL MATCH (art)-[:HAS_PART*1..5]->(leaf)
WHERE leaf.text_for_analysis IS NOT NULL

WITH art, parents,
     collect(DISTINCT {
       id: leaf.id,
       kind: leaf.kind,
       text: leaf.text_for_analysis,
       raw_text: leaf.text,
       ref: leaf.display_ref,
       binding_force: leaf.binding_force
     })[..25] AS children

// Sibling expansion for Guidance nodes: when a guidance_paragraph is
// retrieved, also pull its siblings under the same parent so both the
// "significant" and "non-significant" example lists appear together.
OPTIONAL MATCH (parent_guidance:Guidance)-[:HAS_PART]->(art)
WHERE art:Guidance
OPTIONAL MATCH (parent_guidance)-[:HAS_PART]->(sibling:Guidance)
WHERE sibling <> art AND sibling.text_for_analysis IS NOT NULL

WITH art, parents, children,
     collect(DISTINCT {
       id:   sibling.id,
       kind: sibling.kind,
       text: sibling.text_for_analysis,
       raw_text: sibling.text,
       ref:  sibling.display_ref,
       binding_force: sibling.binding_force
     })[..10] AS siblings

// Internal citations (same regulation)
OPTIONAL MATCH (art)-[:HAS_PART*0..5]->()-[:CITES]->(cited:Provision)
WHERE cited.text_for_analysis IS NOT NULL
  AND cited.celex = art.celex

WITH art, parents, children, siblings,
     [x IN collect(DISTINCT {
       id:   cited.id,
       ref:  cited.display_ref,
       text: cited.text_for_analysis,
       binding_force: cited.binding_force
     }) WHERE x.id IS NOT NULL][..12] AS internal_cited

// Cross-regulation citations (different regulation)
OPTIONAL MATCH (art)-[:HAS_PART*0..5]->()-[:CITES]->(xref:Provision)
WHERE xref.text_for_analysis IS NOT NULL
  AND xref.celex <> art.celex

WITH art, parents, children, siblings, internal_cited,
     collect(DISTINCT {
       id:   xref.id,
       ref:  xref.display_ref,
       text: xref.text_for_analysis,
       binding_force: xref.binding_force
     })[..8] AS cross_reg_cited

// Inbound INTERPRETS: when `art` is a legislation provision, pull the MDCG
// guidance that interprets it so authoritative interpretation appears beside
// the binding text (edge direction is (:Guidance)-[:INTERPRETS]->(:Provision)).
OPTIONAL MATCH (interp_g:Guidance)-[:INTERPRETS]->(art)
WHERE interp_g.text_for_analysis IS NOT NULL

WITH art, parents, children, siblings, internal_cited, cross_reg_cited,
     [x IN collect(DISTINCT {
       id:   interp_g.id,
       ref:  interp_g.display_ref,
       text: interp_g.text_for_analysis
     }) WHERE x.id IS NOT NULL][..4] AS interpreting_guidance

// Outbound INTERPRETS: when `art` is a guidance node, pull the legislation
// provisions it interprets so the guidance is anchored to the binding source.
OPTIONAL MATCH (art)-[:INTERPRETS]->(interp_p:Provision)
WHERE interp_p.text_for_analysis IS NOT NULL

WITH art, parents, children, siblings, internal_cited, cross_reg_cited,
     interpreting_guidance,
     [x IN collect(DISTINCT {
       id:   interp_p.id,
       ref:  interp_p.display_ref,
       text: interp_p.text_for_analysis
     }) WHERE x.id IS NOT NULL][..4] AS interpreted_provisions

RETURN
  art.id              AS article_id,
  art.celex           AS celex,
  art.regulation_id   AS regulation,
    art.community_id    AS community_id,
  art.display_ref     AS article_ref,
  art.display_path    AS article_path,
  art.text_for_analysis AS article_text,
  art.provision_role  AS provision_role,
  art.binding_force   AS binding_force,
  parents + children + siblings AS children,
  internal_cited + cross_reg_cited AS cited_provisions,
  cross_reg_cited,
  interpreting_guidance,
  interpreted_provisions
"""

# Reverse cross-regulation expansion: find articles in OTHER regulations
# that have CITES edges pointing at any of the retrieved provision IDs
# (or their descendants). This surfaces "the other side" of cross-reg links.
_REVERSE_XREF_CYPHER = """\
UNWIND $ids AS aid
MATCH (art:Provision {id: aid})
MATCH (src)-[:CITES]->(art)
WHERE src.celex <> art.celex
MATCH (srcArt:Provision)-[:HAS_PART*0..5]->(src)
WHERE srcArt.kind IN ['article', 'annex_section', 'annex_subsection',
                       'annex_point', 'annex_part', 'recital', 'section']
  AND NOT srcArt.id IN $ids
WITH srcArt, COUNT(src) AS citation_freq
RETURN DISTINCT
  srcArt.id            AS article_id,
  srcArt.celex         AS celex,
  srcArt.regulation_id AS regulation,
  srcArt.display_ref   AS article_ref,
  srcArt.display_path  AS article_path,
  srcArt.text_for_analysis AS article_text,
  srcArt.provision_role AS provision_role,
  srcArt.binding_force AS binding_force,
  citation_freq
ORDER BY citation_freq DESC
LIMIT 10
"""

# Cited-container drilldown: when a CITES edge points to a high-level
# container node (e.g. "Annex XIV") whose own text is very short, fetch
# the container's direct children so the LLM sees the actual content.
_CITED_CHILDREN_CYPHER = """\
UNWIND $ids AS cid
MATCH (c:Provision {id: cid})-[:HAS_PART]->(child)
WHERE child.text_for_analysis IS NOT NULL
RETURN cid               AS container_id,
       child.id           AS id,
       child.kind         AS kind,
       child.display_ref  AS ref,
       child.text_for_analysis AS text,
       child.binding_force AS binding_force
ORDER BY cid, child.hierarchy_depth, child.id
"""

# Traverse legal reasoning edges from a seed provision.
# Returns all provisions reachable via TRIGGERS_OBLIGATION_CLUSTER or
# IS_PREREQUISITE_FOR edges within 2 hops.  Used by retrieve_by_chain.
_CHAIN_SEED_LOOKUP_CYPHER = """\
UNWIND $refs AS ref
OPTIONAL MATCH (p1:Provision {celex: $celex})
  WHERE toLower(p1.display_ref) = toLower(ref)
  AND p1.kind IN ['article', 'annex_section', 'annex_part', 'annex',
                  'recital', 'section', 'chapter', 'title']
OPTIONAL MATCH (p2:Guidance {celex: $celex})
  WHERE toLower(p2.display_ref) = toLower(ref)
WITH coalesce(p1, p2) AS seed
WHERE seed IS NOT NULL
RETURN seed.id AS seed_id
ORDER BY seed.hierarchy_depth ASC
LIMIT 5
"""

_CHAIN_TRAVERSE_CYPHER = """\
UNWIND $seed_ids AS sid
MATCH (seed) WHERE seed.id = sid
OPTIONAL MATCH (seed)-[:TRIGGERS_OBLIGATION_CLUSTER|IS_PREREQUISITE_FOR|REQUIRES_PRIOR_CHECK*1..2]->(linked)
WHERE linked IS NOT NULL
  AND linked.kind IN ['article', 'annex_section', 'annex_part', 'annex',
                      'recital', 'section']
RETURN DISTINCT
  linked.id           AS article_id,
  linked.celex        AS celex,
  linked.display_ref  AS article_ref,
  linked.display_path AS article_path,
  linked.text_for_analysis AS article_text,
  linked.provision_role AS provision_role,
  linked.binding_force AS binding_force
"""

# Direct provision lookup by display_ref.  Used for structural questions
# that explicitly name a provision ("What does Annex I contain?",
# "What does Article 26 require?").  Bypasses vector similarity entirely.
_DIRECT_REF_CYPHER = """\
UNWIND $refs AS ref
OPTIONAL MATCH (p1:Provision) WHERE toLower(p1.display_ref) = toLower(ref)
OPTIONAL MATCH (p2:Guidance)  WHERE toLower(p2.display_ref) = toLower(ref)
WITH ref, collect(p1) + collect(p2) AS nodes
UNWIND nodes AS art
RETURN art.id AS article_id, art.celex AS celex, art.display_ref AS display_ref, art.binding_force AS binding_force
ORDER BY art.hierarchy_depth ASC
LIMIT 20
"""

# Role-aware provision lookup. Starts from one or more ActorRole nodes,
# expands through composite-role and curated equivalence edges, then returns
# obligation-bearing provisions linked to any reachable role.
# Upper bound on obligations returned per role-obligation query. Generous
# enough to carry a role's full statutory article set (the largest, MDR
# manufacturer, has ~26) without dumping every annex fragment; the context
# budget trims any low-relevance tail downstream.
_ROLE_OBLIGATION_CAP = 14

_ROLE_OBLIGATIONS_CYPHER = """\
UNWIND $role_ids AS rid
MATCH (seed:ActorRole {id: rid})
OPTIONAL MATCH (seed)-[:INCLUDES_ROLE]->(included:ActorRole)
OPTIONAL MATCH (seed)-[:EQUIVALENT_ROLE]->(equiv:ActorRole)
WITH collect(DISTINCT seed) + collect(DISTINCT included) + collect(DISTINCT equiv) AS roles
UNWIND roles AS role
WITH DISTINCT role
MATCH (p:Provision)-[:OBLIGATION_OF]->(role)
WHERE p.kind IN ['article', 'annex', 'annex_section', 'annex_subsection', 'annex_point', 'annex_part', 'recital', 'section']
  AND ($target_celexes IS NULL OR p.celex IN $target_celexes)
RETURN DISTINCT
    p.id AS article_id,
    p.kind AS kind,
    p.celex AS celex,
    p.regulation_id AS regulation,
    p.display_ref AS article_ref,
    p.display_path AS article_path,
    p.text_for_analysis AS article_text,
    p.binding_force AS binding_force,
    role.id AS matched_role_id,
    role.term_normalized AS matched_role
ORDER BY (CASE WHEN kind = 'article' THEN 0 ELSE 1 END), article_id
LIMIT 60
"""


class GraphRetriever:
    """Hybrid vector + graph retriever backed by Neo4j.

    On init, loads all embedded provisions into a numpy matrix.
    Queries compute cosine similarity in-memory (~1ms for 6k vectors),
    then expand top-k results via Cypher graph traversal.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self._model = SentenceTransformer(model_name)
        uri = _normalize_neo4j_uri(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        )
        self._driver = GraphDatabase.driver(
            uri,
            auth=(
                os.environ.get(
                    "NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")
                ),
                os.environ.get("NEO4J_PASSWORD", "password"),
            ),
        )
        self._db = os.environ.get("NEO4J_DATABASE", "neo4j")

        # Load the embedding index into memory
        self._ids: list[str] = []
        self._kinds: list[str] = []
        self._path_strings: list[str] = []
        self._celexes: list[str] = []
        self._id_to_kind: dict[str, str] = {}
        self._matrix: np.ndarray | None = None
        self._load_index()

        # Lexical (BM25) channel via a Neo4j full-text index.  Disabled by
        # setting CRSS_LEXICAL=0.  Created idempotently on startup.
        self._lexical_enabled = False
        if os.environ.get("CRSS_LEXICAL", "1") != "0":
            self._ensure_fulltext_index()

        # Cross-encoder reranker (optional).  Disabled by setting
        # CRSS_RERANKER=0.  Model overrideable via CRSS_RERANKER_MODEL.
        self._reranker: Any = None
        if os.environ.get("CRSS_RERANKER", "1") != "0":
            _rr_model = os.environ.get(
                "CRSS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
            )
            try:
                import torch
                from sentence_transformers import CrossEncoder as _CrossEncoder
                _device = (
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu"
                )
                self._reranker = _CrossEncoder(
                    _rr_model, max_length=512, device=_device
                )
                logger.info(
                    "Cross-encoder reranker loaded: %s (device=%s)",
                    _rr_model, _device,
                )
            except Exception as exc:
                logger.warning(
                    "Reranker unavailable (%s) — cosine-only retrieval.", exc
                )

    def _load_index(self) -> None:
        """Fetch all embedded provisions and guidance nodes from Neo4j into a numpy matrix."""
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                "MATCH (n:Provision) "
                "WHERE n.embedding IS NOT NULL "
                "RETURN n.id AS id, n.kind AS kind, "
                "n.path_string AS path_string, n.celex AS celex, "
                "n.embedding AS emb "
                "UNION ALL "
                "MATCH (n:Guidance) "
                "WHERE n.embedding IS NOT NULL "
                "RETURN n.id AS id, n.kind AS kind, "
                "n.path_string AS path_string, n.celex AS celex, "
                "n.embedding AS emb"
            ).data()

        if not rows:
            logger.warning("No embeddings found in Neo4j. Run embed_provisions.py first.")
            return

        self._ids = [r["id"] for r in rows]
        self._kinds = [r["kind"] for r in rows]
        self._path_strings = [r["path_string"] or "" for r in rows]
        self._celexes = [r["celex"] or "" for r in rows]
        self._id_to_kind = dict(zip(self._ids, self._kinds))
        self._id_index = {pid: i for i, pid in enumerate(self._ids)}
        self._matrix = np.array([r["emb"] for r in rows], dtype=np.float32)
        logger.info(
            "Loaded %d provision embeddings (%d dims) into memory.",
            len(self._ids), self._matrix.shape[1],
        )

    def _ensure_fulltext_index(self) -> None:
        """Create the full-text (BM25) index over provision text if absent.

        Idempotent — ``IF NOT EXISTS`` makes re-runs cheap.  The index spans
        both :Provision and :Guidance on ``text_for_analysis`` (which carries
        the context-prefixed ancestor headings, so an annex's title term like
        "TECHNICAL DOCUMENTATION" is searchable) and ``display_ref``.
        """
        try:
            with self._driver.session(database=self._db) as s:
                s.run(
                    f"CREATE FULLTEXT INDEX {_FULLTEXT_INDEX} IF NOT EXISTS "
                    "FOR (n:Provision|Guidance) "
                    "ON EACH [n.text_for_analysis, n.display_ref]"
                )
                # Index population is async; wait so the first query sees it.
                s.run("CALL db.awaitIndexes(30000)")
            self._lexical_enabled = True
            logger.info("Full-text BM25 index ready: %s", _FULLTEXT_INDEX)
        except Exception as exc:
            logger.warning(
                "Full-text index unavailable (%s) — dense-only retrieval.", exc
            )
            self._lexical_enabled = False

    def _lexical_search(
        self,
        question: str,
        limit: int,
        celex_filter: set[str] | None = None,
    ) -> dict[str, int]:
        """Return ``{node_id: rank}`` from the BM25 full-text index (0-based).

        Returns an empty mapping when the lexical channel is disabled or the
        sanitised query is empty, so callers degrade gracefully to dense-only.

        ``celex_filter`` restricts hits to the in-scope regulation(s).  Without
        it, a cross-regulation query's lexically-verbose guidance docs (MDCG)
        can flood the result budget and starve the operative provision the
        query targets.
        """
        if not self._lexical_enabled:
            return {}
        lucene_q = _sanitise_lucene(question)
        if not lucene_q:
            return {}
        try:
            with self._driver.session(database=self._db) as s:
                # Exclude recitals/citations: they are verbose and repeat
                # operative vocabulary (e.g. "AI system"), so BM25 term-density
                # over-promotes them and displaces the terse normative
                # provision that actually answers the query.  The dense channel
                # still surfaces recitals when they are genuinely on-topic.
                rows = s.run(
                    f"CALL db.index.fulltext.queryNodes('{_FULLTEXT_INDEX}', $q) "
                    "YIELD node, score "
                    "WHERE NOT node.kind IN ['recital', 'citation', 'preamble'] "
                    "  AND ($celexes IS NULL OR node.celex IN $celexes) "
                    "RETURN node.id AS id ORDER BY score DESC LIMIT $lim",
                    q=lucene_q,
                    lim=limit,
                    celexes=list(celex_filter) if celex_filter else None,
                ).data()
            return {r["id"]: rank for rank, r in enumerate(rows)}
        except Exception as exc:
            logger.warning("Lexical search failed (%s) — dense-only.", exc)
            return {}

    def encode_as_passage(self, text: str) -> np.ndarray:
        """Encode *text* with the passage prefix, matching the provision embedding space.

        Use this for HyDE (Hypothetical Document Embedding): the generated
        hypothetical answer should be encoded as a passage so it sits in the
        same embedding space as the stored provision embeddings.
        """
        return self._model.encode(
            PASSAGE_PREFIX + text, normalize_embeddings=True,
        ).astype(np.float32)

    def encode_as_query(self, text: str) -> np.ndarray:
        """Encode *text* with the query prefix.

        Use this when you want to embed the raw question for community-level
        retrieval, bypassing HyDE.  Community summaries are semantically rich
        enough to match a plain question vector; using a query prefix (rather
        than a passage prefix) aligns with how the model was trained.
        """
        return self._model.encode(
            QUERY_PREFIX + text, normalize_embeddings=True,
        ).astype(np.float32)

    def _find_anchor(self, node_id: str, node_kind: str, path_string: str) -> str:
        """Return the nearest parent-kind ancestor ID for the given node.

        If the node itself is a parent kind, returns it directly.
        Otherwise walks up path_string (root → immediate parent, "/"-separated)
        to find the closest ancestor whose kind is in _PARENT_KINDS.
        Falls back to node_id if no parent-kind ancestor is found.
        """
        if node_kind in _PARENT_KINDS:
            return node_id
        for ancestor_id in reversed(path_string.split("/") if path_string else []):
            if ancestor_id and self._id_to_kind.get(ancestor_id) in _PARENT_KINDS:
                return ancestor_id
        return node_id

    def _expand_cited_containers(self, results: list[dict]) -> None:
        """Drill into cited provisions that are high-level containers.

        When a CITES edge points to a container node (e.g. "Annex XIV")
        whose own text is very short but has children, this method fetches
        the children's text and appends them to the cited_provisions list
        so the LLM sees the actual substantive content.
        """
        container_ids: list[str] = []
        for r in results:
            for c in r.get("cited_provisions") or []:
                text = c.get("text") or ""
                # Short text + id present → likely a container heading
                if c.get("id") and len(text) <= 80:
                    container_ids.append(c["id"])
        if not container_ids:
            return
        # Deduplicate
        container_ids = list(dict.fromkeys(container_ids))
        with self._driver.session(database=self._db) as s:
            children_rows = s.run(
                _CITED_CHILDREN_CYPHER, ids=container_ids,
            ).data()
        if not children_rows:
            return
        # Group children by container
        children_by_container: dict[str, list[dict]] = {}
        for row in children_rows:
            cid = row["container_id"]
            children_by_container.setdefault(cid, []).append({
                "id": row["id"],
                "kind": row["kind"],
                "ref": row["ref"],
                "text": (row["text"] or "")[:500],
                "binding_force": row.get("binding_force"),
            })
        # Inject children into the cited_provisions entries
        injected = 0
        for r in results:
            cited = r.get("cited_provisions") or []
            extras: list[dict] = []
            for c in cited:
                kids = children_by_container.get(c.get("id", ""))
                if kids:
                    for kid in kids[:8]:
                        extras.append(kid)
                        injected += 1
            if extras:
                cited.extend(extras)
        if injected:
            logger.info(
                "Container drilldown: injected %d children from %d "
                "container(s).",
                injected, len(children_by_container),
            )

    def _rerank(
        self,
        question: str,
        results: list[dict],
        k: int,
    ) -> list[dict]:
        """Refine ranking with the cross-encoder, then return the top-k.

        The cross-encoder score is *blended* with the original cosine score
        rather than replacing it.  Pure cross-encoder ordering tends to bury
        container nodes (e.g. "Annex II") and gate articles (e.g. "Article 43")
        that share vocabulary with their neighbours; blending lets a strong
        cosine match survive while still benefiting from cross-encoder
        precision.  Both scores are min-max normalised within the candidate
        set before blending so they are on a comparable scale.
        """
        pairs: list[tuple[str, str]] = []
        for r in results:
            text = r.get("article_text") or ""
            if len(text) < 200:
                # Container or thin node — enrich with children text so the
                # cross-encoder has enough signal (e.g. Annex II title alone
                # is unscoreable; its sub-items carry the actual content).
                children_text = " ".join(
                    (c.get("text") or "")[:400]
                    for c in (r.get("children") or [])[:8]
                ).strip()
                text = (text + " " + children_text).strip() or text
            pairs.append((question, text[:_RERANK_MAX_LEN]))

        rr_scores = [
            float(s) for s in self._reranker.predict(pairs, show_progress_bar=False)
        ]
        cos_scores = [float(r.get("score", 0.0)) for r in results]

        def _normalise(xs: list[float]) -> list[float]:
            lo, hi = min(xs), max(xs)
            span = hi - lo
            if span < 1e-9:
                return [1.0 for _ in xs]
            return [(x - lo) / span for x in xs]

        rr_norm = _normalise(rr_scores)
        cos_norm = _normalise(cos_scores)
        for r, rr_raw, rr_n, cos_n in zip(results, rr_scores, rr_norm, cos_norm):
            r["rerank_score"] = rr_raw
            r["_blended_score"] = (
                _RERANK_WEIGHT * rr_n + (1.0 - _RERANK_WEIGHT) * cos_n
            )
        results.sort(key=lambda r: r.get("_blended_score", 0.0), reverse=True)
        return results[:k]

    def retrieve(
        self,
        question: str,
        k: int = 5,
        target_celexes: set[str] | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k provisions with children and cross-references.

        Parameters
        ----------
        target_celexes:
            If provided, filters results to the given CELEX(es).  When more
            than one CELEX is given, allocates slots per regulation so each
            regulation gets adequate coverage.
        query_vec:
            Optional pre-computed query vector (e.g. from HyDE).  When
            provided the *question* string is used only for logging; encoding
            is skipped.
        """
        if self._matrix is None or len(self._ids) == 0:
            return []

        # Widen the cosine candidate pool when a reranker is active so the
        # cross-encoder gets enough candidates to discriminate.
        candidate_k = (
            min(k * _CANDIDATE_MULTIPLIER, _CANDIDATE_CAP)
            if self._reranker is not None
            else k
        )

        # Encode query (or use a pre-computed vector, e.g. from HyDE)
        if query_vec is not None:
            q_vec = query_vec
        else:
            q_vec = self._model.encode(
                QUERY_PREFIX + question, normalize_embeddings=True
            ).astype(np.float32)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = self._matrix @ q_vec

        # ------------------------------------------------------------------
        # Reciprocal Rank Fusion of the dense (cosine) and lexical (BM25)
        # channels.  Dense cosine alone smears legally-distinct but
        # vocabulary-similar provisions into a near-tie (e.g. Annex II
        # "Technical Documentation" vs Annex IX "assessment of technical
        # documentation").  A BM25 ranking lets an exact heading/term match
        # break the tie.  RRF fuses *ranks*, not scores, so the two channels'
        # incompatible score scales need no calibration.
        # ------------------------------------------------------------------
        dense_order = sorted(
            range(len(self._ids)), key=lambda i: scores[i], reverse=True,
        )
        dense_rank = {idx: rank for rank, idx in enumerate(dense_order)}

        # Pull a generous lexical pool so BM25 hits outside the dense top
        # still contribute their rank to the fusion.  Filter to the in-scope
        # regulation(s) so cross-reg guidance can't starve the budget.
        lexical_rank = self._lexical_search(
            question, candidate_k * 8, celex_filter=target_celexes,
        )

        def _fused_score(idx: int) -> float:
            f = 1.0 / (_RRF_K + dense_rank[idx])
            nid = self._ids[idx]
            if nid in lexical_rank:
                f += _LEXICAL_WEIGHT / (_RRF_K + lexical_rank[nid])
            return f

        fused_order = sorted(
            range(len(self._ids)), key=_fused_score, reverse=True,
        )

        # Walk the fused ranking, mapping each embedded node to its nearest
        # parent-kind ancestor. This lets deep leaf nodes (point, roman_item)
        # surface their parent article/section with the most specific match.
        score_map: dict[str, float] = {}
        leaf_map: dict[str, str] = {}

        if target_celexes and len(target_celexes) > 1:
            # Multi-regulation mode: allocate slots per regulation to
            # guarantee coverage of each mentioned regulation.
            # Use ceiling division so k=6 across 2 regs → 3 each (not 2).
            per_reg = max(3, -(-candidate_k // len(target_celexes)))  # ceiling division
            per_reg_count: dict[str, int] = {c: 0 for c in target_celexes}

            for idx in fused_order:
                anchor_id = self._find_anchor(
                    self._ids[idx], self._kinds[idx], self._path_strings[idx],
                )
                if anchor_id in score_map:
                    continue
                # Determine celex of the anchor
                anchor_celex = self._celexes[idx]
                for c in target_celexes:
                    if anchor_id.startswith(c):
                        anchor_celex = c
                        break

                # Only include results from target regulations
                if anchor_celex not in per_reg_count:
                    continue
                if per_reg_count[anchor_celex] >= per_reg:
                    continue
                per_reg_count[anchor_celex] += 1

                score_map[anchor_id] = _fused_score(idx)
                leaf_map[anchor_id] = self._ids[idx]

                if sum(per_reg_count.values()) >= candidate_k:
                    break
        else:
            # Standard mode (single regulation or no filter)
            for idx in fused_order:
                # If a celex filter is set (single-regulation question),
                # skip embeddings from other regulations so they don't consume
                # top-k slots.
                if target_celexes and self._celexes[idx] not in target_celexes:
                    continue
                anchor_id = self._find_anchor(
                    self._ids[idx], self._kinds[idx], self._path_strings[idx],
                )
                if anchor_id not in score_map:
                    score_map[anchor_id] = _fused_score(idx)
                    leaf_map[anchor_id] = self._ids[idx]
                if len(score_map) >= candidate_k:
                    break

        top_ids = list(score_map.keys())

        # Graph expansion via Cypher
        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=top_ids).data()

        # Attach cosine scores and matched leaf IDs
        for r in results:
            r["score"] = score_map.get(r["article_id"], 0.0)
            r["matched_leaf_id"] = leaf_map.get(r["article_id"])

        # Rerank the widened candidate pool to the final top-k, or fall back
        # to cosine ordering when no reranker is loaded.
        if self._reranker is not None and len(results) > k:
            if target_celexes and len(target_celexes) > 1:
                # Multi-regulation: rerank within each regulation's slot so the
                # cross-encoder cannot make one regulation crowd out another.
                per_reg = max(1, -(-k // len(target_celexes)))  # ceiling div
                reranked: list[dict] = []
                for celex in target_celexes:
                    bucket = [r for r in results if r.get("celex") == celex]
                    if len(bucket) > per_reg:
                        bucket = self._rerank(question, bucket, per_reg)
                    else:
                        bucket.sort(
                            key=lambda r: r.get("rerank_score", r.get("score", 0.0)),
                            reverse=True,
                        )
                    reranked.extend(bucket)
                results = reranked
            else:
                results = self._rerank(question, results, k)
        else:
            results.sort(key=lambda r: r["score"], reverse=True)

        # Use only the reranked top-k IDs for cross-reg expansion, so the
        # widened pool doesn't pollute the reverse-xref budget.
        reranked_ids = [r["article_id"] for r in results]

        # Drill into cited container nodes (e.g. "Annex XIV" headings)
        # to surface their children's text for the LLM.
        self._expand_cited_containers(results)

        # Cross-regulation expansion: if the retrieved provisions have
        # cross-regulation CITES links, also retrieve the "other side"
        # provisions from the other regulation(s).  This ensures that
        # when a question spans multiple regulations, both sides of the
        # cross-reference chain appear in context.
        if not target_celexes:
            with self._driver.session(database=self._db) as s:
                reverse = s.run(
                    _REVERSE_XREF_CYPHER, ids=reranked_ids
                ).data()
            if reverse:
                rev_ids = [r["article_id"] for r in reverse]
                with self._driver.session(database=self._db) as s:
                    rev_expanded = s.run(
                        _EXPAND_CYPHER, ids=rev_ids
                    ).data()
                for r in rev_expanded:
                    r["score"] = 0.0
                    r["matched_leaf_id"] = None
                    r["_cross_reg_expansion"] = True
                results.extend(rev_expanded)

        return results

    def get_defined_terms_index(self) -> dict[str, str]:
        """Return ``{lowercase_term: term_normalized}`` for all DefinedTerm nodes.

        The index is fetched once from Neo4j and cached for the lifetime of
        the retriever instance.  Used by the agent to detect which regulatory
        terms appear in a user question so it can enrich context with their
        legal definitions.
        """
        if not hasattr(self, "_term_index"):
            with self._driver.session(database=self._db) as s:
                rows = s.run(
                    "MATCH (d:DefinedTerm) "
                    "RETURN d.term AS term, d.term_normalized AS tn"
                ).data()
            self._term_index: dict[str, str] = {
                r["term"].lower(): r["tn"] for r in rows
            }
        return self._term_index

    def find_by_term(self, term: str) -> list[dict[str, Any]]:
        """Exact-match lookup for a :DefinedTerm by its normalized term name.

        Normalises *term* the same way the extraction pipeline does
        (lowercased, whitespace → underscore) and performs a direct Neo4j
        lookup — no embedding needed.

        For each match the full :Point provision that defines the term is
        returned alongside its parent article for context.

        Parameters
        ----------
        term:
            The term to look up, e.g. ``"provider"``, ``"AI system"``, or
            ``"high-risk AI system"``.

        Returns
        -------
        list of result dicts, each containing:
          ``term``, ``term_normalized``, ``category``, ``regulation``,
          ``celex``, ``definition_text``, ``source_provision_id``,
          ``article_ref``, ``article_path``
        """
        # Mirror the normalisation used in definitions.py
        import re as _re
        term_normalized = _re.sub(r"\s+", "_", term.strip().lower())

        cypher = """\
MATCH (d:DefinedTerm {term_normalized: $term_normalized})
MATCH (d)-[:DEFINED_BY]->(p:Provision)
OPTIONAL MATCH (p)<-[:HAS_PART]-(art:Provision)
RETURN
    d.term                AS term,
    d.term_normalized     AS term_normalized,
    d.category            AS category,
    d.definition_type     AS definition_type,
    d.regulation          AS regulation,
    d.celex               AS celex,
    p.text                AS definition_text,
    p.id                  AS source_provision_id,
    art.display_ref       AS article_ref,
    art.display_path      AS article_path
"""
        with self._driver.session(database=self._db) as s:
            return s.run(cypher, term_normalized=term_normalized).data()

    def find_by_category(self, category: str, celex: str | None = None) -> list[dict[str, Any]]:
        """Return all :DefinedTerm nodes for a given semantic category.

        Parameters
        ----------
        category:
            One of ``"actor"``, ``"system"``, ``"data"``, ``"document"``,
            ``"process"``, ``"concept"``, ``"body"``, or ``"other"``.
        celex:
            Optional CELEX filter (e.g. ``"32024R1689"`` for AI Act only).

        Returns
        -------
        list of dicts with ``term``, ``category``, ``regulation``, ``celex``,
        ``source_provision_id``.
        """
        if celex:
            cypher = """\
MATCH (d:DefinedTerm {category: $category, celex: $celex})
RETURN d.term AS term, d.term_normalized AS term_normalized,
       d.category AS category, d.regulation AS regulation,
       d.celex AS celex, d.source_provision_id AS source_provision_id
ORDER BY d.term_normalized
"""
            with self._driver.session(database=self._db) as s:
                return s.run(cypher, category=category, celex=celex).data()
        else:
            cypher = """\
MATCH (d:DefinedTerm {category: $category})
RETURN d.term AS term, d.term_normalized AS term_normalized,
       d.category AS category, d.regulation AS regulation,
       d.celex AS celex, d.source_provision_id AS source_provision_id
ORDER BY d.celex, d.term_normalized
"""
            with self._driver.session(database=self._db) as s:
                return s.run(cypher, category=category).data()

    def retrieve_by_refs(
        self,
        refs: list[str],
        celex_filter: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Direct lookup of provisions by their display_ref (e.g. 'Annex I', 'Article 26').

        Performs an exact case-insensitive match against ``display_ref``,
        completely bypassing vector similarity.  Ideal for structural /
        navigational questions that explicitly name a provision.

        The matched provisions are then expanded via the same Cypher graph
        traversal used by :meth:`retrieve`, so children and cross-references
        are included in the result.

        Parameters
        ----------
        refs:
            Normalised provision references extracted from the question,
            e.g. ``['Annex I', 'Article 26']``.
        celex_filter:
            Optional set of CELEX codes to restrict results to.
        """
        if not refs:
            return []

        with self._driver.session(database=self._db) as s:
            rows = s.run(_DIRECT_REF_CYPHER, refs=refs).data()

        if celex_filter:
            rows = [r for r in rows if r["celex"] in celex_filter]

        # Deduplicate, prefer the shortest display_ref (outermost/parent node)
        seen: set[str] = set()
        top_ids: list[str] = []
        for row in rows:
            art_id = row["article_id"]
            if art_id not in seen:
                seen.add(art_id)
                top_ids.append(art_id)

        if not top_ids:
            return []

        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=top_ids).data()

        for r in results:
            r["score"] = 1.0  # perfect score — explicit structural match
            r["matched_leaf_id"] = None
            r["_direct_ref_match"] = True

        return results

    def retrieve_by_roles(
        self,
        role_specs: list[tuple[str, str]],
        *,
        k: int = 8,
        query_vec: np.ndarray | None = None,
        target_celexes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the obligations of one or more resolved actor roles.

        The role's ``OBLIGATION_OF`` set *is* the statutory answer to a
        role-obligation question, so it is treated as a guaranteed candidate set
        rather than a vector top-k: article-grained obligations are preferred
        over annex/recital fragments, and — when ``query_vec`` is supplied — the
        set is ranked by cosine relevance to the question so the on-point
        articles (e.g. Article 53 for "GPAI provider obligations") lead and
        survive trimming. This replaces the previous "first ``k`` in arbitrary
        Neo4j order" selection, which silently dropped relevant obligations.

        Parameters
        ----------
        role_specs:
            ``[(term_normalized, celex), ...]`` pairs resolved by the agent.
        k:
            Lower bound on the obligation budget. The effective cap is
            ``max(k, _ROLE_OBLIGATION_CAP)`` so a small caller ``k`` (e.g. the
            eval's k=8) never truncates a role's complete article set.
        query_vec:
            Optional query embedding (``encode_as_query``). When given, ranks
            obligations by cosine relevance; otherwise falls back to the
            Cypher's article-first ordering.
        target_celexes:
            When provided, drops obligations outside these regulations. The
            Cypher expands ``EQUIVALENT_ROLE`` / ``INCLUDES_ROLE`` across
            regulations (e.g. AI-Act ``provider`` ≡ MDR ``manufacturer``), which
            is wanted only when the question actually spans them; scoping keeps a
            single-regulation question from inheriting another reg's obligations.
            The filter is applied *inside* the Cypher (before its ``LIMIT``) so
            in-scope obligations are never crowded out of the row cap by
            cross-regulation ones (MDR ids sort ahead of AI-Act ids).
        """
        if not role_specs:
            return []

        role_ids = [f"{celex}::role::{term_normalized}" for term_normalized, celex in role_specs]
        celex_param = sorted(target_celexes) if target_celexes else None
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                _ROLE_OBLIGATIONS_CYPHER, role_ids=role_ids, target_celexes=celex_param,
            ).data()

        if not rows:
            return []

        # Dedup by provision id, first occurrence wins (preserves the Cypher's
        # article-first ordering as the no-query fallback). Cross-regulation
        # scoping already happened in the Cypher.
        role_hits: dict[str, tuple[str | None, str | None]] = {}
        uniq: list[dict[str, Any]] = []
        for row in rows:
            art_id = row["article_id"]
            if art_id not in role_hits:
                role_hits[art_id] = (row.get("matched_role_id"), row.get("matched_role"))
                uniq.append(row)

        def _relevance(row: dict[str, Any]) -> float:
            if query_vec is None:
                return 0.0
            idx = self._id_index.get(row["article_id"])
            if idx is None or self._matrix is None:
                return -1.0  # no embedding: keep, but rank last
            return float(self._matrix[idx] @ query_vec)

        articles = sorted(
            (r for r in uniq if r.get("kind") == "article"),
            key=_relevance, reverse=True,
        )
        others = sorted(
            (r for r in uniq if r.get("kind") != "article"),
            key=_relevance, reverse=True,
        )

        # Article-grained obligations lead; annex/recital fragments fill the
        # remaining budget. The cap is generous and floored so a role's full
        # article set is never crowded out by a small caller k.
        cap = max(k, _ROLE_OBLIGATION_CAP)
        top_rows = (articles + others)[:cap]
        top_ids = [r["article_id"] for r in top_rows]

        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=top_ids).data()

        for r in results:
            matched_role_id, matched_role = role_hits.get(r["article_id"], (None, None))
            r["score"] = 1.0
            r["matched_leaf_id"] = None
            r["matched_role_id"] = matched_role_id
            r["matched_role"] = matched_role
            r["_role_retrieval"] = True

        return results

    # ------------------------------------------------------------------
    # Community-based (summary-first) retrieval
    # ------------------------------------------------------------------

    def _load_community_index(self) -> None:
        """Lazily fetch Community summaries into an in-memory numpy matrix.

        Loads **Level-0** community embeddings into the cosine-similarity
        matrix used for community-first retrieval.  Level-1 (chapter-level)
        summaries are stored separately in ``_l1_community_summaries`` and
        used for the map-reduce pass.
        """
        if hasattr(self, "_community_ids"):
            return
        with self._driver.session(database=self._db) as s:
            rows = s.run(
                "MATCH (c:Community) "
                "WHERE c.summary_embedding IS NOT NULL "
                "  AND (c.level IS NULL OR c.level = 0) "
                "RETURN c.id AS id, c.summary_text AS summary_text, "
                "c.member_count AS member_count, "
                "c.regulations AS regulations, "
                "c.summary_embedding AS emb"
            ).data()
            l1_rows = s.run(
                "MATCH (c:Community) "
                "WHERE c.level = 1 AND c.summary_text IS NOT NULL "
                "RETURN c.id AS id, c.summary_text AS summary_text, "
                "c.regulations AS regulations, c.label AS label, "
                "c.member_count AS member_count, "
                "c.summary_embedding AS emb"
            ).data()
        if not rows:
            self._community_ids: list[str] = []
            self._community_meta: list[dict] = []
            self._community_matrix: np.ndarray | None = None
            logger.warning(
                "No community summary embeddings found. "
                "Run scripts/generate_community_summaries.py first."
            )
        else:
            self._community_ids = [r["id"] for r in rows]
            self._community_meta = [
                {
                    "id": r["id"],
                    "summary_text": r.get("summary_text") or "",
                    "member_count": r.get("member_count") or 0,
                    "regulations": r.get("regulations") or [],
                }
                for r in rows
            ]
            self._community_matrix = np.array(
                [r["emb"] for r in rows], dtype=np.float32
            )
            logger.info(
                "Loaded %d Level-0 community summary embeddings into memory.",
                len(self._community_ids),
            )

        # Level-1 summaries (no embedding search needed — map-reduce uses all)
        self._l1_community_summaries: list[dict] = [
            {
                "id": r["id"],
                "summary_text": r.get("summary_text") or "",
                "regulations": r.get("regulations") or [],
                "label": r.get("label") or r["id"],
                "member_count": r.get("member_count") or 0,
            }
            for r in l1_rows
        ]
        if l1_rows:
            logger.info(
                "Loaded %d Level-1 chapter-community summaries.",
                len(l1_rows),
            )

    def retrieve_by_communities_hierarchical(
        self,
        question: str,
        *,
        k_communities: int = 5,
        k_provisions: int = 20,
        target_celexes: set[str] | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Community-first retrieval: search summaries, then fetch member provisions.

        Two-stage search:
        1. Encode question (or use *query_vec*) → cosine similarity against
           ~300 community summary embeddings → top-``k_communities``.
        2. For each matched community, fetch up to ``k_provisions`` member
           Provision nodes (anchored to parent kinds), then run the same
           graph expansion used by :meth:`retrieve`.

        Returns the same dict shape as :meth:`retrieve` so results can be
        merged transparently.  Each provision dict gains two extra keys:

        * ``community_id`` — the community it came from
        * ``_community_retrieval`` — ``True`` for attribution in audit traces
        * ``community_summary`` — the community's summary text

        Falls back to an empty list when no community embeddings are loaded.
        """
        self._load_community_index()
        if not self._community_ids or self._community_matrix is None:
            return []

        if query_vec is not None:
            q_vec = query_vec
        else:
            q_vec = self._model.encode(
                QUERY_PREFIX + question, normalize_embeddings=True
            ).astype(np.float32)

        # Step 1 — rank communities by summary similarity
        scores = self._community_matrix @ q_vec
        sorted_indices = scores.argsort()[::-1]

        top_community_ids: list[str] = []
        top_community_meta: dict[str, dict] = {}
        for idx in sorted_indices:
            cid = self._community_ids[idx]
            meta = self._community_meta[idx]
            # Optional CELEX filter: include community if any of its
            # regulations match, or if it has no regulation tagging.
            if target_celexes:
                regs = set(meta.get("regulations") or [])
                if regs and not regs.intersection(target_celexes):
                    continue
            top_community_ids.append(cid)
            top_community_meta[cid] = {**meta, "score": float(scores[idx])}
            if len(top_community_ids) >= k_communities:
                break

        if not top_community_ids:
            return []

        # Step 2 — fetch member provisions for matched communities
        per_community = max(1, k_provisions // len(top_community_ids))
        member_query = """\
UNWIND $community_ids AS cid
MATCH (p:Provision)-[:MEMBER_OF]->(c:Community {id: cid})
WHERE p.kind IN ['article', 'annex_section', 'annex_subsection', 'annex_point',
                  'annex_part', 'recital', 'section',
                  'guidance_section', 'guidance_subsection']
  AND p.text_for_analysis IS NOT NULL
WITH cid, p
ORDER BY p.hierarchy_depth ASC
WITH cid, collect(p.id)[..$per_community] AS ids
UNWIND ids AS pid
RETURN pid AS provision_id, cid AS community_id
"""
        with self._driver.session(database=self._db) as s:
            member_rows = s.run(
                member_query,
                community_ids=top_community_ids,
                per_community=per_community,
            ).data()

        if not member_rows:
            return []

        provision_to_community: dict[str, str] = {
            row["provision_id"]: row["community_id"] for row in member_rows
        }
        top_ids = list(dict.fromkeys(provision_to_community))  # preserves order, deduplicates

        # Step 3 — graph expansion (same as retrieve())
        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=top_ids).data()

        # Attach community metadata to each result
        for r in results:
            cid = provision_to_community.get(r["article_id"], "")
            meta = top_community_meta.get(cid, {})
            r["community_id"] = cid
            r["community_summary"] = meta.get("summary_text", "")
            r["score"] = meta.get("score", 0.0)
            r["matched_leaf_id"] = None
            r["_community_retrieval"] = True

        self._expand_cited_containers(results)
        return results

    def get_all_community_summaries(self, *, level: int = 1) -> list[dict]:
        """Return all Community summaries for the given level.

        Used by the map-reduce pass in the community_summary_search route.
        Each dict has keys: ``id``, ``summary_text``, ``regulations``, ``label``.

        Level-0 results include summaries loaded at startup.
        Level-1 results are the chapter-level aggregation summaries.
        """
        self._load_community_index()
        if level == 1:
            return list(self._l1_community_summaries)
        # level == 0: return from the standard community meta list
        return [
            {
                "id": m["id"],
                "summary_text": m["summary_text"],
                "regulations": m["regulations"],
                "label": m["id"],
            }
            for m in self._community_meta
        ]

    # ------------------------------------------------------------------
    # Legal reasoning chain retrieval
    # ------------------------------------------------------------------

    def retrieve_by_chain(
        self,
        refs: list[str],
        celex: str,
        *,
        seed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve provisions reachable via legal reasoning edges.

        Given one or more seed provision references (e.g. ``["Article 6"]``),
        this method:

        1. Resolves the seed nodes in Neo4j by ``display_ref`` + ``celex``.
        2. Traverses ``TRIGGERS_OBLIGATION_CLUSTER`` and
           ``IS_PREREQUISITE_FOR`` edges up to 2 hops.
        3. Returns the expanded graph context for all reachable provisions
           (same structure as :meth:`retrieve`).

        Falls back to :meth:`retrieve_by_refs` when no legal reasoning edges
        exist for the seed provisions (e.g. for newly loaded regulations
        before the edge loading script has run).

        Parameters
        ----------
        refs:
            List of ``display_ref`` strings to start from.
        celex:
            CELEX of the regulation.
        seed_only:
            When ``True``, skip the chain traversal and return only the seed
            provisions' expanded context.  Useful for provisions that are
            already a complete obligation cluster (e.g. ``Article 26``).
        """
        if not refs:
            return []

        with self._driver.session(database=self._db) as s:
            seed_rows = s.run(
                _CHAIN_SEED_LOOKUP_CYPHER, refs=refs, celex=celex,
            ).data()

        seed_ids = [r["seed_id"] for r in seed_rows]
        if not seed_ids:
            logger.debug(
                "retrieve_by_chain: no seed nodes found for %s in %s — "
                "falling back to retrieve_by_refs",
                refs, celex,
            )
            return self.retrieve_by_refs(refs, celex_filter={celex})

        if seed_only:
            linked_ids = seed_ids
        else:
            with self._driver.session(database=self._db) as s:
                chain_rows = s.run(
                    _CHAIN_TRAVERSE_CYPHER, seed_ids=seed_ids,
                ).data()

            linked_ids = [r["article_id"] for r in chain_rows if r["article_id"]]
            if not linked_ids:
                logger.debug(
                    "retrieve_by_chain: no linked provisions found for %s — "
                    "returning seed provisions only",
                    seed_ids,
                )
                linked_ids = seed_ids

        # Merge: seeds first, then chain-linked provisions
        all_ids = list(dict.fromkeys(seed_ids + linked_ids))

        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=all_ids).data()

        for r in results:
            is_seed = r["article_id"] in set(seed_ids)
            r["score"] = 1.0 if is_seed else 0.9
            r["matched_leaf_id"] = None
            r["_chain_retrieval"] = True
            r["_chain_seed"] = is_seed

        self._expand_cited_containers(results)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
