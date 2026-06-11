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

# Parent-level kinds we want the vector search to return.
# annex_point is included so that deep annex leaves (subpoints, bullets)
# anchor to a specific numbered point rather than a broad subsection.
_PARENT_KINDS = frozenset({
    "article", "annex_section", "annex_subsection", "annex_point",
    "annex_part",
    "recital", "section",
    # Guidance (MDCG)
    "guidance_section", "guidance_subsection",
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
  cross_reg_cited
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
OPTIONAL MATCH (seed)-[:TRIGGERS_OBLIGATION_CLUSTER|IS_PREREQUISITE_FOR*1..2]->(linked)
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
RETURN DISTINCT
    p.id AS article_id,
    p.celex AS celex,
    p.regulation_id AS regulation,
    p.display_ref AS article_ref,
    p.display_path AS article_path,
    p.text_for_analysis AS article_text,
    p.binding_force AS binding_force,
    role.id AS matched_role_id,
    role.term_normalized AS matched_role
LIMIT 40
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
        self._matrix = np.array([r["emb"] for r in rows], dtype=np.float32)
        logger.info(
            "Loaded %d provision embeddings (%d dims) into memory.",
            len(self._ids), self._matrix.shape[1],
        )

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

        # Encode query (or use a pre-computed vector, e.g. from HyDE)
        if query_vec is not None:
            q_vec = query_vec
        else:
            q_vec = self._model.encode(
                QUERY_PREFIX + question, normalize_embeddings=True
            ).astype(np.float32)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = self._matrix @ q_vec

        # Score all embedded nodes, then map each to its nearest parent-kind
        # ancestor. This lets deep leaf nodes (point, roman_item) surface their
        # parent article/section with a score driven by the most specific match.
        score_map: dict[str, float] = {}
        leaf_map: dict[str, str] = {}

        if target_celexes and len(target_celexes) > 1:
            # Multi-regulation mode: allocate slots per regulation to
            # guarantee coverage of each mentioned regulation.
            # Use ceiling division so k=6 across 2 regs → 3 each (not 2).
            per_reg = max(3, -(-k // len(target_celexes)))  # ceiling division
            per_reg_count: dict[str, int] = {c: 0 for c in target_celexes}

            for idx, sc in sorted(
                enumerate(scores.tolist()), key=lambda x: x[1], reverse=True,
            ):
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

                score_map[anchor_id] = float(sc)
                leaf_map[anchor_id] = self._ids[idx]

                if sum(per_reg_count.values()) >= k:
                    break
        else:
            # Standard mode (single regulation or no filter)
            for idx, sc in sorted(
                enumerate(scores.tolist()), key=lambda x: x[1], reverse=True,
            ):
                # If a celex filter is set (single-regulation question),
                # skip embeddings from other regulations so they don't consume
                # top-k slots.
                if target_celexes and self._celexes[idx] not in target_celexes:
                    continue
                anchor_id = self._find_anchor(
                    self._ids[idx], self._kinds[idx], self._path_strings[idx],
                )
                if anchor_id not in score_map:
                    score_map[anchor_id] = float(sc)
                    leaf_map[anchor_id] = self._ids[idx]
                if len(score_map) >= k:
                    break

        top_ids = list(score_map.keys())

        # Graph expansion via Cypher
        with self._driver.session(database=self._db) as s:
            results = s.run(_EXPAND_CYPHER, ids=top_ids).data()

        # Attach scores, matched leaf, and sort
        for r in results:
            r["score"] = score_map.get(r["article_id"], 0.0)
            r["matched_leaf_id"] = leaf_map.get(r["article_id"])
        results.sort(key=lambda r: r["score"], reverse=True)

        # Drill into cited container nodes (e.g. "Annex XIV" headings)
        # to surface their children's text for the LLM.
        self._expand_cited_containers(results)

        # Cross-regulation expansion: if the retrieved provisions have
        # cross-regulation CITES links, also retrieve the "other side"
        # provisions from the other regulation(s).  This ensures that
        # when a question spans multiple regulations, both sides of the
        # cross-reference chain appear in context.
        if not target_celexes:
            # Try reverse xref to expand cross-references even if multiple regs are hit
            with self._driver.session(database=self._db) as s:
                reverse = s.run(
                    _REVERSE_XREF_CYPHER, ids=top_ids
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
    ) -> list[dict[str, Any]]:
        """Return provisions linked to one or more resolved actor roles.

        Parameters
        ----------
        role_specs:
            ``[(term_normalized, celex), ...]`` pairs resolved by the agent.
        k:
            Maximum number of expanded parent provisions to return.
        """
        if not role_specs:
            return []

        role_ids = [f"{celex}::role::{term_normalized}" for term_normalized, celex in role_specs]
        with self._driver.session(database=self._db) as s:
            rows = s.run(_ROLE_OBLIGATIONS_CYPHER, role_ids=role_ids).data()

        if not rows:
            return []

        top_ids: list[str] = []
        role_hits: dict[str, tuple[str | None, str | None]] = {}
        for row in rows:
            art_id = row["article_id"]
            if art_id not in role_hits:
                role_hits[art_id] = (row.get("matched_role_id"), row.get("matched_role"))
                top_ids.append(art_id)
            if len(top_ids) >= k:
                break

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
