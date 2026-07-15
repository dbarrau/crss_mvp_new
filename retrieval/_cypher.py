"""Cypher queries and kind sets for graph-aware retrieval.

Pure data — no logic.  Every query here is executed by the facade
(:mod:`retrieval.graph_retriever`) or the traversal helpers
(:mod:`retrieval._traversal`).
"""
from __future__ import annotations

# Parent-level kinds we want the vector search to return.
# annex_point is included so that deep annex leaves (subpoints, bullets)
# anchor to a specific numbered point rather than a broad subsection.
_PARENT_KINDS = frozenset({
    "article", "annex_section", "annex_subsection", "annex_point",
    "annex_part", "annex_chapter",
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
     })[..60] AS children

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
                       'annex_point', 'annex_part', 'annex_chapter',
                       'recital', 'section']
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
  AND p1.kind IN ['article', 'annex_section', 'annex_part', 'annex_chapter',
                  'annex', 'recital', 'section', 'chapter', 'title']
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
  AND linked.kind IN ['article', 'annex_section', 'annex_part', 'annex_chapter',
                      'annex', 'recital', 'section']
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
#
# The CELEX filter and the row cap are both applied *inside* the query, per
# ref: display_ref is heavily duplicated across regulations and depths (e.g.
# "Paragraph 1" matches hundreds of nodes), so filtering in Python after a
# global LIMIT could evict every in-scope node before the filter ever ran,
# and one ambiguous ref could starve the others' budget.
_DIRECT_REF_CYPHER = """\
UNWIND $refs AS ref
CALL {
  WITH ref
  OPTIONAL MATCH (p1:Provision)
    WHERE toLower(p1.display_ref) = toLower(ref)
      AND ($celexes IS NULL OR p1.celex IN $celexes)
  OPTIONAL MATCH (p2:Guidance)
    WHERE toLower(p2.display_ref) = toLower(ref)
      AND ($celexes IS NULL OR p2.celex IN $celexes)
  WITH collect(DISTINCT p1) + collect(DISTINCT p2) AS nodes
  UNWIND nodes AS art
  WITH art WHERE art IS NOT NULL
  RETURN art
  ORDER BY art.hierarchy_depth ASC
  LIMIT 8
}
RETURN art.id AS article_id, art.celex AS celex, art.display_ref AS display_ref, art.binding_force AS binding_force
ORDER BY art.hierarchy_depth ASC
"""

# Ordered subtree of a directly-looked-up provision, for faithful structural
# rendering.  Returns the root plus every HAS_PART descendant with its OWN text
# (node.text, never the flattened text_for_analysis), its exact display_ref, and
# a depth for indentation.  Ordering by the list of edge `order` values along the
# path yields a pre-order depth-first walk, i.e. the provision exactly as it is
# numbered in the source document — so "Article 53(1), point (b)(i)" renders as a
# real, referenceable unit nested under its chapeau rather than as run-on prose.
_SUBTREE_CYPHER = """\
UNWIND $ids AS rootId
MATCH (root {id: rootId})
CALL {
  WITH root
  RETURN root AS node, 0 AS depth, [] AS ord
  UNION
  WITH root
  MATCH path = (root)-[:HAS_PART*1..6]->(desc)
  WHERE desc.text IS NOT NULL
  RETURN desc AS node, length(path) AS depth,
         [r IN relationships(path) | r.order] AS ord
}
RETURN rootId          AS root_id,
       node.id         AS id,
       node.display_ref AS ref,
       node.number     AS number,
       node.kind       AS kind,
       node.text       AS text,
       depth
ORDER BY rootId, ord
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
WHERE p.kind IN ['article', 'annex', 'annex_section', 'annex_subsection', 'annex_point', 'annex_part', 'annex_chapter', 'recital', 'section']
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
LIMIT 400
"""
# The LIMIT above is only a runaway guard: it must clear the largest role's
# complete OBLIGATION_OF set (MDR manufacturer, ~230 edges) so the *relevance*
# ranking in Python — not this query's lexicographic id order — decides which
# obligations survive the downstream cap.  A tight LIMIT here silently
# pre-trimmed large roles (GDPR controller/supervisory authority, MDR
# manufacturer) by id sort before the query vector ever saw them.
