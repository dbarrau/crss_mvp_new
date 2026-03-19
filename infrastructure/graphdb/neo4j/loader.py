"""
infrastructure/graphdb/neo4j/loader.py
=======================================
Loads EU regulation ``parsed.json`` files into a Neo4j graph database.

Graph model (structural / hierarchical only)
--------------------------------------------
Nodes – 15 legal-structural labels
    (:Document)           – regulation root
  (:Citation)
  (:Recital)
  (:Chapter)
  (:Section)
  (:Article)
  (:Paragraph)
  (:Subparagraph)       – ordinal subparagraph within a paragraph
  (:Point)              – includes ``roman_item`` provisions
  (:Annex)
  (:AnnexChapter)       – CHAPTER I, II, III inside an annex
  (:AnnexSection)       – numbered section headings (10., 10.4., etc.)
  (:AnnexPoint)         – numbered requirements (10.1., 10.2., etc.)
  (:AnnexSubpoint)      – lettered sub-items (a), (b), (c)
  (:AnnexBullet)        – dash/bullet items

Editorial containers (``preamble``, ``enacting_terms``, ``final_provisions``,
``annexes``) are **not** loaded as graph nodes.  Their children are
re-parented directly under the Document node.

Edges
  (parent)-[:HAS_PART {order}]->(child)   ordered containment (traverse parent with <-[:HAS_PART]-)

Node properties
    id               – unique provision ID
    celex            – CELEX identifier  (e.g. "32024R1689")
    regulation_id    – human name        (e.g. "EU AI Act")
    lang             – language code     (e.g. "EN")
    kind             – raw kind field    (e.g. "article")
    level            – normalized structural level (currently same as ``kind``)
    text             – full text content
    hierarchy_depth  – integer depth from root (0 = document)
    number           – item number       (e.g. "1", "I", "a")
    title            – optional heading
    path_string      – "/"-joined ancestor IDs
    display_ref      – human-friendly structural reference (e.g. "Article 72")
    display_path     – human-friendly path (e.g. "Chapter IX / Section 1 / Article 72")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase, Driver

from domain.regulations_catalog import REGULATIONS

logger = logging.getLogger(__name__)

# Map regulation numbers (e.g. "2017/745") to CELEX IDs for loaded regs.
# Used to avoid creating ExternalAct stubs for regulations already in the graph.
# Derived from the single source of truth in domain/regulations_catalog.py.
_NUMBER_TO_CELEX: dict[str, str] = {
    _meta["number"]: _celex for _celex, _meta in REGULATIONS.items()
}


# ---------------------------------------------------------------------------
# URI normalization
# ---------------------------------------------------------------------------

def _normalize_neo4j_uri(uri: str) -> str:
    """
    The Neo4j *browser* runs on HTTP port 7474; the Python driver requires
    the Bolt protocol (port 7687).  This helper transparently converts the
    common mistake of supplying the browser URL::

        http://localhost:7474  →  bolt://localhost:7687
        https://localhost:7473 →  bolt+s://localhost:7688

    Any URI that already uses a bolt / neo4j scheme is returned unchanged.
    """
    if uri.startswith(("bolt", "neo4j")):
        return uri

    from urllib.parse import urlparse, urlunparse
    p = urlparse(uri)
    if p.scheme in ("http", "https"):
        new_scheme = "bolt+s" if p.scheme == "https" else "bolt"
        port_map = {7474: 7687, 7473: 7688}
        new_port = port_map.get(p.port, p.port)
        new_netloc = f"{p.hostname}:{new_port}" if new_port else p.hostname
        converted = urlunparse(p._replace(scheme=new_scheme, netloc=new_netloc))
        logger.warning(
            "NEO4J_URI '%s' uses an HTTP scheme (browser port).  "
            "Auto-converting to Bolt URI: '%s'",
            uri, converted,
        )
        return converted

    return uri


# ---------------------------------------------------------------------------
# kind → Neo4j label
# ---------------------------------------------------------------------------
_KIND_LABEL: dict[str, str] = {
    "document":         "Document",
    "preamble":         "Preamble",
    "enacting_terms":   "EnactingTerms",
    "final_provisions": "FinalProvisions",
    "annexes":          "Annexes",
    "citation":         "Citation",
    "recital":          "Recital",
    "chapter":          "Chapter",
    "section":          "Section",
    "article":          "Article",
    "paragraph":        "Paragraph",
    "subparagraph":     "Subparagraph",
    "point":            "Point",
    "roman_item":       "Point",       # sub-points (i), (ii) → folded into Point
    "annex":            "Annex",
    "annex_chapter":    "AnnexChapter",
    "annex_part":       "AnnexPart",
    "annex_section":    "AnnexSection",
    "annex_point":      "AnnexPoint",
    "annex_subpoint":   "AnnexSubpoint",
    "annex_bullet":     "AnnexBullet",
}

# No editorial containers are flattened – all structural nodes are kept
# as real graph nodes for navigable hierarchy.
_CONTAINER_KINDS: set[str] = set()

# Batch size for UNWIND queries (avoid hitting Neo4j bolt message limits)
_BATCH = 500


def _kind_label(kind: str) -> str:
    """Return the Neo4j label for a provision kind, falling back to a title-cased name."""
    return _KIND_LABEL.get(kind, kind.replace("_", " ").title().replace(" ", ""))


def _batched(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RegulationGraphLoader:
    """
    Load ``parsed.json`` regulation data into Neo4j as a structural hierarchy.

    Usage (context-manager)::

        with RegulationGraphLoader(uri, user, password) as loader:
            loader.setup_schema()
            stats = loader.load_file("data/regulations/32024R1689/EN/parsed.json")
            print(stats)

    Environment variables read by the convenience constructor
    :meth:`from_env` (also used by the CLI):

    * ``NEO4J_URI``      – default ``bolt://localhost:7687``
    * ``NEO4J_USER``     – default ``neo4j``
    * ``NEO4J_PASSWORD`` – default ``password``
    * ``NEO4J_DATABASE`` – default ``neo4j``
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "RegulationGraphLoader":
        import os
        from pathlib import Path as _Path
        from dotenv import load_dotenv

        _dotenv = dotenv_path or _Path.cwd() / ".env"
        load_dotenv(_dotenv, override=False)

        return cls(
            uri=_normalize_neo4j_uri(os.environ.get("NEO4J_URI", "bolt://localhost:7687")),
            user=os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j")),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def setup_schema(self) -> None:
        """
        Create uniqueness constraint and supporting indexes on a
        dedicated ``Provision`` label.  This label is used only for
        constraints / indexes and **not** attached to nodes at
        runtime, so it does not appear in query results.

        Safe to call repeatedly (uses ``IF NOT EXISTS``).
        """
        stmts = [
            # Uniqueness – prevents duplicate provisions across re-loads
            "CREATE CONSTRAINT provision_id IF NOT EXISTS "
            "FOR (p:Provision) REQUIRE p.id IS UNIQUE",
            # Fast look-up by regulation
            "CREATE INDEX provision_celex IF NOT EXISTS "
            "FOR (p:Provision) ON (p.celex)",
            # Fast look-up by structural type
            "CREATE INDEX provision_kind IF NOT EXISTS "
            "FOR (p:Provision) ON (p.kind)",
            # DefinedTerm – uniqueness and lookup indexes
            "CREATE CONSTRAINT defined_term_id IF NOT EXISTS "
            "FOR (d:DefinedTerm) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX defined_term_normalized IF NOT EXISTS "
            "FOR (d:DefinedTerm) ON (d.term_normalized)",
            "CREATE INDEX defined_term_category IF NOT EXISTS "
            "FOR (d:DefinedTerm) ON (d.category)",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in stmts:
                session.run(stmt)
        logger.info("Schema constraints / indexes ensured.")

    # ------------------------------------------------------------------
    # Public load API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path, wipe: bool = False) -> dict:
        """
        Parse a single ``parsed.json`` and upsert it into Neo4j.

        Parameters
        ----------
        path:
            Path to the ``parsed.json`` file.
        wipe:
            If *True*, delete all existing nodes for this CELEX before
            loading (useful for a clean re-import).

        Returns
        -------
        dict
            ``{"celex": str, "nodes": int, "relationships": int}``
        """
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        celex          = data["celex_id"]
        regulation_id  = data.get("regulation_id", celex)
        provisions     = data["provisions"]
        # relations includes cross-references AND DEFINED_BY edges (added by
        # the definitions semantic layer during parsing)
        xref_relations = data.get("relations", [])
        defined_terms  = data.get("defined_terms", [])

        logger.info("Loading %s – %d provisions …", celex, len(provisions))

        if wipe:
            self.wipe_regulation(celex)

        nodes, edges = self._prepare_data(provisions, celex, regulation_id)

        # DEFINED_BY relations reference DefinedTerm nodes (not Provision nodes)
        # so they must be handled after defined_terms are upserted; separate them
        # from the standard cross-reference set.
        defined_by_rels = [r for r in xref_relations if r.get("type") == "DEFINED_BY"]
        std_xref_rels   = [r for r in xref_relations if r.get("type") != "DEFINED_BY"]

        with self._driver.session(database=self._database) as session:
            n_nodes      = self._upsert_nodes(session, nodes)
            self._apply_kind_labels(session, celex)
            n_rels       = self._upsert_relationships(session, edges)
            n_xrefs      = self._upsert_cross_references(session, std_xref_rels, celex)
            n_dterms     = self._upsert_defined_terms(session, defined_terms)
            n_defined_by = self._upsert_defined_by_edges(session, defined_terms)

        stats = {
            "celex":         celex,
            "nodes":         n_nodes,
            "relationships": n_rels,
            "cross_references": n_xrefs,
            "defined_terms": n_dterms,
            "defined_by_edges": n_defined_by,
        }
        logger.info(
            "  → %d nodes, %d structural rels, %d cross-ref edges, "
            "%d defined-terms, %d DEFINED_BY edges.",
            n_nodes, n_rels, n_xrefs, n_dterms, n_defined_by,
        )
        return stats

    def wipe_regulation(self, celex: str) -> None:
        """Delete all nodes for *celex* (regardless of labels)."""
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (p {celex: $celex}) DETACH DELETE p",
                celex=celex,
            )
        logger.info("Wiped all nodes for %s.", celex)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(
        provisions: list[dict],
        celex: str,
        regulation_id: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        Transform provisions list into flat node-dicts and edge-dicts
        ready for UNWIND Cypher queries.

        Editorial containers (preamble, enacting_terms, final_provisions,
        annexes) are removed: their children are spliced into the parent's
        children list, paths are cleaned, and hierarchy_depth decremented.
        """
        # ----------------------------------------------------------
        # Phase 0 – flatten editorial containers out of the tree
        # ----------------------------------------------------------
        by_id: dict[str, dict] = {p["id"]: p for p in provisions}
        container_ids: set[str] = {
            p["id"] for p in provisions
            if p.get("kind") in _CONTAINER_KINDS
        }

        if container_ids:
            # Splice each container's children into its parent's
            # children list (preserving order at the container's position).
            for prov in provisions:
                old_children = prov.get("children", [])
                if not old_children:
                    continue
                new_children: list[str] = []
                for cid in old_children:
                    if cid in container_ids:
                        # Replace the container with its own children
                        container = by_id[cid]
                        new_children.extend(container.get("children", []))
                    else:
                        new_children.append(cid)
                prov["children"] = new_children

            # Strip container IDs from every descendant's path and
            # re-compute hierarchy_depth.
            for prov in provisions:
                if prov["id"] in container_ids:
                    continue
                raw_path = prov.get("path", []) or []
                cleaned = [pid for pid in raw_path if pid not in container_ids]
                prov["path"] = cleaned
                prov["hierarchy_depth"] = len(cleaned)

            # Remove container provisions themselves
            provisions = [
                p for p in provisions if p["id"] not in container_ids
            ]
            # Rebuild lookup without containers
            by_id = {p["id"]: p for p in provisions}

            logger.info(
                "Flattened %d editorial containers; %d provisions remain.",
                len(container_ids), len(provisions),
            )

        # ----------------------------------------------------------
        # Phase 1 – build node / edge dicts for Neo4j
        # ----------------------------------------------------------
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        def canonical_ref(p: dict) -> str:
            kind = p.get("kind", "") or ""
            number = p.get("number")

            if kind == "preamble":
                return "Preamble"
            if kind == "enacting_terms":
                return "Enacting Terms"
            if kind == "final_provisions":
                return "Final Provisions"
            if kind == "annexes":
                return "Annexes"
            if kind == "chapter" and number:
                return f"Chapter {number}"
            if kind == "section" and number:
                return f"Section {number}"
            if kind == "article" and number:
                return f"Article {number}"
            if kind == "paragraph" and number:
                return f"Paragraph {number}"
            if kind == "subparagraph" and number:
                return f"Subparagraph {number}"
            if kind in ("point", "roman_item") and number:
                return f"Point ({number})"
            if kind == "annex" and number:
                return f"Annex {number}"
            if kind == "annex_chapter" and number:
                return f"Annex chapter {number}"
            if kind == "annex_part" and number:
                return f"Annex part {number}"
            if kind == "annex_section" and number:
                return f"Annex section {number}"
            if kind == "annex_point" and number:
                return f"Annex point {number}"
            if kind == "annex_subpoint" and number:
                return f"Annex subpoint ({number})"
            if kind == "annex_bullet":
                return "Annex bullet"

            title = p.get("title")
            if title:
                return title
            text = (p.get("text") or "").strip()
            if text:
                return text[:80]
            return kind or ""

        for prov in provisions:
            raw_path = prov.get("path", []) or []

            # Build a human-readable path from structural ancestors plus self
            segments: list[str] = []
            for anc_id in raw_path:
                anc = by_id.get(anc_id)
                if not anc:
                    continue
                ref = canonical_ref(anc)
                if ref:
                    segments.append(ref)

            self_ref = canonical_ref(prov)
            if self_ref:
                segments.append(self_ref)

            nodes.append({
                "id":              prov["id"],
                "celex":           celex,
                "regulation_id":   regulation_id,
                "lang":            prov.get("lang", "EN"),
                "kind":            prov.get("kind", ""),
                "level":           prov.get("kind", ""),
                "text":            prov.get("text", ""),
                "text_for_analysis": prov.get("text_for_analysis"),
                "hierarchy_depth": prov.get("hierarchy_depth", 0),
                "number":          prov.get("number"),
                "title":           prov.get("title"),
                "path_string":     "/".join(raw_path),
                "display_ref":     self_ref,
                "display_path":    " / ".join(segments),
            })

            for order, child_id in enumerate(prov.get("children", [])):
                edges.append({
                    "parent_id": prov["id"],
                    "child_id":  child_id,
                    "order":     order,
                })

        return nodes, edges

    def _upsert_nodes(self, session, nodes: list[dict]) -> int:
        """
        MERGE all provision nodes in batches.
        Returns total count of nodes touched.
        """
        total = 0
        cypher = """
            UNWIND $batch AS n
            MERGE (p:Provision {id: n.id})
            SET
                p.celex           = n.celex,
                p.regulation_id   = n.regulation_id,
                p.lang            = n.lang,
                p.kind            = n.kind,
                p.level           = n.level,
                p.text            = n.text,
                p.text_for_analysis = n.text_for_analysis,
                p.hierarchy_depth = n.hierarchy_depth,
                p.number          = n.number,
                p.title           = n.title,
                p.path_string     = n.path_string,
                p.display_ref     = n.display_ref,
                p.display_path    = n.display_path
            RETURN count(p) AS c
        """
        for chunk in _batched(nodes, _BATCH):
            result = session.run(cypher, batch=chunk)
            total += result.single()["c"]
        return total

    def _apply_kind_labels(self, session, celex: str) -> None:
        """
        Add a specific structural label (e.g. ``:Article``) to each node
        based on its ``kind`` property.  Runs one Cypher statement per kind.
        """
        for kind, label in _KIND_LABEL.items():
            # Only touch nodes of this regulation that have this kind
            session.run(
                f"MATCH (p {{celex: $celex, kind: $kind}}) SET p:{label}",
                celex=celex,
                kind=kind,
            )

    def _upsert_relationships(self, session, edges: list[dict]) -> int:
        """
        MERGE HAS_PART edges in batches.
        Returns count of HAS_PART relationships created/merged.
        To traverse upward use the reverse direction: MATCH (x)<-[:HAS_PART]-(parent).
        """
        total = 0
        cypher = """
            UNWIND $batch AS e
            MATCH (parent:Provision {id: e.parent_id})
            MATCH (child:Provision  {id: e.child_id})
            MERGE (parent)-[r:HAS_PART]->(child)
              ON CREATE SET r.order = e.order
            RETURN count(r) AS c
        """
        for chunk in _batched(edges, _BATCH):
            result = session.run(cypher, batch=chunk)
            total += result.single()["c"]
        return total

    # ------------------------------------------------------------------
    # Cross-reference edges
    # ------------------------------------------------------------------

    # Allowed relationship types — prevents Cypher injection via dynamic labels.
    _XREF_TYPES: set[str] = {"CITES", "CITES_EXTERNAL", "AMENDS"}

    def _upsert_cross_references(
        self, session, relations: list[dict], celex: str,
    ) -> int:
        """
        Create cross-reference edges from the ``relations`` array in
        ``parsed.json``.

        Internal references (CITES, CITES_RANGE) link existing provision
        nodes.  External references (CITES_EXTERNAL, AMENDS) MERGE
        lightweight ``ExternalAct`` stub nodes so the edge always has a
        target.
        """
        if not relations:
            return 0

        internal: list[dict] = []
        external: list[dict] = []

        for rel in relations:
            rel_type = rel.get("type", "")
            if rel_type not in self._XREF_TYPES:
                logger.warning("Skipping unknown relation type: %s", rel_type)
                continue
            props = rel.get("properties") or {}
            entry = {
                "source": rel["source"],
                "target": rel["target"],
                "rel_type": rel_type,
                "ref_text": props.get("ref_text", ""),
                "number":  props.get("number", ""),
            }
            if rel_type in ("CITES_EXTERNAL", "AMENDS"):
                external.append(entry)
            else:
                internal.append(entry)

        total = 0
        total += self._upsert_internal_xrefs(session, internal)
        total += self._upsert_external_xrefs(session, external, celex)

        logger.info(
            "  Cross-refs: %d internal, %d external → %d edges total.",
            len(internal), len(external), total,
        )
        return total

    def _upsert_internal_xrefs(self, session, refs: list[dict]) -> int:
        """MERGE CITES / CITES_RANGE edges between existing provision nodes."""
        total = 0
        # Group by rel_type so we can use a fixed relationship label per query
        by_type: dict[str, list[dict]] = {}
        for r in refs:
            by_type.setdefault(r["rel_type"], []).append(r)

        for rel_type, entries in by_type.items():
            if rel_type not in self._XREF_TYPES:
                continue
            cypher = (
                "UNWIND $batch AS e "
                "MATCH (s:Provision {id: e.source}) "
                "MATCH (t:Provision {id: e.target}) "
                f"MERGE (s)-[r:{rel_type}]->(t) "
                "ON CREATE SET r.ref_text = e.ref_text "
                "RETURN count(r) AS c"
            )
            for chunk in _batched(entries, _BATCH):
                result = session.run(cypher, batch=chunk)
                total += result.single()["c"]
        return total

    # ------------------------------------------------------------------
    # DefinedTerm nodes and DEFINED_BY edges
    # ------------------------------------------------------------------

    def _upsert_defined_terms(self, session, defined_terms: list[dict]) -> int:
        """MERGE :DefinedTerm nodes in batches.

        Returns total count of DefinedTerm nodes touched.
        """
        if not defined_terms:
            return 0
        total = 0
        cypher = """
            UNWIND $batch AS dt
            MERGE (d:DefinedTerm {id: dt.id})
            SET
                d.term                = dt.term,
                d.term_normalized     = dt.term_normalized,
                d.category            = dt.category,
                d.celex               = dt.celex,
                d.regulation          = dt.regulation,
                d.source_provision_id = dt.source_provision_id
            RETURN count(d) AS c
        """
        for chunk in _batched(defined_terms, _BATCH):
            result = session.run(cypher, batch=chunk)
            total += result.single()["c"]
        logger.info("  DefinedTerm nodes: %d upserted.", total)
        return total

    def _upsert_defined_by_edges(self, session, defined_terms: list[dict]) -> int:
        """MERGE DEFINED_BY edges from :DefinedTerm → :Provision in batches.

        Returns total count of edges created/merged.
        """
        if not defined_terms:
            return 0
        total = 0
        cypher = """
            UNWIND $batch AS dt
            MATCH (d:DefinedTerm {id: dt.id})
            MATCH (p:Provision   {id: dt.source_provision_id})
            MERGE (d)-[r:DEFINED_BY]->(p)
            RETURN count(r) AS c
        """
        for chunk in _batched(defined_terms, _BATCH):
            result = session.run(cypher, batch=chunk)
            total += result.single()["c"]
        logger.info("  DEFINED_BY edges: %d upserted.", total)
        return total

    def _upsert_external_xrefs(
        self, session, refs: list[dict], celex: str,
    ) -> int:
        """Create cross-reference edges for external references.

        References to regulations already loaded in the graph (identified
        by ``_NUMBER_TO_CELEX``) are **skipped** here — they are resolved
        into concrete CITES edges by the crosslinker post-processing step.
        Only truly external references (regulations we do not load) get
        the lightweight ``ExternalAct`` stub node.
        """
        truly_external: list[dict] = []
        skipped = 0
        for r in refs:
            number = r.get("number", "")
            target_celex = _NUMBER_TO_CELEX.get(number)
            if target_celex is not None:
                # Target regulation is loaded — crosslinker will resolve
                skipped += 1
                continue
            truly_external.append(r)

        if skipped:
            logger.info(
                "  Skipped %d external refs (target regulation loaded; "
                "crosslinker will resolve).", skipped,
            )

        total = 0
        by_type: dict[str, list[dict]] = {}
        for r in truly_external:
            by_type.setdefault(r["rel_type"], []).append(r)

        for rel_type, entries in by_type.items():
            if rel_type not in self._XREF_TYPES:
                continue
            cypher = (
                "UNWIND $batch AS e "
                "MATCH (s:Provision {id: e.source}) "
                "MERGE (t:ExternalAct {id: e.target}) "
                "ON CREATE SET t.ref_text = e.ref_text "
                f"MERGE (s)-[r:{rel_type}]->(t) "
                "ON CREATE SET r.ref_text = e.ref_text "
                "RETURN count(r) AS c"
            )
            for chunk in _batched(entries, _BATCH):
                result = session.run(cypher, batch=chunk)
                total += result.single()["c"]
        return total
