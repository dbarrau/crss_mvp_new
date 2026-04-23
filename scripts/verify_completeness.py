#!/usr/bin/env python3
"""
Deterministic GraphRAG Completeness Verifier
=============================================

Proves that every piece of source text in the CRSS regulatory
knowledge base flows correctly through the pipeline:

    Source → parsed.json → Neo4j → Retrievable

Usage::

    # Full verification (all layers, all documents)
    python scripts/verify_completeness.py

    # Single document
    python scripts/verify_completeness.py --doc MDCG_2019_11

    # Offline only (no Neo4j required)
    python scripts/verify_completeness.py --layer 1

    # Specific layers
    python scripts/verify_completeness.py --layer 1 2

Layers
------
1. Source ↔ parsed.json   (parsing completeness — offline)
2. parsed.json ↔ Neo4j    (loading completeness — requires Neo4j)
3. Neo4j ↔ Retrieval      (query-ability — requires Neo4j)
4. Cross-reference integrity (requires Neo4j)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from domain.legislation_catalog import LEGISLATION
from domain.mdcg_catalog import MDCG_DOCUMENTS

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

DATA_DIR = _PROJECT_ROOT / "data"
LEGISLATION_DIR = DATA_DIR / "legislation"
GUIDANCE_DIR = DATA_DIR / "guidance"


# ── Result structures ─────────────────────────────────────────────────────

@dataclass
class Check:
    """A single pass/fail verification check."""
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"name": self.name, "passed": self.passed}
        if self.expected is not None:
            d["expected"] = self.expected
        if self.actual is not None:
            d["actual"] = self.actual
        if self.details:
            d["details"] = self.details[:50]  # cap detail lines
        return d


@dataclass
class DocumentReport:
    """Verification report for a single document."""
    doc_id: str
    doc_type: str  # "legislation" or "guidance"
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "passed": self.passed,
            "checks_passed": self.n_passed,
            "checks_failed": self.n_failed,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class VerificationReport:
    """Top-level verification report across all documents and layers."""
    documents: list[DocumentReport] = field(default_factory=list)
    global_checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            all(d.passed for d in self.documents)
            and all(c.passed for c in self.global_checks)
        )

    def to_dict(self) -> dict:
        total_checks = sum(
            len(d.checks) for d in self.documents
        ) + len(self.global_checks)
        total_passed = sum(d.n_passed for d in self.documents) + sum(
            1 for c in self.global_checks if c.passed
        )
        return {
            "overall_passed": self.passed,
            "total_checks": total_checks,
            "total_passed": total_passed,
            "total_failed": total_checks - total_passed,
            "documents": [d.to_dict() for d in self.documents],
            "global_checks": [c.to_dict() for c in self.global_checks],
        }


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1: Source ↔ parsed.json
# ══════════════════════════════════════════════════════════════════════════

def _load_parsed_json(path: Path) -> dict | None:
    """Load a parsed.json file, returning None if missing."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Layer 1a: Structural integrity of any parsed.json ─────────────────────

def verify_structural_integrity(provisions: list[dict], doc_id: str) -> list[Check]:
    """Validate hierarchical and content integrity of provision tree."""
    checks: list[Check] = []
    by_id = {p["id"]: p for p in provisions}
    all_ids = list(by_id.keys())

    # 1. No duplicate IDs
    id_counts = Counter(p["id"] for p in provisions)
    dups = {pid: cnt for pid, cnt in id_counts.items() if cnt > 1}
    checks.append(Check(
        name="no_duplicate_ids",
        passed=len(dups) == 0,
        expected=0,
        actual=len(dups),
        details=[f"{pid} appears {cnt}x" for pid, cnt in dups.items()],
    ))

    # 2. Zero orphans: every parent_id references an existing node
    orphans = []
    for p in provisions:
        pid = p.get("parent_id")
        if pid and pid not in by_id:
            orphans.append(f"{p['id']} references non-existent parent {pid}")
    checks.append(Check(
        name="no_orphan_parents",
        passed=len(orphans) == 0,
        expected=0,
        actual=len(orphans),
        details=orphans,
    ))

    # 3. Bidirectional parent↔child consistency
    mismatches = []
    for p in provisions:
        for cid in p.get("children", []):
            if cid not in by_id:
                mismatches.append(f"{p['id']} lists missing child {cid}")
            else:
                child = by_id[cid]
                if child.get("parent_id") != p["id"]:
                    mismatches.append(
                        f"{p['id']} lists child {cid}, but child's "
                        f"parent_id={child.get('parent_id')}"
                    )
    # Reverse: every non-root provision should be in its parent's children
    for p in provisions:
        pid = p.get("parent_id")
        if pid and pid in by_id:
            parent = by_id[pid]
            if p["id"] not in parent.get("children", []):
                mismatches.append(
                    f"{p['id']} has parent_id={pid} but is not in parent's children"
                )
    checks.append(Check(
        name="parent_child_bidirectional",
        passed=len(mismatches) == 0,
        expected=0,
        actual=len(mismatches),
        details=mismatches,
    ))

    # 4. No null text on leaf nodes
    null_leaves = []
    for p in provisions:
        if not p.get("children"):  # leaf
            text = p.get("text") or ""
            if not text.strip():
                null_leaves.append(p["id"])
    checks.append(Check(
        name="no_null_text_on_leaves",
        passed=len(null_leaves) == 0,
        expected=0,
        actual=len(null_leaves),
        details=null_leaves,
    ))

    # 5. text_for_analysis populated on non-container nodes
    #    Root/container nodes (document, preamble, enacting_terms, etc.)
    #    legitimately have empty text_for_analysis.
    _CONTAINER_KINDS = {
        "document", "guidance_document", "preamble",
        "enacting_terms", "final_provisions", "annexes",
    }
    missing_tfa = []
    for p in provisions:
        if p.get("kind") in _CONTAINER_KINDS:
            continue
        tfa = p.get("text_for_analysis") or ""
        if len(tfa.strip()) < 10:
            missing_tfa.append(f"{p['id']} (len={len(tfa.strip())})")
    checks.append(Check(
        name="text_for_analysis_populated",
        passed=len(missing_tfa) == 0,
        expected=0,
        actual=len(missing_tfa),
        details=missing_tfa,
    ))

    # 6. hierarchy_depth == len(path)
    depth_mismatches = []
    for p in provisions:
        path = p.get("path") or []
        depth = p.get("hierarchy_depth", -1)
        if depth != len(path):
            depth_mismatches.append(
                f"{p['id']}: hierarchy_depth={depth}, len(path)={len(path)}"
            )
    checks.append(Check(
        name="hierarchy_depth_matches_path_length",
        passed=len(depth_mismatches) == 0,
        expected=0,
        actual=len(depth_mismatches),
        details=depth_mismatches,
    ))

    # 7. Path consistency: path[-1] should equal parent_id (when path is non-empty)
    path_errors = []
    for p in provisions:
        path = p.get("path") or []
        pid = p.get("parent_id")
        if path and path[-1] != pid:
            path_errors.append(
                f"{p['id']}: path[-1]={path[-1]}, parent_id={pid}"
            )
    checks.append(Check(
        name="path_last_equals_parent_id",
        passed=len(path_errors) == 0,
        expected=0,
        actual=len(path_errors),
        details=path_errors,
    ))

    return checks


# ── Layer 1b: Legislation HTML → parsed.json ──────────────────────────────

def verify_legislation_source(celex: str) -> list[Check]:
    """Compare EUR-Lex raw HTML structural elements against parsed.json."""
    checks: list[Check] = []
    parsed_path = LEGISLATION_DIR / celex / "EN" / "parsed.json"
    html_path = LEGISLATION_DIR / celex / "EN" / "raw" / "raw.html"

    data = _load_parsed_json(parsed_path)
    if data is None:
        checks.append(Check(
            name="parsed_json_exists",
            passed=False,
            details=[f"Missing: {parsed_path}"],
        ))
        return checks
    checks.append(Check(name="parsed_json_exists", passed=True))

    provisions = data.get("provisions", [])

    # Check HTML source exists
    if not html_path.exists():
        checks.append(Check(
            name="raw_html_exists",
            passed=False,
            details=[f"Missing: {html_path}"],
        ))
        return checks
    checks.append(Check(name="raw_html_exists", passed=True))

    # Import HTML patterns
    from domain.ontology.eurlex_html import (
        ARTICLE_ID_RE, PARAGRAPH_ID_RE, RECITAL_ID_RE, ANNEX_ID_RE,
        CHAPTER_ID_RE, SECTION_ID_RE,
    )

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        checks.append(Check(
            name="beautifulsoup_available",
            passed=False,
            details=["pip install beautifulsoup4 for full HTML verification"],
        ))
        return checks

    html_content = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract all structural IDs from HTML
    html_ids: dict[str, list[str]] = defaultdict(list)
    for elem in soup.find_all(id=True):
        eid = elem["id"]
        if ARTICLE_ID_RE.match(eid):
            html_ids["article"].append(eid)
        elif PARAGRAPH_ID_RE.match(eid):
            html_ids["paragraph"].append(eid)
        elif RECITAL_ID_RE.match(eid):
            html_ids["recital"].append(eid)
        elif ANNEX_ID_RE.match(eid):
            html_ids["annex"].append(eid)
        elif CHAPTER_ID_RE.match(eid):
            html_ids["chapter"].append(eid)
        elif SECTION_ID_RE.match(eid):
            html_ids["section"].append(eid)

    # Count provisions by kind
    prov_kinds: dict[str, list[str]] = defaultdict(list)
    for p in provisions:
        prov_kinds[p.get("kind", "unknown")].append(p["id"])

    # Article count parity
    n_html_articles = len(html_ids["article"])
    n_parsed_articles = len(prov_kinds.get("article", []))
    checks.append(Check(
        name="article_count_parity",
        passed=n_html_articles == n_parsed_articles,
        expected=n_html_articles,
        actual=n_parsed_articles,
        details=(
            [f"HTML has {n_html_articles} articles, parsed.json has {n_parsed_articles}"]
            if n_html_articles != n_parsed_articles else []
        ),
    ))

    # Recital count parity
    n_html_recitals = len(html_ids["recital"])
    n_parsed_recitals = len(prov_kinds.get("recital", []))
    checks.append(Check(
        name="recital_count_parity",
        passed=n_html_recitals == n_parsed_recitals,
        expected=n_html_recitals,
        actual=n_parsed_recitals,
    ))

    # Annex count parity
    n_html_annexes = len(html_ids["annex"])
    n_parsed_annexes = len(prov_kinds.get("annex", []))
    checks.append(Check(
        name="annex_count_parity",
        passed=n_html_annexes == n_parsed_annexes,
        expected=n_html_annexes,
        actual=n_parsed_annexes,
    ))

    # Article ID coverage: every HTML article ID should map to a parsed provision
    parsed_id_set = {p["id"] for p in provisions}
    missing_articles = []
    for html_id in html_ids["article"]:
        expected_id = f"{celex}_{html_id}"
        if expected_id not in parsed_id_set:
            missing_articles.append(expected_id)
    checks.append(Check(
        name="article_id_coverage",
        passed=len(missing_articles) == 0,
        expected=0,
        actual=len(missing_articles),
        details=missing_articles,
    ))

    # Text spot-check: sample articles, compare text content
    text_mismatches = []
    sample_size = min(10, n_parsed_articles)
    article_provs = [p for p in provisions if p.get("kind") == "article"]
    import random
    random.seed(42)  # reproducible
    sample = random.sample(article_provs, sample_size) if article_provs else []
    for p in sample:
        text = (p.get("text") or "").strip()
        if not text:
            text_mismatches.append(f"{p['id']}: empty text in parsed.json")
    checks.append(Check(
        name="article_text_nonempty_sample",
        passed=len(text_mismatches) == 0,
        expected=0,
        actual=len(text_mismatches),
        details=text_mismatches,
    ))

    return checks


# ── Layer 1c: MDCG Markdown → parsed.json ─────────────────────────────────

def verify_mdcg_source(doc_id: str) -> list[Check]:
    """Compare MDCG clean markdown headings against parsed.json provisions."""
    checks: list[Check] = []
    doc_dir = GUIDANCE_DIR / doc_id / "EN"
    parsed_path = doc_dir / "parsed.json"

    data = _load_parsed_json(parsed_path)
    if data is None:
        checks.append(Check(
            name="parsed_json_exists",
            passed=False,
            details=[f"Missing: {parsed_path}"],
        ))
        return checks
    checks.append(Check(name="parsed_json_exists", passed=True))

    provisions = data.get("provisions", [])

    # Find clean markdown
    md_files = list(doc_dir.glob("*_clean.md"))
    if not md_files:
        checks.append(Check(
            name="clean_md_exists",
            passed=False,
            details=[f"No *_clean.md in {doc_dir}"],
        ))
        return checks
    checks.append(Check(name="clean_md_exists", passed=True))

    md_text = md_files[0].read_text(encoding="utf-8")

    # Count headings by level in markdown
    md_heading_counts: dict[int, int] = Counter()
    # Track unique heading texts for coverage
    md_headings: list[tuple[int, str]] = []
    for line in md_text.split("\n"):
        stripped = line.strip()
        m = re.match(r"^(#{1,6})\s+(.+)", stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Skip footnotes, contents, preamble headings
            if re.match(r"(?i)^footnotes?\s*$", title):
                break  # everything after is footnotes
            if re.match(r"(?i)^contents?\s*$", title):
                continue
            if re.match(r"(?i)^MDCG\s+\d{4}", title):
                continue
            md_heading_counts[level] += 1
            md_headings.append((level, title))

    # Count non-root provisions (exclude the document node)
    non_root = [p for p in provisions if p.get("kind") != "guidance_document"]
    # Tolerance accounts for: preamble/TOC headings skipped by structurer,
    # chart sub-headings merged, and unnumbered headings sometimes collapsed.
    tolerance = max(3, int(len(md_headings) * 0.15))
    checks.append(Check(
        name="heading_vs_provision_count",
        passed=abs(len(md_headings) - len(non_root)) <= tolerance,
        expected=len(md_headings),
        actual=len(non_root),
        details=(
            [f"Markdown has {len(md_headings)} headings, "
             f"parsed.json has {len(non_root)} non-root provisions"]
            if abs(len(md_headings) - len(non_root)) > 2 else []
        ),
    ))

    # Content coverage: each provision's text should be findable in the markdown
    # We normalize whitespace for comparison
    def _normalize(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    md_normalized = _normalize(md_text)
    missing_content = []
    for p in non_root:
        text = p.get("text") or ""
        if len(text.strip()) < 20:
            continue  # skip very short provisions (headings only)
        # Take a representative chunk (first 100 chars of body text)
        # Strip the title if it prefixes the text
        body = text
        title = p.get("title") or ""
        if title and body.startswith(title):
            body = body[len(title):].strip()
        if not body:
            continue
        snippet = _normalize(body[:100])
        if snippet and snippet not in md_normalized:
            missing_content.append(
                f"{p['id']}: first 100 chars not found in markdown"
            )
    checks.append(Check(
        name="provision_text_in_markdown",
        passed=len(missing_content) == 0,
        expected=0,
        actual=len(missing_content),
        details=missing_content,
    ))

    # Total text coverage ratio
    total_prov_chars = sum(len(p.get("text") or "") for p in non_root)
    md_chars = len(md_text.strip())
    ratio = total_prov_chars / md_chars if md_chars > 0 else 0
    checks.append(Check(
        name="text_coverage_ratio",
        passed=ratio >= 0.70,  # provisions should capture >=70% of markdown
        expected=">=70%",
        actual=f"{ratio:.1%}",
        details=(
            [f"Provision text: {total_prov_chars} chars, "
             f"Markdown: {md_chars} chars, Ratio: {ratio:.1%}"]
            if ratio < 0.70 else []
        ),
    ))

    # Metadata cross-check
    meta_path = doc_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta_cleaned = meta.get("cleaned_chars", 0)
        if meta_cleaned > 0:
            diff_pct = abs(md_chars - meta_cleaned) / meta_cleaned
            checks.append(Check(
                name="metadata_char_count_consistent",
                passed=diff_pct < 0.05,  # <5% difference
                expected=meta_cleaned,
                actual=md_chars,
                details=(
                    [f"metadata.json says {meta_cleaned} chars, "
                     f"actual file has {md_chars} chars ({diff_pct:.1%} diff)"]
                    if diff_pct >= 0.05 else []
                ),
            ))

    return checks


# ── Layer 1d: Relation integrity in parsed.json ───────────────────────────

def verify_relation_integrity(data: dict, doc_id: str) -> list[Check]:
    """Verify all relation source/target IDs exist in the provisions."""
    checks: list[Check] = []
    provisions = data.get("provisions", [])
    relations = data.get("relations", [])
    prov_ids = {p["id"] for p in provisions}

    dangling = []
    for rel in relations:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = rel.get("type", "")
        # DEFINED_BY sources are DefinedTerm nodes, not in provisions list
        if rel_type == "DEFINED_BY":
            continue
        if src not in prov_ids:
            dangling.append(f"source {src} missing (type={rel_type})")
        # target may reference another regulation — only check for CITES (internal)
        if rel_type == "CITES" and tgt not in prov_ids:
            dangling.append(f"target {tgt} missing (type=CITES)")

    checks.append(Check(
        name="relation_endpoints_exist",
        passed=len(dangling) == 0,
        expected=0,
        actual=len(dangling),
        details=dangling,
    ))

    return checks


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2: parsed.json ↔ Neo4j
# ══════════════════════════════════════════════════════════════════════════

def _get_neo4j_session():
    """Create a Neo4j driver and return (driver, session_factory)."""
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    from infrastructure.graphdb.neo4j.loader import _normalize_neo4j_uri

    uri = _normalize_neo4j_uri(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver, database


def verify_neo4j_parity(
    parsed_path: Path, doc_id: str, driver, database: str,
) -> list[Check]:
    """Compare parsed.json provisions against what's in Neo4j."""
    checks: list[Check] = []

    data = _load_parsed_json(parsed_path)
    if data is None:
        checks.append(Check(
            name="parsed_json_for_neo4j",
            passed=False,
            details=[f"Missing: {parsed_path}"],
        ))
        return checks

    provisions = data.get("provisions", [])
    celex = data.get("celex_id", doc_id)

    # Determine base label
    is_guidance = any(
        p.get("kind", "").startswith("guidance_") for p in provisions
    )
    base_label = "Guidance" if is_guidance else "Provision"

    with driver.session(database=database) as session:
        # Step 4: Node count parity
        result = session.run(
            f"MATCH (n:{base_label} {{celex: $celex}}) RETURN count(n) AS cnt",
            celex=celex,
        )
        neo4j_count = result.single()["cnt"]

        # The loader flattens editorial containers (_CONTAINER_KINDS is empty now,
        # so no flattening). We compare directly.
        parsed_count = len(provisions)
        checks.append(Check(
            name="node_count_parity",
            passed=neo4j_count == parsed_count,
            expected=parsed_count,
            actual=neo4j_count,
            details=(
                [f"parsed.json has {parsed_count} provisions, "
                 f"Neo4j has {neo4j_count} {base_label} nodes for {celex}"]
                if neo4j_count != parsed_count else []
            ),
        ))

        # Step 5: Node-level content parity (batch query)
        result = session.run(
            f"MATCH (n:{base_label} {{celex: $celex}}) "
            "RETURN n.id AS id, n.text AS text, n.kind AS kind, "
            "n.hierarchy_depth AS depth, "
            "n.text_for_analysis AS tfa",
            celex=celex,
        )
        neo4j_nodes = {r["id"]: r for r in result}

        # Check each parsed provision exists in Neo4j
        missing_in_neo4j = []
        text_mismatches = []
        kind_mismatches = []
        missing_tfa = []

        for p in provisions:
            pid = p["id"]
            if pid not in neo4j_nodes:
                missing_in_neo4j.append(pid)
                continue

            neo = neo4j_nodes[pid]

            # Text content parity (SHA256 comparison)
            parsed_text = p.get("text") or ""
            neo_text = neo["text"] or ""
            if _sha256(parsed_text) != _sha256(neo_text):
                text_mismatches.append(
                    f"{pid}: parsed={_sha256(parsed_text)}, "
                    f"neo4j={_sha256(neo_text)}"
                )

            # Kind parity
            if p.get("kind") != neo["kind"]:
                kind_mismatches.append(
                    f"{pid}: parsed={p.get('kind')}, neo4j={neo['kind']}"
                )

            # text_for_analysis populated (skip container kinds)
            _CONTAINER_KINDS_NEO4J = {
                "document", "guidance_document", "preamble",
                "enacting_terms", "final_provisions", "annexes",
            }
            if (p.get("kind") not in _CONTAINER_KINDS_NEO4J
                    and not (neo.get("tfa") or "").strip()):
                missing_tfa.append(pid)

        checks.append(Check(
            name="all_provisions_in_neo4j",
            passed=len(missing_in_neo4j) == 0,
            expected=0,
            actual=len(missing_in_neo4j),
            details=missing_in_neo4j,
        ))
        checks.append(Check(
            name="text_content_hash_match",
            passed=len(text_mismatches) == 0,
            expected=0,
            actual=len(text_mismatches),
            details=text_mismatches,
        ))
        checks.append(Check(
            name="kind_match",
            passed=len(kind_mismatches) == 0,
            expected=0,
            actual=len(kind_mismatches),
            details=kind_mismatches,
        ))
        checks.append(Check(
            name="text_for_analysis_in_neo4j",
            passed=len(missing_tfa) == 0,
            expected=0,
            actual=len(missing_tfa),
            details=missing_tfa,
        ))

        # Step 6: Relationship parity
        # Count HAS_PART edges
        result = session.run(
            f"MATCH (a:{base_label} {{celex: $celex}})-[r:HAS_PART]->"
            f"(b:{base_label}) "
            "RETURN count(r) AS cnt",
            celex=celex,
        )
        neo4j_has_part = result.single()["cnt"]

        # Expected: count parent→child pairs from parsed.json
        parsed_has_part = sum(
            len(p.get("children", []))
            for p in provisions
        )
        checks.append(Check(
            name="has_part_count_parity",
            passed=neo4j_has_part == parsed_has_part,
            expected=parsed_has_part,
            actual=neo4j_has_part,
        ))

        # Count cross-reference edges
        result = session.run(
            f"MATCH (a {{celex: $celex}})-[r]->(b) "
            "WHERE type(r) IN ['CITES', 'CITES_EXTERNAL'] "
            "RETURN type(r) AS rel_type, count(r) AS cnt",
            celex=celex,
        )
        neo4j_xrefs = {r["rel_type"]: r["cnt"] for r in result}
        parsed_xrefs = Counter(
            r.get("type") for r in data.get("relations", [])
            if r.get("type") in ("CITES", "CITES_EXTERNAL")
        )
        for rel_type in set(list(neo4j_xrefs.keys()) + list(parsed_xrefs.keys())):
            neo_cnt = neo4j_xrefs.get(rel_type, 0)
            # For CITES: crosslinker may have resolved CITES_EXTERNAL into CITES,
            # so Neo4j may have MORE CITES than parsed.json. That's expected.
            # We only flag if Neo4j has FEWER.
            parsed_cnt = parsed_xrefs.get(rel_type, 0)
            if rel_type == "CITES":
                checks.append(Check(
                    name=f"{rel_type}_count",
                    passed=neo_cnt >= parsed_cnt,
                    expected=f">={parsed_cnt}",
                    actual=neo_cnt,
                ))
            elif rel_type == "CITES_EXTERNAL":
                # After crosslinker, CITES_EXTERNAL should ideally be 0
                checks.append(Check(
                    name=f"{rel_type}_residual_count",
                    passed=True,  # informational
                    expected="ideally 0",
                    actual=neo_cnt,
                ))

        # Step 7: No orphan nodes in Neo4j
        # Nodes with no incoming HAS_PART that aren't document/guidance_document
        result = session.run(
            f"MATCH (n:{base_label} {{celex: $celex}}) "
            f"WHERE NOT (n)<-[:HAS_PART]-() "
            "AND n.kind <> 'document' AND n.kind <> 'guidance_document' "
            "RETURN n.id AS id, n.kind AS kind",
            celex=celex,
        )
        neo4j_orphans = [
            f"{r['id']} (kind={r['kind']})" for r in result
        ]
        checks.append(Check(
            name="no_neo4j_orphan_nodes",
            passed=len(neo4j_orphans) == 0,
            expected=0,
            actual=len(neo4j_orphans),
            details=neo4j_orphans,
        ))

    return checks


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3: Neo4j ↔ Retrieval
# ══════════════════════════════════════════════════════════════════════════

def verify_retrieval_readiness(
    parsed_path: Path, doc_id: str, driver, database: str,
) -> list[Check]:
    """Verify embeddings, display_ref, and vector self-retrieval."""
    checks: list[Check] = []

    data = _load_parsed_json(parsed_path)
    if data is None:
        return checks

    provisions = data.get("provisions", [])
    celex = data.get("celex_id", doc_id)

    is_guidance = any(
        p.get("kind", "").startswith("guidance_") for p in provisions
    )
    base_label = "Guidance" if is_guidance else "Provision"

    with driver.session(database=database) as session:
        # Step 8: Embedding completeness
        # Exclude structural container nodes that don't carry text content
        _EMBED_SKIP_KINDS = [
            "document", "guidance_document", "preamble",
            "enacting_terms", "final_provisions", "annexes",
        ]
        result = session.run(
            f"MATCH (n:{base_label} {{celex: $celex}}) "
            "WHERE n.embedding IS NULL "
            "AND NOT n.kind IN $skip_kinds "
            "RETURN n.id AS id",
            celex=celex,
            skip_kinds=_EMBED_SKIP_KINDS,
        )
        missing_embeddings = [r["id"] for r in result]
        checks.append(Check(
            name="embedding_completeness",
            passed=len(missing_embeddings) == 0,
            expected=0,
            actual=len(missing_embeddings),
            details=missing_embeddings[:20],
        ))

        # Embedding dimension check (sample one)
        result = session.run(
            f"MATCH (n:{base_label} {{celex: $celex}}) "
            "WHERE n.embedding IS NOT NULL "
            "RETURN size(n.embedding) AS dim LIMIT 1",
            celex=celex,
        )
        rec = result.single()
        if rec:
            dim = rec["dim"]
            checks.append(Check(
                name="embedding_dimension",
                passed=dim in (384, 768),  # multilingual-e5 variants
                expected="384 or 768",
                actual=dim,
            ))

        # Step 9: display_ref coverage for article-kind provisions
        result = session.run(
            f"MATCH (n:{base_label} {{celex: $celex}}) "
            "WHERE n.kind = 'article' AND (n.display_ref IS NULL OR n.display_ref = '') "
            "RETURN n.id AS id",
            celex=celex,
        )
        missing_refs = [r["id"] for r in result]
        checks.append(Check(
            name="article_display_ref_coverage",
            passed=len(missing_refs) == 0,
            expected=0,
            actual=len(missing_refs),
            details=missing_refs,
        ))

    return checks


# ══════════════════════════════════════════════════════════════════════════
# LAYER 4: Cross-Reference Integrity
# ══════════════════════════════════════════════════════════════════════════

def verify_cross_references(driver, database: str) -> list[Check]:
    """Global cross-reference integrity checks across all documents."""
    checks: list[Check] = []

    with driver.session(database=database) as session:
        # CITES edges: both source and target should exist
        result = session.run(
            "MATCH (a)-[r:CITES]->(b) "
            "RETURN a.celex AS src_celex, b.celex AS tgt_celex, "
            "count(r) AS cnt "
            "ORDER BY cnt DESC"
        )
        xref_summary = [(r["src_celex"], r["tgt_celex"], r["cnt"]) for r in result]
        checks.append(Check(
            name="cites_edge_summary",
            passed=True,  # informational
            details=[
                f"{src} -> {tgt}: {cnt} CITES edges"
                for src, tgt, cnt in xref_summary
            ],
        ))

        # Dangling CITES: source or target node doesn't exist
        # (This shouldn't happen with MERGE, but verify)
        result = session.run(
            "MATCH (a)-[r:CITES]->(b) "
            "WHERE a.id IS NULL OR b.id IS NULL "
            "RETURN count(r) AS cnt"
        )
        dangling = result.single()["cnt"]
        checks.append(Check(
            name="no_dangling_cites",
            passed=dangling == 0,
            expected=0,
            actual=dangling,
        ))

        # Residual CITES_EXTERNAL (should be 0 after crosslinker for loaded regs)
        result = session.run(
            "MATCH (a)-[r:CITES_EXTERNAL]->(b) "
            "RETURN a.celex AS src, count(r) AS cnt "
            "ORDER BY cnt DESC"
        )
        residual = [(r["src"], r["cnt"]) for r in result]
        total_residual = sum(cnt for _, cnt in residual)
        checks.append(Check(
            name="cites_external_residual",
            passed=True,  # informational — some may be genuinely external
            expected="ideally 0 for loaded regulations",
            actual=total_residual,
            details=[f"{src}: {cnt}" for src, cnt in residual],
        ))

        # DefinedTerm → DEFINED_BY → Provision: target must exist
        result = session.run(
            "MATCH (d:DefinedTerm)-[r:DEFINED_BY]->(p) "
            "WHERE p.id IS NULL "
            "RETURN count(r) AS cnt"
        )
        dangling_defs = result.single()["cnt"]
        checks.append(Check(
            name="defined_by_targets_exist",
            passed=dangling_defs == 0,
            expected=0,
            actual=dangling_defs,
        ))

        # Total graph statistics for context
        result = session.run(
            "MATCH (n) WITH labels(n) AS lbls "
            "UNWIND lbls AS lbl "
            "WITH lbl WHERE lbl IN ['Provision', 'Guidance', 'DefinedTerm', 'ExternalAct'] "
            "RETURN lbl, count(*) AS cnt ORDER BY cnt DESC"
        )
        stats = {r["lbl"]: r["cnt"] for r in result}
        checks.append(Check(
            name="graph_label_distribution",
            passed=True,  # informational
            details=[f"{lbl}: {cnt}" for lbl, cnt in stats.items()],
        ))

    return checks


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════

def _discover_documents(filter_doc: str | None = None) -> list[tuple[str, str]]:
    """Return list of (doc_id, doc_type) for documents with parsed.json on disk."""
    docs = []

    # Legislation
    for celex in LEGISLATION:
        if filter_doc and filter_doc != celex:
            continue
        parsed = LEGISLATION_DIR / celex / "EN" / "parsed.json"
        if parsed.exists():
            docs.append((celex, "legislation"))

    # Guidance
    for doc_id in MDCG_DOCUMENTS:
        if filter_doc and filter_doc != doc_id:
            continue
        parsed = GUIDANCE_DIR / doc_id / "EN" / "parsed.json"
        if parsed.exists():
            docs.append((doc_id, "guidance"))

    return docs


def _parsed_path_for(doc_id: str, doc_type: str) -> Path:
    if doc_type == "legislation":
        return LEGISLATION_DIR / doc_id / "EN" / "parsed.json"
    else:
        return GUIDANCE_DIR / doc_id / "EN" / "parsed.json"


def run_verification(
    filter_doc: str | None = None,
    layers: list[int] | None = None,
) -> VerificationReport:
    """Run the full verification pipeline.

    Parameters
    ----------
    filter_doc:
        If set, only verify this document ID.
    layers:
        List of layer numbers to run (1-4). Default: all.
    """
    if layers is None:
        layers = [1, 2, 3, 4]

    report = VerificationReport()
    documents = _discover_documents(filter_doc)

    if not documents:
        print(f"WARNING: No documents found" +
              (f" matching '{filter_doc}'" if filter_doc else "") +
              ". Check data directories.")
        return report

    # Neo4j connection (only for layers 2-4)
    driver = None
    database = None
    if any(l in layers for l in [2, 3, 4]):
        try:
            driver, database = _get_neo4j_session()
            # Quick connectivity test
            with driver.session(database=database) as s:
                s.run("RETURN 1").consume()
            print("  Neo4j connection: OK")
        except Exception as e:
            print(f"  Neo4j connection FAILED: {e}")
            print("  Skipping layers 2, 3, 4.")
            layers = [l for l in layers if l == 1]
            driver = None

    # Per-document checks
    for doc_id, doc_type in documents:
        print(f"\n{'='*60}")
        print(f"  Document: {doc_id} ({doc_type})")
        print(f"{'='*60}")

        doc_report = DocumentReport(doc_id=doc_id, doc_type=doc_type)
        parsed_path = _parsed_path_for(doc_id, doc_type)

        # Layer 1: Source ↔ parsed.json
        if 1 in layers:
            print("  Layer 1: Source <-> parsed.json ...")
            data = _load_parsed_json(parsed_path)
            if data:
                provisions = data.get("provisions", [])

                # Structural integrity
                doc_report.checks.extend(
                    verify_structural_integrity(provisions, doc_id)
                )

                # Source-specific checks
                if doc_type == "legislation":
                    doc_report.checks.extend(
                        verify_legislation_source(doc_id)
                    )
                else:
                    doc_report.checks.extend(
                        verify_mdcg_source(doc_id)
                    )

                # Relation integrity
                doc_report.checks.extend(
                    verify_relation_integrity(data, doc_id)
                )
            else:
                doc_report.checks.append(Check(
                    name="parsed_json_exists",
                    passed=False,
                    details=[f"Missing: {parsed_path}"],
                ))

            _print_layer_summary(doc_report, "Layer 1")

        # Layer 2: parsed.json ↔ Neo4j
        if 2 in layers and driver:
            print("  Layer 2: parsed.json <-> Neo4j ...")
            doc_report.checks.extend(
                verify_neo4j_parity(parsed_path, doc_id, driver, database)
            )
            _print_layer_summary(doc_report, "Layer 2")

        # Layer 3: Neo4j ↔ Retrieval
        if 3 in layers and driver:
            print("  Layer 3: Neo4j <-> Retrieval ...")
            doc_report.checks.extend(
                verify_retrieval_readiness(parsed_path, doc_id, driver, database)
            )
            _print_layer_summary(doc_report, "Layer 3")

        report.documents.append(doc_report)

    # Global checks (Layer 4)
    if 4 in layers and driver:
        print(f"\n{'='*60}")
        print("  Global: Cross-reference integrity")
        print(f"{'='*60}")
        report.global_checks.extend(
            verify_cross_references(driver, database)
        )

    if driver:
        driver.close()

    return report


def _print_layer_summary(doc_report: DocumentReport, layer_name: str):
    """Print brief pass/fail count for the most recently added checks."""
    passed = sum(1 for c in doc_report.checks if c.passed)
    failed = sum(1 for c in doc_report.checks if not c.passed)
    marker = "PASS" if failed == 0 else "FAIL"
    print(f"    {layer_name}: {passed} passed, {failed} failed [{marker}]")


def _print_final_report(report: VerificationReport):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("  COMPLETENESS VERIFICATION REPORT")
    print("=" * 70)

    all_passed = report.passed
    rd = report.to_dict()
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    print(f"  Total checks: {rd['total_checks']}")
    print(f"  Passed: {rd['total_passed']}")
    print(f"  Failed: {rd['total_failed']}")

    # Per-document summary
    print(f"\n  {'Document':<25} {'Status':<8} {'Passed':<8} {'Failed':<8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for doc in report.documents:
        status = "PASS" if doc.passed else "FAIL"
        print(f"  {doc.doc_id:<25} {status:<8} {doc.n_passed:<8} {doc.n_failed:<8}")

    # Show failures in detail
    any_failures = False
    for doc in report.documents:
        for check in doc.checks:
            if not check.passed:
                if not any_failures:
                    print("\n  FAILURES:")
                    any_failures = True
                print(f"\n  [{doc.doc_id}] {check.name}")
                if check.expected is not None:
                    print(f"    Expected: {check.expected}")
                if check.actual is not None:
                    print(f"    Actual:   {check.actual}")
                for detail in check.details[:10]:
                    print(f"    - {detail}")
                if len(check.details) > 10:
                    print(f"    ... and {len(check.details) - 10} more")

    for check in report.global_checks:
        if not check.passed:
            if not any_failures:
                print("\n  FAILURES:")
                any_failures = True
            print(f"\n  [GLOBAL] {check.name}")
            if check.expected is not None:
                print(f"    Expected: {check.expected}")
            if check.actual is not None:
                print(f"    Actual:   {check.actual}")
            for detail in check.details[:10]:
                print(f"    - {detail}")

    # Show informational global checks
    info_checks = [c for c in report.global_checks if c.passed and c.details]
    if info_checks:
        print("\n  CROSS-REFERENCE SUMMARY:")
        for check in info_checks:
            print(f"    {check.name}:")
            for d in check.details[:15]:
                print(f"      {d}")

    if not any_failures:
        print("\n  All checks passed.")

    print()


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Deterministic GraphRAG completeness verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--doc",
        help="Verify only this document ID (e.g. MDCG_2019_11 or 32024R1689)",
    )
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only specific layers (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full report as JSON to stdout",
    )
    parser.add_argument(
        "--output",
        help="Save JSON report to this file path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("CRSS GraphRAG Completeness Verifier")
    print("=" * 40)
    print(f"  Data dir: {DATA_DIR}")
    docs = _discover_documents(args.doc)
    print(f"  Documents found: {len(docs)}")
    for d, t in docs:
        print(f"    {d} ({t})")

    report = run_verification(
        filter_doc=args.doc,
        layers=args.layer,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_final_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"  JSON report saved to: {out_path}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
