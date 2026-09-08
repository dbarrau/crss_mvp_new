"""Compute the *superseded-provision* record from a consolidation.

When an amending act DELETES a provision (the Digital Omnibus deletes AI Act
Article 10(5)), the consolidated graph simply loses the node — current law, but
it erases the fact that the number ever existed. The model, trained on the
pre-amendment text, then cites the deleted provision as if it were still in force
(observed: "process special categories under Article 10(5)"), and no runtime guard
catches it: the phantom guard is article-grained (Article 10 exists → 10(5)
passes) and the faithfulness check only inspects quotes.

This module turns the consolidation's own operation report into a durable record
of *what was deleted, by whom, and where its substance went* — the receipt the
applier currently computes and discards. It is the single source of truth for the
runtime superseded-citation guard (``application/_superseded``).

Two facts, from two places (see the delete op vs. the relocation):

* **deletion** — certain: a ``delete`` :class:`OpResult` carries the deleted ref,
  its node id and the amending act's amendment-list point (e.g. Omnibus Art 1
  point (9) deletes Article 10(5)).
* **relocation ("see …")** — derived and *gated*: the delete op does not say where
  the content went (delete and insert are separate operations). We match the
  deleted provision's subject against the articles the SAME act inserted; a
  confident heading match yields the new home (Article 4a, then its best-matching
  paragraph 4a(1)). No confident match → ``see_ref`` is ``None`` and the guard
  says only "deleted by <act>". A wrong "see" is never emitted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from consolidation.amendment_parser import parse_amending_regulation
from consolidation.applier import consolidate
from consolidation.build import _parsed_path, _DATA_ROOT
from domain.legislation_catalog import LEGISLATION

# Recall of an inserted article heading's distinctive terms inside the deleted
# provision's text, above which the insert is accepted as the relocation home.
_RELOCATION_MIN_RECALL = 0.6
_STOP = frozenset(
    "the a an of to in on for with by and or as is are be shall this that which "
    "such at from into under within not no all any its their to of".split()
)


@dataclass(frozen=True)
class SupersededRecord:
    """One provision deleted by an amendment, with its relocation (if any)."""
    celex: str                 # act the deleted provision belongs to (base act)
    deleted_ref: str           # human ref, e.g. "Article 10(5)"
    deleted_id: str            # node id, e.g. "32024R1689_010.005"
    amender_celex: str         # e.g. "32026R1744"
    amender_name: str          # e.g. "Digital Omnibus on AI"
    point_num: str             # amending act's amendment-list point, e.g. "9"
    see_ref: Optional[str] = None   # relocation home, e.g. "Article 4a(1)"
    see_id: Optional[str] = None


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in _STOP}


def _derive_ref(node_id: str, celex: str) -> Optional[str]:
    """Human ref from a node id, since inserted nodes get their ``display_ref``
    only at graph-load: ``32024R1689_004a.001`` → "Article 4a(1)",
    ``32024R1689_art_4a`` → "Article 4a"."""
    rest = node_id[len(celex) + 1:] if node_id.startswith(celex) else node_id
    m = re.match(r"^0*(\d+[a-z]?)\.0*(\d+)$", rest)
    if m:
        return f"Article {m.group(1)}({int(m.group(2))})"
    m = re.match(r"^art_(\d+[a-z]?)$", rest)
    if m:
        return f"Article {m.group(1)}"
    return None


def _node_text(node: dict, by_id: dict) -> str:
    """A node's own text plus its descendants' — the full subtree body."""
    parts = [node.get("text") or "", node.get("title") or ""]
    for cid in node.get("children", []) or []:
        child = by_id.get(cid)
        if child:
            parts.append(_node_text(child, by_id))
    return " ".join(p for p in parts if p)


def _best_relocation(
    deleted_terms: set[str],
    inserted_articles: List[dict],
    cons_by_id: dict,
    celex: str,
) -> tuple[Optional[str], Optional[str]]:
    """Pick the inserted article whose heading is best covered by the deleted
    provision's terms, then its best-matching paragraph. Gated: returns
    ``(None, None)`` unless the heading recall clears ``_RELOCATION_MIN_RECALL``."""
    best_article, best_recall = None, 0.0
    for art in inserted_articles:
        heading_terms = _terms(art.get("title") or art.get("text") or "")
        if not heading_terms:
            continue
        recall = len(heading_terms & deleted_terms) / len(heading_terms)
        if recall > best_recall:
            best_article, best_recall = art, recall
    if best_article is None or best_recall < _RELOCATION_MIN_RECALL:
        return None, None
    # Refine to the paragraph of that article whose text best overlaps the
    # deleted provision (Article 4a(1) — high-risk providers — over 4a(2)).
    para_children = [
        cons_by_id[c] for c in best_article.get("children", []) or []
        if c in cons_by_id and cons_by_id[c].get("kind") == "paragraph"
        and (cons_by_id[c].get("number") or "").strip()
    ]
    best_para, best_overlap = None, 0
    for para in para_children:
        overlap = len(_terms(_node_text(para, cons_by_id)) & deleted_terms)
        if overlap > best_overlap:
            best_para, best_overlap = para, overlap
    target = best_para or best_article
    tid = target.get("id") or ""
    ref = target.get("display_ref") or _derive_ref(tid, celex)
    return ref, tid


def _celex_of(node_id: str) -> str:
    m = re.match(r"^(\d{5}[A-Z]\d{4})_", node_id or "")
    return m.group(1) if m else ""


def compute_superseded(
    base_celex: str,
    amender_celex: str,
    lang: str = "EN",
    data_root: Optional[Path] = None,
) -> List[SupersededRecord]:
    """Return the :class:`SupersededRecord`s for one ``base ← amender`` build.

    Runs the consolidation report-only (writes nothing): every applied ``delete``
    becomes a record, with the relocation home derived from the same act's
    inserted articles (gated)."""
    root = Path(data_root) if data_root else _DATA_ROOT
    import json
    base = json.loads(_parsed_path(base_celex, lang, root).read_text(encoding="utf-8"))
    base_by_id = {n["id"]: n for n in base["provisions"]}

    instructions = parse_amending_regulation(amender_celex, lang, root)
    consolidated, report = consolidate(base["provisions"], instructions, amender_celex)
    cons_by_id = {n["id"]: n for n in consolidated}
    inserted_articles = [
        n for n in consolidated
        if n.get("kind") == "article" and n["id"] not in base_by_id
        and n.get("amended_by") == amender_celex
    ]
    amender_name = LEGISLATION.get(amender_celex, {}).get("name") or amender_celex

    records: List[SupersededRecord] = []
    for r in report:
        if r.op != "delete" or r.status != "applied":
            continue
        del_id = r.node_ids[0] if r.node_ids else None
        del_node = base_by_id.get(del_id) if del_id else None
        see_ref = see_id = None
        if del_node is not None:
            see_ref, see_id = _best_relocation(
                _terms(_node_text(del_node, base_by_id)),
                inserted_articles, cons_by_id, base_celex,
            )
        records.append(SupersededRecord(
            celex=base_celex, deleted_ref=r.target_ref, deleted_id=del_id or "",
            amender_celex=amender_celex, amender_name=amender_name,
            point_num=str(r.point_num), see_ref=see_ref, see_id=see_id,
        ))
    return records


def record_dicts(records: List[SupersededRecord]) -> List[dict]:
    return [asdict(r) for r in records]
