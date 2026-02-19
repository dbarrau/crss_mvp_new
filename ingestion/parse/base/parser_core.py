"""Shared helpers for regulation-specific parsers.

These utilities aim to keep MDR, EU AI Act, and future regulation
parsers DRY by centralising stack/path handling, canonical ID
construction, and provision dict assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from bs4 import Tag

from .cross_refs import extract_references
from .hierarchy import LEVEL_ORDER
from .parser_utils import ROLE_DETECTOR, make_provenance
from .requirement_patterns import classify_requirement_type, is_requirement_text


@dataclass(frozen=True)
class ParserConfig:
    """Configuration describing shared parser behaviour."""

    celex_id: str
    source_name: str
    regulation_id: str
    parser_name: str
    parser_version: str = "0.2"
    level_normalization: Dict[str, str] = field(default_factory=dict)
    context_levels: Sequence[str] = ("title", "chapter", "section", "article")
    non_requirement_levels: Set[str] = field(
        default_factory=lambda: {"title", "chapter", "section", "article", "annex", "recital"}
    )
    reference_excluded_levels: Set[str] = field(
        default_factory=lambda: {"title", "chapter", "section", "article", "annex"}
    )
    path_skip_levels: Set[str] = field(default_factory=lambda: {"title"})
    role_detector: Optional[Callable[[str, str], Sequence[str]]] = None

    def resolve_role_detector(self) -> Callable[[str, str], Sequence[str]]:
        return self.role_detector or ROLE_DETECTOR


def canonicalize_numbered_level(level: Optional[str], config: ParserConfig) -> Optional[str]:
    """Normalize ad-hoc numbering levels to canonical hierarchy labels."""

    if not level:
        return None
    return config.level_normalization.get(level, level)


def make_provision_id(
    config: ParserConfig, parent_id: Optional[str], level: str, marker: Optional[str], lang: str
) -> str:
    marker_token = str(marker if marker is not None else "UNNUMBERED").replace(" ", "_")
    if parent_id:
        return f"{parent_id}_{level}_{marker_token}"
    return f"{config.celex_id}_{lang}_{level}_{marker_token}"


def make_path_segment(level: Optional[str], marker: Optional[str]) -> Optional[str]:
    if not level:
        return None
    if marker is None:
        return level.upper()
    return f"{level.upper()}_{str(marker).replace(' ', '_').upper()}"


def build_path(stack: List[Dict], level: str, marker: Optional[str], path_skip_levels: Set[str]) -> List[str]:
    segments: List[str] = []
    for ancestor in stack:
        if ancestor["level"] in path_skip_levels:
            continue
        seg = make_path_segment(ancestor["level"], ancestor.get("item_number"))
        if seg:
            segments.append(seg)
    cur_seg = make_path_segment(level, marker)
    if cur_seg:
        segments.append(cur_seg)
    return segments


def build_context(stack: List[Dict], current: Dict, config: ParserConfig) -> Dict[str, Optional[str]]:
    chain = stack + [current]
    context = {lvl: None for lvl in config.context_levels}
    context["root_id"] = chain[0].get("id") if chain else current.get("id")
    for node in chain:
        lvl = node.get("level")
        if lvl in context:
            context[lvl] = node.get("title") or node.get("item_number")
    return context


def _compute_requirement_fields(
    config: ParserConfig,
    level: str,
    text_for_analysis: str,
    lang: str,
) -> Dict[str, object]:
    if level in config.non_requirement_levels:
        return {
            "is_requirement": False,
            "requirement_type": "other",
            "roles": [],
        }

    roles = list(config.resolve_role_detector()(text_for_analysis, lang) or [])
    return {
        "is_requirement": is_requirement_text(text_for_analysis, lang),
        "requirement_type": classify_requirement_type(text_for_analysis, lang),
        "roles": roles,
    }


def _compute_references(config: ParserConfig, level: str, text: str, lang: str) -> List[str]:
    if level in config.reference_excluded_levels:
        return []
    return extract_references(text, lang)


def build_provision_record(
    *,
    config: ParserConfig,
    stack: List[Dict],
    level: str,
    marker: Optional[str],
    lang: str,
    title: Optional[str],
    text: str,
    intro_text: str,
    parent_id: Optional[str],
    raw_html: str,
    tag: Tag,
    html_file: Path,
    text_for_analysis: Optional[str] = None,
    references_text: Optional[str] = None,
    metadata_extra: Optional[Dict] = None,
    regulation_metadata: Optional[Dict] = None,
) -> Dict:
    text_for_analysis = text_for_analysis if text_for_analysis is not None else text
    references_text = references_text if references_text is not None else text_for_analysis

    provision_id = make_provision_id(config, parent_id, level, marker, lang)
    path = build_path(stack, level, marker, config.path_skip_levels)
    req_fields = _compute_requirement_fields(config, level, text_for_analysis, lang)
    references = _compute_references(config, level, references_text, lang)

    snippet_end = min(len(text_for_analysis), 240)
    snippet = text_for_analysis[:240] + ("..." if len(text_for_analysis) > 240 else "")

    provision = {
        "id": provision_id,
        "parent_id": parent_id,
        "level": level,
        "kind": level,
        "item_number": marker,
        "lang": lang,
        "celex": config.celex_id,
        "regulation_id": config.regulation_id,
        "source": config.source_name,
        "title": title,
        "text": text,
        "intro_text": intro_text,
        "path": path,
        "path_string": "/".join(path) if path else "",
        "depth": len(path),
        "canonical_id": provision_id,
        "canonical_tags": [
            f"celex:{config.celex_id}",
            f"lang:{lang}",
            f"level:{level}",
            f"requirement:{'yes' if req_fields['is_requirement'] else 'no'}",
            f"requirement_type:{req_fields['requirement_type']}",
        ],
        "obligations": [],
        "snippet": snippet,
        "snippet_char_offsets": {"start": 0, "end": snippet_end},
        "embedding_id": f"{provision_id}_emb_0",
        "metadata": {},
        "references": references,
        "provenance": make_provenance(raw_html, tag, config.parser_name, config.parser_version, html_file),
        "regulation_metadata": regulation_metadata,
        **req_fields,
    }

    context = build_context(stack, provision, config)
    metadata = {
        "celex_id": config.celex_id,
        "source": config.source_name,
        "lang": lang,
        **context,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    provision["metadata"] = metadata

    return provision


def record_relation(relations: List[Tuple[str, str, str]], source: str, relation_type: str, target: str) -> None:
    relations.append((source, relation_type, target))


def flatten_relations(relations: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    return [
        {"source": src, "type": rel_type, "target": tgt}
        for (src, rel_type, tgt) in relations
    ]
