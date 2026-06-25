"""Defined-term detection and context expansion.

Identifies regulatory defined terms mentioned in a question or in already-retrieved
provisions, then fetches their formal definitions from the graph index.
No LLM calls; no Mistral I/O.
"""
from __future__ import annotations

import logging
import re

from application._config import (
    _MAX_DEFINITIONS,
    _MAX_RELATED_DEFINITIONS,
    _RELATED_DEFINITION_SCAN_LIMIT,
    _detect_mentioned_regulations,
)
from domain.legislation_catalog import (
    AI_ACT_CELEX,
    MDR_CELEX,
    IVDR_CELEX,
    GDPR_CELEX,
)

logger = logging.getLogger(__name__)

# Foundational subject-matter definition of each regulation, keyed by CELEX.
# When a regulation is explicitly in scope (its CELEX is in ``target_celexes``)
# its root definition is the most load-bearing anchor in the answer — yet being
# short, these terms lose the longest-first race against the
# ``_MAX_RELATED_DEFINITIONS`` cap and used to be dropped from context, forcing
# the LLM to backfill them from training memory (the AI Act Article 3(1)
# 'AI system' fallback).  These anchors are force-injected past the cap.
_ANCHOR_DEFINITION_TERMS: dict[str, str] = {
    AI_ACT_CELEX: "ai system",                           # AI Act, Article 3(1)
    MDR_CELEX: "medical device",                         # MDR, Article 2(1)
    IVDR_CELEX: "in vitro diagnostic medical device",    # IVDR, Article 2(2)
    GDPR_CELEX: "personal data",                         # GDPR, Article 4(1)
}


def _term_match_pattern(term_lower: str) -> re.Pattern:
    """Compile a word-boundary regex for *term_lower* that also matches its plural.

    Defined-term index keys are stored in canonical singular form (e.g.
    ``"ai system"``), but regulation and question text overwhelmingly use the
    plural (e.g. ``"high-risk AI systems"``).  A naive ``\\bai system\\b`` misses
    every plural occurrence — which is exactly how the AI Act's Article 3(1)
    'AI system' definition was silently dropped from cross-regulation answers,
    forcing the LLM to backfill it from training memory.  Allowing an optional
    trailing ``s`` on the final token closes that gap without the false
    positives of unbounded substring matching.
    """
    return re.compile(r"\b" + re.escape(term_lower) + r"s?\b", re.IGNORECASE)


def _detect_defined_terms(
    question: str, retriever,
) -> list[dict]:
    """Identify regulatory defined terms mentioned in *question*.

    Matches against the DefinedTerm index cached on the retriever.
    Returns ``find_by_term()`` results for every matched term (longest
    terms matched first to avoid partial-match shadowing).

    When the same term has definitions in multiple regulations,
    only the definition from the regulation(s) mentioned in the
    question is kept (or one arbitrary definition if none match).
    """
    try:
        term_index = retriever.get_defined_terms_index()
    except Exception:
        logger.debug("Could not load defined-terms index; skipping.", exc_info=True)
        return []

    mentioned_regs = _detect_mentioned_regulations(question)
    q_lower = question.lower()
    matched: list[dict] = []
    seen_terms: set[str] = set()

    # Sort longest-first so "high-risk AI system" matches before "AI system"
    for term_lower, _tn in sorted(
        term_index.items(), key=lambda x: len(x[0]), reverse=True,
    ):
        # Word-boundary match (plural-aware) to avoid spurious substring hits
        if _term_match_pattern(term_lower).search(q_lower):
            if term_lower in seen_terms:
                continue
            seen_terms.add(term_lower)
            results = retriever.find_by_term(term_lower)
            # Deduplicate: keep one definition per term, preferring
            # definitions from regulations mentioned in the question.
            if mentioned_regs:
                preferred = [
                    r for r in results
                    if r.get("regulation") in mentioned_regs
                ]
                if preferred:
                    results = preferred[:1]
                else:
                    results = results[:1]
            else:
                results = results[:1]
            matched.extend(results)
            if len(matched) >= _MAX_DEFINITIONS:
                break

    return matched[:_MAX_DEFINITIONS]


def _expand_definitions_from_provisions(
    provisions: list[dict],
    retriever,
    existing: list[dict],
    target_celexes: set[str] | None = None,
) -> list[dict]:
    """Add formal definitions for defined terms mentioned in retrieved context.

    This captures cases where the user asks about a provision that relies on a
    formally defined concept without naming that concept explicitly in the
    question. For example, Article 25(1)(b) uses "substantial modification",
    whose formal definition lives in Article 3(23).
    """
    try:
        term_index = retriever.get_defined_terms_index()
    except Exception:
        logger.debug(
            "Could not load defined-terms index for context expansion.",
            exc_info=True,
        )
        return existing

    seen_terms = {d.get("term", "").lower() for d in existing}
    expanded = list(existing)

    text_parts: list[str] = []
    for prov in provisions[:_RELATED_DEFINITION_SCAN_LIMIT]:
        article_text = prov.get("article_text") or ""
        if article_text:
            text_parts.append(article_text)
        matched_leaf = prov.get("matched_leaf_id")
        if not matched_leaf:
            continue
        for child in prov.get("children") or []:
            if child.get("id") != matched_leaf:
                continue
            child_text = child.get("raw_text") or child.get("text") or ""
            if child_text:
                text_parts.append(child_text)
            break

    context_text = "\n".join(text_parts).lower()
    if not context_text:
        return expanded

    added = 0
    for term_lower, _tn in sorted(
        term_index.items(), key=lambda x: len(x[0]), reverse=True,
    ):
        if term_lower in seen_terms:
            continue
        if not _term_match_pattern(term_lower).search(context_text):
            continue

        results = retriever.find_by_term(term_lower)
        if target_celexes:
            filtered = [r for r in results if r.get("celex") in target_celexes]
            if filtered:
                results = filtered
        formal = [r for r in results if r.get("definition_type") == "formal"]
        if formal:
            results = formal
        if not results:
            continue

        expanded.append(results[0])
        seen_terms.add(term_lower)
        added += 1
        if added >= _MAX_RELATED_DEFINITIONS:
            break

    if added:
        logger.info(
            "Expanded context with %d related definition(s): %s",
            added,
            ", ".join(d.get("term", "?") for d in expanded[len(existing):]),
        )

    # Force-inject each in-scope regulation's foundational definition past the
    # cap above.  Prepended so the subject-matter anchor leads the definitions
    # block; skipped when the term was already captured by the scan or has no
    # graph match.
    if target_celexes:
        for celex in sorted(target_celexes):
            anchor = _ANCHOR_DEFINITION_TERMS.get(celex)
            if not anchor or anchor in seen_terms:
                continue
            results = retriever.find_by_term(anchor)
            same_reg = [r for r in results if r.get("celex") == celex]
            results = same_reg or results
            formal = [r for r in results if r.get("definition_type") == "formal"]
            results = formal or results
            if not results:
                continue
            expanded.insert(0, results[0])
            seen_terms.add(anchor)
            logger.info(
                "Force-injected anchor definition '%s' for %s.", anchor, celex,
            )

    return expanded
