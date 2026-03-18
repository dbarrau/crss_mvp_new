"""Regulatory compliance Q&A agent backed by Neo4j graph retrieval + Mistral.

Provides the :func:`ask` function that:
1. Detects regulatory terms in the question and fetches legal definitions
2. Retrieves relevant provisions from the knowledge graph
3. Assembles structured context with definitions and cross-references
4. Sends to Mistral (EU-hosted) for a grounded answer
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a European regulatory compliance expert specializing in \
MDR 2017/745, IVDR 2017/746, and the EU AI Act (Regulation 2024/1689).

You answer questions STRICTLY and ONLY based on the REGULATORY CONTEXT provided below.

CRITICAL GROUNDING RULES:
- Every paragraph number, definition, article reference, annex section, recital \
number, and quoted text MUST appear explicitly in the REGULATORY CONTEXT.
- NEVER supply regulatory details (paragraph numbers, definitions, subparagraph \
ordinals, recital numbers, annex rule numbers) from your training memory — these \
are version-sensitive and your training data may be wrong or refer to a \
different version of the legislation.
- When citing a paragraph number, it MUST match a number that appears literally \
in the provision text shown (e.g. "4.   'active device' means..." → paragraph 4).
- If the context lacks a specific detail the question asks about, state exactly: \
"The retrieved context does not include [specific item]." Do NOT fill gaps from memory.
- Cross-references are acceptable ONLY if they appear in the "Cross-references" \
section of the provided context.
- When listing sub-items of a provision (points labelled (a), (b)… or roman items \
labelled (i), (ii), (iii)…), you MUST preserve their exact original order and \
labels exactly as they appear in the REGULATORY CONTEXT. Do NOT reorder, \
renumber, or relabel them under any circumstances — not by severity, importance, \
or any other criterion.
- When a sub-item text contains a qualifying reference (e.g. "offences referred \
to in Annex II"), that qualifier MUST be reproduced in full; do not paraphrase \
it away.

Format your answer with:
1. A direct answer based solely on the provided context
2. Relevant quotes or paraphrases, each labelled with the provision reference \
shown in the context header (e.g. "[1] Article 2")
3. Cross-references that appear explicitly in the context
"""


# Truncate rolled-up article/annex body to prevent definition-heavy articles
# (e.g. Article 2 with 65 definitions) from flooding the context window and
# crowding out actually-relevant children.
_BODY_LIMIT = 1200

# Maximum number of definition terms to inject into the context.
_MAX_DEFINITIONS = 5


def _detect_defined_terms(
    question: str, retriever,
) -> list[dict]:
    """Identify regulatory defined terms mentioned in *question*.

    Matches against the DefinedTerm index cached on the retriever.
    Returns ``find_by_term()`` results for every matched term (longest
    terms matched first to avoid partial-match shadowing).
    """
    try:
        term_index = retriever.get_defined_terms_index()
    except Exception:
        logger.debug("Could not load defined-terms index; skipping.", exc_info=True)
        return []

    q_lower = question.lower()
    matched: list[dict] = []
    seen_terms: set[str] = set()

    # Sort longest-first so "high-risk AI system" matches before "AI system"
    for term_lower, _tn in sorted(
        term_index.items(), key=lambda x: len(x[0]), reverse=True,
    ):
        # Word-boundary match to avoid spurious substring hits
        if re.search(r"\b" + re.escape(term_lower) + r"\b", q_lower):
            if term_lower in seen_terms:
                continue
            seen_terms.add(term_lower)
            results = retriever.find_by_term(term_lower)
            matched.extend(results)
            if len(matched) >= _MAX_DEFINITIONS:
                break

    return matched[:_MAX_DEFINITIONS]


def _format_definitions(definitions: list[dict]) -> str:
    """Format definition lookup results as a context block for the LLM."""
    parts: list[str] = []
    for d in definitions:
        reg = d.get("regulation", "")
        ref = d.get("article_ref", "")
        term = d.get("term", "")
        text = d.get("definition_text", "")
        label = f"Definition of \u2018{term}\u2019"
        if ref:
            label += f" — {ref}"
        if reg:
            label += f" ({reg})"
        parts.append(f"{label}:\n{text}")
    return "\n\n".join(parts)


def _format_context(provisions: list[dict]) -> str:
    """Turn retriever results into a structured text block for the LLM."""
    parts: list[str] = []
    for i, p in enumerate(provisions, 1):
        regulation = p.get("regulation", "")
        header = f"[{i}] {p.get('article_ref', 'Unknown')} ({regulation})"
        path = p.get("article_path", "")
        if path:
            header += f"\n    Path: {path}"

        body = p.get("article_text", "") or ""
        if len(body) > _BODY_LIMIT:
            body = body[:_BODY_LIMIT] + " […see paragraph details below…]"

        # Child provisions — use raw provision text so paragraph numbers are
        # unambiguous (e.g. "4.   'active device' means..."), without the
        # repeated ancestry prefix that obscures the numbering.
        children = p.get("children") or []
        matched_leaf = p.get("matched_leaf_id")
        child_lines: list[str] = []
        matched_lines: list[str] = []
        for c in children:
            ref = c.get("ref") or c.get("kind", "")
            is_match = bool(matched_leaf and c.get("id") == matched_leaf)
            text = (c.get("raw_text") or c.get("text") or "")[:(1000 if is_match else 400)]
            if text:
                if is_match:
                    matched_lines.append(f"  [\u2605 MATCHED] {ref}: {text}")
                else:
                    child_lines.append(f"  {ref}: {text}")
        child_lines = (matched_lines + child_lines)[:15]

        # Cross-referenced provisions
        cited = p.get("cited_provisions") or []
        cite_lines: list[str] = []
        for c in cited:
            ref = c.get("ref", "")
            text = (c.get("text") or "")[:300]
            if text:
                cite_lines.append(f"  -> {ref}: {text}")

        section = header + "\n" + body
        if child_lines:
            section += "\n\nParagraphs/Points:\n" + "\n".join(child_lines)
        if cite_lines:
            section += "\n\nCross-references:\n" + "\n".join(cite_lines)
        parts.append(section)

    return "\n\n---\n\n".join(parts)


def ask(question: str, retriever, k: int = 5) -> str:
    """Retrieve context from the graph and generate an answer via Mistral.

    Parameters
    ----------
    question:
        The user's natural-language question.
    retriever:
        A :class:`retrieval.graph_retriever.GraphRetriever` instance.
    k:
        Number of top provisions to retrieve.

    Returns
    -------
    str
        The LLM-generated answer grounded in regulatory text.
    """
    from mistralai.client import Mistral

    # --- 1. Fetch legal definitions for terms mentioned in the question ---
    definitions = _detect_defined_terms(question, retriever)
    if definitions:
        logger.info(
            "Injecting %d definition(s): %s",
            len(definitions),
            ", ".join(d.get("term", "?") for d in definitions),
        )

    # --- 2. Vector + graph retrieval ---
    provisions = retriever.retrieve(question, k=k)
    if not provisions and not definitions:
        return (
            "No relevant provisions were found in the knowledge graph. "
            "Please check that embeddings have been generated "
            "(run scripts/embed_provisions.py)."
        )

    # --- 3. Assemble context: definitions first, then provisions ---
    context_parts: list[str] = []
    if definitions:
        context_parts.append(
            "LEGAL DEFINITIONS (from the definitions article):\n"
            + _format_definitions(definitions)
        )
    if provisions:
        context_parts.append(_format_context(provisions))
    context = "\n\n---\n\n".join(context_parts)

    logger.debug(
        "Context assembled: %d provisions + %d definitions, %d chars",
        len(provisions), len(definitions), len(context),
    )

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model=os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"REGULATORY CONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}"
                ),
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content
