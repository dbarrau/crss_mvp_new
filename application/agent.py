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
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a European regulatory compliance expert specializing in \
MDR 2017/745, IVDR 2017/746, and the EU AI Act (Regulation 2024/1689).

You answer questions based on the REGULATORY CONTEXT provided below.

TEXTUAL GROUNDING RULES (strict):
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

REGULATORY REASONING (permitted):
- While you must quote text strictly from the provided context, you SHOULD use \
your expert understanding of European law to draw logical inferences from the \
provisions shown. For example, if a provision states that a notified body \
"shall apply" certain requirements, you may conclude that the notified body is \
permitted and required to perform that assessment.
- You may resolve procedural overlaps between regulations (e.g., how the AI Act \
integrates with the MDR conformity assessment framework) by reasoning over the \
provisions in context.
- Clearly distinguish between what the text explicitly states (quote it) and \
what you logically infer from the text (label it as an inference).

Format your answer with:
1. A direct answer based on the provided context and sound regulatory reasoning
2. Relevant quotes or paraphrases, each labelled with the provision reference \
shown in the context header (e.g. "[1] Article 2")
3. Cross-references that appear explicitly in the context
"""


# Truncate rolled-up article/annex body to prevent definition-heavy articles
# (e.g. Article 2 with 65 definitions) from flooding the context window and
# crowding out actually-relevant children.
_BODY_LIMIT = 2000

# Maximum number of definition terms to inject into the context.
_MAX_DEFINITIONS = 5

# Regulation name → CELEX lookup for multi-regulation retrieval.
_REG_NAME_TO_CELEX: dict[str, str] = {
    "EU AI Act": "32024R1689",
    "MDR 2017/745": "32017R0745",
    "IVDR 2017/746": "32017R0746",
}

# Regulation name patterns for detecting which regulations a question targets.
_REG_PATTERNS: dict[str, list[str]] = {
    "EU AI Act": ["ai act", "2024/1689", "eu ai"],
    "MDR 2017/745": ["mdr", "2017/745", "medical device regulation"],
    "IVDR 2017/746": ["ivdr", "2017/746", "in vitro"],
}

# Regex for detecting explicit provision references in a question.
# Matches "Annex I", "Annex XIV", "Article 5", "Article 26a", "Recital 47".
_PROVISION_REF_RE = re.compile(
    r"\b(annex\s+[IVX]{1,5}"
    r"|article\s+\d{1,3}[a-z]?(?:\(\d+\))?"  # catches Article 26(3)
    r"|recital\s+\d{1,4})\b",
    re.IGNORECASE,
)

# Regex for inline provision pointers found inside retrieved provision text.
# Matches references like "Annex VII", "Article 17", "Annex IV", "Section 2",
# "Chapter III" that appear inside the body or children of retrieved provisions.
_INLINE_REF_RE = re.compile(
    r"\b(Annex(?:es)?\s+[IVX]{1,5}"
    r"|Article\s+\d{1,3}[a-z]?"
    r"|Recital\s+\d{1,4}"
    r"|Section\s+\d{1,3}"
    r"|Chapter\s+[IVX]{1,5})\b",
)


def _detect_mentioned_regulations(question: str) -> set[str]:
    """Return regulation names mentioned in the question."""
    q_lower = question.lower()
    found: set[str] = set()
    for reg_name, patterns in _REG_PATTERNS.items():
        if any(p in q_lower for p in patterns):
            found.add(reg_name)
    return found


def _extract_provision_refs(question: str) -> list[str]:
    """Extract and normalise explicit provision references from *question*.

    Returns references like ``['Annex I', 'Article 26']`` ready for direct
    lookup.  Roman numerals are uppercased; article/recital numbers are
    preserved as-is.

    Examples
    --------
    >>> _extract_provision_refs("What does Annex I of the EU AI Act contain?")
    ['Annex I']
    >>> _extract_provision_refs("What are the obligations under Article 26?")
    ['Article 26']
    """
    seen: set[str] = set()
    result: list[str] = []
    for m in _PROVISION_REF_RE.finditer(question):
        parts = m.group(0).strip().split(None, 1)  # ["annex", "i"] or ["article", "5"]
        if len(parts) == 2:
            normalized = parts[0].capitalize() + " " + parts[1].upper()
        else:
            normalized = parts[0].capitalize()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _hyde_query(question: str, client: Any) -> str:
    """Generate a short hypothetical regulatory excerpt for HyDE retrieval.

    Produces a brief passage that resembles what EU regulatory text would
    say in response to *question*.  Encoding this passage with the same
    ``passage:`` prefix used for stored provisions places it in the same
    embedding space, yielding much better cosine similarity scores than
    encoding the question directly (query-to-document mismatch).

    The generation is capped at 100 tokens to minimise latency.
    """
    resp = client.chat.complete(
        model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
        messages=[{
            "role": "user",
            "content": (
                "Write one short paragraph (50\u201380 words) of dense EU regulatory "
                "text that directly answers the question below. "
                "Use precise legal terminology. Output only the regulatory text, "
                "no headings, no citations, no explanations.\n\n"
                f"Question: {question}"
            ),
        }],
        temperature=0.0,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


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
        # Word-boundary match to avoid spurious substring hits
        if re.search(r"\b" + re.escape(term_lower) + r"\b", q_lower):
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


_MAX_POINTER_REFS = 5


def _normalize_ref(raw: str) -> str:
    """Normalise a raw inline reference match to canonical form."""
    ref = raw.strip().replace("\xa0", " ")
    if ref.lower().startswith("annexes"):
        ref = "Annex" + ref[7:]
    return ref


def _extract_inline_refs(provisions: list[dict]) -> list[str]:
    """Scan retrieved provisions for inline references to other provisions.

    Returns normalised references (e.g. ``['Annex VII', 'Article 17']``)
    that are mentioned in the body or children of the already-retrieved
    provisions but are not themselves among those provisions.

    Capped at ``_MAX_POINTER_REFS`` to prevent context flooding.
    """
    already = {p.get("article_ref", "") for p in provisions}
    found: dict[str, None] = {}  # preserves insertion order, deduplicates

    for p in provisions:
        # Scan the parent body text
        body = (p.get("article_text", "") or "").replace("\xa0", " ")
        for m in _INLINE_REF_RE.finditer(body):
            ref = _normalize_ref(m.group(0))
            if ref not in already and ref not in found:
                found[ref] = None
                if len(found) >= _MAX_POINTER_REFS:
                    return list(found)

        # Scan children text (paragraphs / points)
        for c in p.get("children") or []:
            text = (c.get("raw_text") or c.get("text") or "").replace("\xa0", " ")
            for m in _INLINE_REF_RE.finditer(text):
                ref = _normalize_ref(m.group(0))
                if ref not in already and ref not in found:
                    found[ref] = None
                    if len(found) >= _MAX_POINTER_REFS:
                        return list(found)

    return list(found)


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
            limit = 1200 if is_match else 600
            text = (c.get("raw_text") or c.get("text") or "")
            if len(text) > limit:
                cut = text[:limit]
                last_period = cut.rfind('.')
                if last_period > limit // 2:
                    text = cut[:last_period + 1]
                else:
                    text = cut
            if text:
                if is_match:
                    matched_lines.append(f"  [\u2605 MATCHED] {ref}: {text}")
                else:
                    child_lines.append(f"  {ref}: {text}")
        child_lines = (matched_lines + child_lines)[:40]

        # Cross-referenced provisions (separate internal vs cross-regulation)
        cited = p.get("cited_provisions") or []
        cross_reg = p.get("cross_reg_cited") or []
        cross_reg_ids = {c.get("id") for c in cross_reg}
        cite_lines: list[str] = []
        for c in cited:
            ref = c.get("ref", "")
            is_xreg = c.get("id") in cross_reg_ids
            # Give more text budget to cross-regulation citations
            limit = 500 if is_xreg else 300
            text = (c.get("text") or "")[:limit]
            if text:
                tag = " [CROSS-REG]" if is_xreg else ""
                cite_lines.append(f"  -> {ref}{tag}: {text}")

        section = header + "\n" + body
        if p.get("_cross_reg_expansion"):
            section = f"[{section.lstrip('[')}  [via cross-regulation link]"
        if p.get("_pointer_expansion"):
            section = f"[{section.lstrip('[')}  [referenced in retrieved provisions]"
        if child_lines:
            section += "\n\nParagraphs/Points:\n" + "\n".join(child_lines)
        if cite_lines:
            section += "\n\nCross-references:\n" + "\n".join(cite_lines)
        parts.append(section)

    return "\n\n---\n\n".join(parts)


def ask(question: str, retriever, k: int = 20) -> str:
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

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # --- 1. Fetch legal definitions for terms mentioned in the question ---
    definitions = _detect_defined_terms(question, retriever)
    if definitions:
        logger.info(
            "Injecting %d definition(s): %s",
            len(definitions),
            ", ".join(d.get("term", "?") for d in definitions),
        )

    # --- 2. Regulation detection + CELEX filter ---
    mentioned_regs = _detect_mentioned_regulations(question)

    # Enrich: if definitions were found from regulations NOT already
    # mentioned, the question implicitly touches those regulations too.
    # e.g. "testing in real-world conditions" is an AI Act defined term
    # even if the user only says "MDR" explicitly.
    for d in definitions:
        reg = d.get("regulation", "")
        if reg and reg not in mentioned_regs and reg in _REG_NAME_TO_CELEX:
            mentioned_regs.add(reg)
            logger.info(
                "Implicit regulation detected via defined term '%s': %s",
                d.get("term", "?"), reg,
            )

    target_celexes: set[str] | None = None
    if len(mentioned_regs) >= 1:
        target_celexes = {
            _REG_NAME_TO_CELEX[r]
            for r in mentioned_regs
            if r in _REG_NAME_TO_CELEX
        }
        if len(mentioned_regs) > 1:
            # For multi-regulation questions, increase k proportionally so each
            # regulation gets adequate coverage (min 3 slots per regulation).
            k = max(k, len(mentioned_regs) * 3)

    # --- 3. Direct structural lookup for explicitly named provisions ---
    # e.g. "What does Annex I contain?" → exact display_ref lookup, no
    # vector similarity needed.
    explicit_refs = _extract_provision_refs(question)
    direct_provisions: list[dict] = []
    if explicit_refs:
        direct_provisions = retriever.retrieve_by_refs(
            explicit_refs, celex_filter=target_celexes,
        )
        # Cross-regulation fallback: if the question requests a specific
        # paragraph (e.g. "Article 2, paragraph 8") but the provision found
        # in the filtered regulation doesn't have that paragraph, widen
        # the search to all regulations.  This handles implicit cross-reg
        # questions where the user names a provision from a regulation they
        # didn't explicitly mention.
        if direct_provisions and target_celexes and len(target_celexes) < len(_REG_NAME_TO_CELEX):
            _para_m = re.search(
                r"paragraph\s+(\d+)", question, re.IGNORECASE,
            )
            if _para_m:
                wanted_para = _para_m.group(1)
                # Check if any direct-lookup result actually has the
                # requested paragraph among its children.
                has_para = False
                para_ref = f"Paragraph {wanted_para}"
                for dp in direct_provisions:
                    for c in dp.get("children") or []:
                        if c.get("ref") == para_ref:
                            has_para = True
                            break
                    if has_para:
                        break
                if not has_para:
                    wider = retriever.retrieve_by_refs(
                        explicit_refs, celex_filter=None,
                    )
                    new_ids = {p["article_id"] for p in direct_provisions}
                    for wp in wider:
                        if wp["article_id"] not in new_ids:
                            direct_provisions.append(wp)
                            new_ids.add(wp["article_id"])
                    logger.info(
                        "Cross-reg fallback: widened direct lookup for "
                        "paragraph %s → %d provision(s) total.",
                        wanted_para, len(direct_provisions),
                    )
        logger.info(
            "Direct lookup: %s → %d provision(s)",
            explicit_refs, len(direct_provisions),
        )

    # --- 4. HyDE: generate a hypothetical answer, encode as a passage ---
    # Encoding a passage-like hypothetical answer instead of the question
    # itself places the query vector in the same embedding space as the
    # stored provisions (document↔document rather than query↔document),
    # yielding much better cosine scores for semantically complex questions.
    hyde_text = _hyde_query(question, client)
    logger.debug("HyDE text: %s", hyde_text[:120])
    hyde_vec = retriever.encode_as_passage(hyde_text)

    provisions = retriever.retrieve(
        question, k=k, target_celexes=target_celexes, query_vec=hyde_vec,
    )

    # --- 5. Merge: inject direct-lookup results that vector search missed ---
    if direct_provisions:
        seen_ids: set[str] = {p["article_id"] for p in provisions}
        new_count = 0
        for p in direct_provisions:
            if p["article_id"] not in seen_ids:
                provisions.insert(0, p)  # direct matches rank first
                seen_ids.add(p["article_id"])
                new_count += 1
        if new_count:
            logger.info(
                "Direct-lookup merge: added %d provision(s) not found by "
                "vector search (total %d).",
                new_count, len(provisions),
            )

    if not provisions and not definitions:
        return (
            "No relevant provisions were found in the knowledge graph. "
            "Please check that embeddings have been generated "
            "(run scripts/embed_provisions.py)."
        )

    # --- 6. Pointer expansion: fetch provisions referenced inside context ---
    # Scan the retrieved text for inline references ("Annex VII",
    # "Article 17", etc.) and pull those provisions into the context so
    # the LLM has the full cross-referenced material.
    inline_refs = _extract_inline_refs(provisions)
    if inline_refs:
        pointer_provisions = retriever.retrieve_by_refs(
            inline_refs, celex_filter=target_celexes,
        )
        if pointer_provisions:
            seen_ids = {p["article_id"] for p in provisions}
            added = 0
            for p in pointer_provisions:
                if p["article_id"] not in seen_ids:
                    p["_pointer_expansion"] = True
                    provisions.append(p)
                    seen_ids.add(p["article_id"])
                    added += 1
                    if added >= _MAX_POINTER_REFS:
                        break
            if added:
                logger.info(
                    "Pointer expansion: %s → added %d provision(s) "
                    "(total %d).",
                    inline_refs[:5], added, len(provisions),
                )

    # --- 7. Assemble context: definitions first, then provisions ---
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

    response = client.chat.complete(
        model=os.environ.get("MISTRAL_MODEL", "mistral-large-latest"),
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
