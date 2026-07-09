"""Second-pass answer auditing — bounded-agentic verify-and-revise loop.

After the draft answer is generated, an *Auditor* LLM call inspects it against
the legal backbone (initial actor status -> primary legal route -> decisive
authority -> calibration) and emits a structured verdict. When it finds gaps it
*names the provisions and topics it needs*; those drive targeted re-retrieval
from the graph, and an *Adjudicator* pass regenerates the answer over the
gap-filled context.

The expensive regeneration is *gated* (CRAG-style): it fires only when the audit
finds a broken legal backbone — a wrong initial actor status or a wrong primary
legal route (the conditions that cap the rubric score). Minor wording/coverage
"issues" alone do NOT trigger a rewrite, so the common case stays single-pass
(draft + one cheap audit). The loop is bounded (default 1 iteration) and the
outer control flow is deterministic — the model is agentic only in deciding
*what to retrieve*, never in deciding the overall trajectory.

The audit call (structured JSON) can run on a smaller/faster model than the
generation call via ``CRSS_AUDIT_MODEL``; the revision regeneration always uses
the main generation model.

Provision references suggested by the Auditor are retrieval *requests*, not
asserted facts: they are looked up against Neo4j, so a hallucinated reference
simply returns nothing and cannot leak into the answer.

Disable with ``CRSS_AUDIT=0``. Tune with ``CRSS_AUDIT_MAX_ITERS`` and
``CRSS_AUDIT_MAX_GAP_REFS``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Routes whose answers involve multi-step legal reasoning and therefore benefit
# from a backbone audit. Simple lookups (definition/provision) are skipped to
# avoid needless latency and cost.
_AUDIT_ROUTES: frozenset[str] = frozenset({
    "legal_qualification",
    "cross_regulation",
    "classification_chain",
    "role_obligations",
})

_DEFAULT_MAX_ITERS = 1
_DEFAULT_MAX_GAP_REFS = 6

# Truncation budgets for the audit prompt (chars), to bound token usage.
_CONTEXT_BUDGET = 9000
_ANSWER_BUDGET = 5000


def _max_iters() -> int:
    try:
        return max(1, int(os.environ.get("CRSS_AUDIT_MAX_ITERS", _DEFAULT_MAX_ITERS)))
    except ValueError:
        return _DEFAULT_MAX_ITERS


def _max_gap_refs() -> int:
    try:
        return max(1, int(os.environ.get("CRSS_AUDIT_MAX_GAP_REFS", _DEFAULT_MAX_GAP_REFS)))
    except ValueError:
        return _DEFAULT_MAX_GAP_REFS


def _audit_model() -> str:
    """Model for the (cheap, structured) audit call.

    The audit returns a structured true/false verdict, a task a mid-tier model
    handles reliably, so it defaults to ``mistral-medium-latest`` — roughly half
    the latency of the large generation model with negligible loss of guardrail
    precision.  Override with ``CRSS_AUDIT_MODEL`` (e.g. ``mistral-large-latest``
    for maximum precision, or ``mistral-small-latest`` for maximum speed).
    """
    return os.environ.get("CRSS_AUDIT_MODEL", "mistral-medium-latest")


def _needs_revision(findings: dict[str, Any]) -> bool:
    """Gate the expensive regeneration on a genuinely broken legal backbone.

    Returns True only when the initial actor status or the primary legal route
    is wrong — the defects that cap the rubric score. Minor "issues" alone are
    not worth a full rewrite, so most answers stay single-pass.
    """
    if findings.get("_parse_failed"):
        return False
    return not findings["initial_status_correct"] or not findings["primary_route_correct"]


def _should_audit(route_id: str) -> bool:
    """Whether the audit loop should run for this route."""
    if os.environ.get("CRSS_AUDIT", "1") == "0":
        return False
    return route_id in _AUDIT_ROUTES


_AUDIT_SYSTEM = (
    "You are a senior EU regulatory compliance auditor (MDR 2017/745, IVDR "
    "2017/746, EU AI Act 2024/1689, GDPR 2016/679). You review a DRAFT answer "
    "for legal reliability and return a structured verdict. You are strict: you "
    "reward correct legal architecture over eloquence."
)

_AUDIT_INSTRUCTIONS = """\
Audit the DRAFT ANSWER against this legal backbone, in order:

1. INITIAL ACTOR STATUS — is each actor's *starting* legal status correct?
   - An entity that develops AND puts an AI system into service under its own
     name is a PROVIDER from inception (AI Act Art 3(3)) — NOT a deployer.
   - A health institution that manufactures a device IS a manufacturer under
     MDR; Article 5(5) is an EXEMPTION from requirements, it does not make the
     institution a non-manufacturer.
2. PRIMARY LEGAL ROUTE — does the draft rest on the correct *decisive* trigger,
   not a merely arguable one? (e.g. distinguish "placing on the market" /
   "putting into service" from mere enabling of use.)
3. DECISIVE AUTHORITY — are any decisive provisions, definitions, or annexes
   missing or under-used?
4. CALIBRATION — does it assert certainty where the law is genuinely ambiguous,
   or hide behind vagueness where it should commit?

Return ONLY a JSON object (no prose, no markdown fences) with EXACTLY this shape:
{
  "initial_status_correct": true or false,
  "primary_route_correct": true or false,
  "issues": ["concise description of each MATERIAL legal defect"],
  "missing_provision_refs": ["Article 2", "Annex VIII"],
  "missing_topics": ["short semantic search query for a gap not addressable by a single ref"],
  "verdict": "PASS" or "REVISE"
}

Rules:
- Default initial_status_correct and primary_route_correct to TRUE. Set either
  to FALSE ONLY for a concrete, material error — never for stylistic, emphasis,
  or completeness preferences:
  * initial_status_correct = FALSE only if the draft assigns a materially WRONG
    starting status — e.g. calls a self-developing operator a "deployer", or
    states that an in-house manufacturer is "not a manufacturer". A draft that
    names the correct status is TRUE even if it could be elaborated.
  * primary_route_correct = FALSE only if the draft rests on the WRONG decisive
    legal trigger — NOT merely because a different framing, interaction, or
    additional provision could be added.
- "verdict" is "REVISE" if and ONLY if initial_status_correct is false OR
  primary_route_correct is false. Otherwise "PASS" — record any lesser concerns
  under "issues" without forcing a revision.
- "missing_provision_refs" are retrieval REQUESTS — name provisions whose text
  would close a backbone gap. Use canonical forms like "Article 6", "Annex III".
  Do not invent obscure sub-references you are unsure exist.
- Keep "issues" focused on backbone defects, not wording preferences.
"""


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[:budget] + "\n…[truncated]"


def _parse_findings(raw: str) -> dict[str, Any]:
    """Extract the JSON verdict from the auditor response, defensively.

    On any parse failure returns a PASS verdict so the loop terminates rather
    than revising blindly on garbage.
    """
    fallback = {
        "initial_status_correct": True,
        "primary_route_correct": True,
        "issues": [],
        "missing_provision_refs": [],
        "missing_topics": [],
        "verdict": "PASS",
        "_parse_failed": True,
    }
    if not raw:
        return fallback
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return fallback
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return fallback

    def _as_str_list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    verdict = str(data.get("verdict", "PASS")).upper()
    verdict = "REVISE" if "REVISE" in verdict else "PASS"
    return {
        "initial_status_correct": bool(data.get("initial_status_correct", True)),
        "primary_route_correct": bool(data.get("primary_route_correct", True)),
        "issues": _as_str_list(data.get("issues")),
        "missing_provision_refs": _as_str_list(data.get("missing_provision_refs")),
        "missing_topics": _as_str_list(data.get("missing_topics")),
        "verdict": verdict,
        "_parse_failed": False,
    }


def _audit_answer(
    question: str,
    context: str,
    answer: str,
    client: Any,
    *,
    model: str,
) -> dict[str, Any]:
    """Run one Auditor LLM call; return parsed findings."""
    user = (
        f"{_AUDIT_INSTRUCTIONS}\n\n"
        f"## QUESTION\n{question}\n\n"
        f"## RETRIEVED PROVISIONS (what the draft was grounded in)\n"
        f"{_truncate(context, _CONTEXT_BUDGET)}\n\n"
        f"## DRAFT ANSWER\n{_truncate(answer, _ANSWER_BUDGET)}\n"
    )
    resp = client.chat.complete(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _AUDIT_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    findings = _parse_findings(raw)
    logger.info(
        "Audit verdict=%s status_ok=%s route_ok=%s issues=%d refs=%s topics=%s",
        findings["verdict"], findings["initial_status_correct"],
        findings["primary_route_correct"], len(findings["issues"]),
        findings["missing_provision_refs"], findings["missing_topics"],
    )
    return findings


def _gap_retrieve(
    findings: dict[str, Any],
    retriever: Any,
    *,
    target_celexes: set[str] | None,
    existing_ids: set[str],
    max_add: int,
) -> list[dict[str, Any]]:
    """Fetch provisions the auditor asked for, deduped against existing_ids."""
    new: list[dict[str, Any]] = []
    seen = set(existing_ids)

    def _add(provs: list[dict[str, Any]]) -> None:
        for p in provs:
            aid = p.get("article_id")
            if not aid or aid in seen:
                continue
            seen.add(aid)
            new.append(p)
            if len(new) >= max_add:
                return

    refs = findings.get("missing_provision_refs") or []
    if refs and len(new) < max_add:
        try:
            _add(retriever.retrieve_by_refs(refs, celex_filter=target_celexes))
        except Exception as exc:  # noqa: BLE001 — retrieval is best-effort
            logger.warning("Audit gap ref-retrieval failed: %s", exc)

    for topic in findings.get("missing_topics") or []:
        if len(new) >= max_add:
            break
        try:
            _add(retriever.retrieve(topic, k=3, target_celexes=target_celexes))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audit gap topic-retrieval failed for %r: %s", topic, exc)

    return new


def _format_findings(findings: dict[str, Any]) -> str:
    """Short human-readable summary for a step event."""
    if findings.get("_parse_failed"):
        return "auditor response unparseable; treated as PASS"
    parts = [
        f"status_ok={findings['initial_status_correct']}",
        f"route_ok={findings['primary_route_correct']}",
    ]
    if findings["issues"]:
        parts.append(f"{len(findings['issues'])} issue(s): " + "; ".join(findings["issues"][:3]))
    if findings["missing_provision_refs"]:
        parts.append("wants: " + ", ".join(findings["missing_provision_refs"][:6]))
    return " | ".join(parts)


# Parenthetical clauses that reference the internal audit machinery; if the
# Adjudicator leaks one despite instructions, strip it conservatively (only
# whole parentheticals mentioning these meta-terms are removed).
_META_LEAK_RE = re.compile(
    r"\s*\([^)]*\b(?:revision\s+directive|internal\s+qa(?:\s+review)?|"
    r"audit\s+findings?|the\s+directive|as\s+(?:confirmed|noted|stated)\s+"
    r"in\s+the\s+(?:revision|review))\b[^)]*\)",
    re.I,
)


def _strip_meta_leak(answer: str) -> str:
    """Remove parenthetical references to the audit machinery from the answer."""
    return _META_LEAK_RE.sub("", answer)


def _build_revision_messages(
    question: str,
    context: str,
    findings: dict[str, Any],
    prior_answer: str,
    *,
    system_prompt: str,
    user_message: str,
) -> list[dict[str, str]]:
    """Assemble the Adjudicator messages: original discipline + audit findings.

    ``user_message`` is the standard per-request user message (already built by
    the caller via ``_build_user_message`` over the gap-filled context), so all
    original grounding/formatting rules carry over. We prepend the auditor's
    findings as a correction directive.
    """
    issues = "\n".join(f"  - {i}" for i in findings["issues"]) or "  - (none enumerated)"
    directive = (
        "INTERNAL QA REVIEW (not part of the answer) — the previous draft was "
        "checked for legal-backbone defects. Points to correct:\n"
        f"- initial actor status correct: {findings['initial_status_correct']}\n"
        f"- primary legal route correct: {findings['primary_route_correct']}\n"
        f"- material issues:\n{issues}\n\n"
        "Additional provisions retrieved to close these gaps have been appended "
        "to the REGULATORY CONTEXT below. Rewrite the answer as a single, "
        "standalone compliance analysis: keep what was correct, fix the flagged "
        "backbone, correct the initial actor status FIRST if it was wrong, and "
        "ground every legal fact in the context. Keep the revision TIGHT — fix "
        "the flagged backbone without inflating length; a senior reviewer prefers "
        "decisive analysis over exhaustive coverage.\n"
        "GROUNDING IS BINDING IN THIS REWRITE (do not relax it to 'fix' a gap): "
        "obey REFERENCES & QUOTATIONS exactly — every quotation is a `[quote: id]` "
        "pointer, NEVER text you type; references are bold prose. If closing the "
        "flagged backbone would require a provision that is NOT in the context "
        "below, state that the context is insufficient for that point — do NOT "
        "quote or reconstruct its wording from memory. A correct paraphrase with a "
        "bold reference fully grounds a point; do not add a quotation to force one.\n"
        "Do NOT reference this review, the audit, any 'directive' or 'findings', or "
        "the fact that the answer was revised — write as if producing the final "
        "answer directly.\n\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": directive + user_message},
    ]
