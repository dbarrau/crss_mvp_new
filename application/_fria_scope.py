"""FRIA-scope guard — flag an over-broad fundamental-rights-impact-assessment claim.

Article 27(1) of the AI Act does **not** oblige every deployer of a high-risk AI
system to perform a fundamental-rights impact assessment (FRIA). The duty is
scoped to three deployer classes only:

  * bodies governed by public law;
  * private entities providing public services;
  * deployers of the high-risk systems in Annex III, points 5(b) and (c)
    (creditworthiness/credit scoring, and life-and-health insurance risk
    assessment/pricing).

(grounded verbatim against AI Act Article 27(1) in the consolidated graph.)

The model recurrently drops that scope caveat and tells a *plain* deployer it
must run a FRIA (measured: HQ_004, HQ_008 over-apply Article 27). This is not a
fabricated quote or a nonexistent citation, so the faithfulness / phantom /
superseded guards never see it — the text is real, the *scope* of the claim is
wrong. Rewriting a legal conclusion is unsafe (see the "surface not mechanize"
lessons), so this guard does not touch the answer body: it appends a loud SCOPE
caveat whenever the answer asserts a FRIA obligation with no scope qualifier
anywhere, and stays silent otherwise.

High precision by construction — it fires only when BOTH hold:
  1. some line asserts the FRIA duty unconditionally (FRIA term + an obligation
     verb, and that line is not itself hedged/negated/conditional); and
  2. NONE of the scope-defining terms (public law/body/authority, public
     services, creditworthiness/credit scoring, life-and-health insurance,
     Annex III point 5(b)/(c), or an explicit "not all deployers" hedge) appears
     anywhere in the answer.

Missing a real defect (a false negative) is the accepted failure mode; a false
flag on a correctly-scoped answer is not — any answer that states the scope
correctly names one of the suppressor terms and is left untouched.
Disable with CRSS_FRIA_GUARD=0.
"""
from __future__ import annotations

import re

# The AI-Act-specific FRIA term anchors the guard (no CELEX resolution needed —
# "fundamental rights impact assessment" / "FRIA" is unambiguous AI Act).
_FRIA_TERM = re.compile(
    r"\bfundamental[\s\-]rights?\s+impact\s+assessment\b|\bFRIA\b", re.I
)

# An APPLICABILITY assertion: the deployer must *perform* the FRIA, or the FRIA
# is required/mandatory/applies. Restricted to performance/applicability verbs —
# a *management* duty about the FRIA ("the FRIA must be documented / notified /
# must include …") is not a scope claim and must not trigger (measured: HQ_001
# false-positive on "The FRIA must be documented").
_APPLIES = re.compile(
    r"\b(?:conduct(?:s|ed|ing)?|perform(?:s|ed|ing)?|carry\s+out|carrying\s+out|"
    r"carried\s+out|undertake[ns]?|undertaking|is\s+required|are\s+required|"
    r"required\s+to|is\s+mandatory|are\s+mandatory|mandatory|applies|apply)\b",
    re.I,
)

# A hedge/negation/conditional on the SAME line means the sentence is not an
# unconditional over-application (e.g. "a FRIA is required only where…",
# "no FRIA is needed unless…"), so it does not trigger the flag.
_HEDGED_LINE = re.compile(
    r"\bno[nt]?\b|\bunless\b|\bonly\b|\bexcept\b|\bwhere\s+the\s+deployer\b"
    r"|\bif\s+(?:the|you|your|a\b|an\b)\b|\bwhether\b"
    # conditional-applicability qualifiers: "conduct a FRIA if required",
    # "FRIA (if applicable)" — the duty is expressly conditioned, not asserted.
    r"|\b(?:if|where|when|as)\s+(?:required|applicable|appropriate|relevant)\b",
    re.I,
)

# The scope caveat, checked in the NEIGHBOURHOOD of the FRIA claim (±_SCOPE_WINDOW
# lines), not answer-wide: an unrelated "public authorities" heading about a
# different duty (registration) elsewhere in a long multi-obligation answer must
# not suppress the flag (measured: HQ_008 wrongly suppressed answer-wide). Point 5
# is tight to 5(b)/(c) — a "point 5(d)" mention is a different use case, not the
# FRIA scope (measured: HQ_004 wrongly suppressed by a loose "point 5").
_SCOPE_NEARBY = re.compile(
    r"public\s+law"
    r"|public\s+bod(?:y|ies)"
    r"|public\s+authorit"
    r"|public[\s\-]sector"
    r"|public\s+services?"
    r"|private\s+entit(?:y|ies)\s+providing"
    r"|creditworthi"
    r"|credit[\s\-]scor"
    r"|life\s+and\s+health\s+insurance"
    r"|health\s+insurance"
    r"|life\s+insurance"
    r"|points?\s*\(?5\)?\s*\([bc]\)",
    re.I,
)
_SCOPE_WINDOW = 2

# An explicit scope hedge anywhere in the answer IS a correct scope statement and
# suppresses answer-wide (it cannot be misread as an unrelated mention).
_SCOPE_GLOBAL = re.compile(
    r"not\s+all\s+deployers"
    r"|only\s+(?:certain|some|specific)\s+(?:deployers|categories)"
    r"|does\s+not\s+apply\s+to\s+(?:all|every)",
    re.I,
)

_FRIA_SCOPE_CAVEAT = (
    "> ⚠ **FRIA SCOPE** — Article 27's fundamental-rights impact "
    "assessment does **not** apply to every deployer of a high-risk AI system. "
    "Under **Article 27(1)** it is required only of deployers that are **bodies "
    "governed by public law** or **private entities providing public services**, "
    "and deployers of the high-risk systems in **Annex III, points 5(b) and (c)** "
    "(creditworthiness/credit scoring, and life-and-health insurance risk "
    "assessment/pricing). Confirm the deployer falls in one of these categories "
    "before treating a FRIA as mandatory."
)


def _scannable(line: str) -> bool:
    """Skip headings and blockquotes (an existing flag/footer may quote 'FRIA')."""
    s = line.lstrip()
    return not (s.startswith("#") or s.startswith(">"))


def flag_fria_overapplication(answer: str) -> tuple[str, list[str]]:
    """Append a FRIA-scope caveat if the answer over-applies Article 27.

    Returns ``(answer, notes)`` — ``notes`` is non-empty (one entry) only when
    the caveat was appended, for logging parity with the other guards.
    """
    if not answer or not _FRIA_TERM.search(answer):
        return answer, []

    # Idempotent: never append the caveat twice (proximity-scoped suppression
    # would not catch the end-appended block).
    if "**FRIA SCOPE**" in answer:
        return answer, []

    # An explicit "not all deployers" scope hedge anywhere → correctly scoped.
    if _SCOPE_GLOBAL.search(answer):
        return answer, []

    lines = answer.splitlines()
    # Fire when some line asserts the FRIA duty unconditionally (FRIA term + a
    # performance/applicability verb, not hedged) AND no scope term sits within
    # ±_SCOPE_WINDOW lines of it.
    for i, line in enumerate(lines):
        if not _scannable(line):
            continue
        if not (_FRIA_TERM.search(line) and _APPLIES.search(line)):
            continue
        if _HEDGED_LINE.search(line):
            continue
        window = "\n".join(lines[max(0, i - _SCOPE_WINDOW): i + _SCOPE_WINDOW + 1])
        if _SCOPE_NEARBY.search(window):
            continue  # the claim is scoped in its own neighbourhood
        answer = answer.rstrip() + "\n\n" + _FRIA_SCOPE_CAVEAT
        return answer, ["Article 27 FRIA asserted without its 27(1) deployer-scope caveat"]

    return answer, []
