"""Taxonomy and rules for classifying provisions by their legal function.

This module defines a **closed taxonomy** of legal roles a provision can play,
plus a small library of **high-precision regex rules** that deterministically
assign roles at canonicalization time.

The taxonomy is the structural backbone for downstream legal-reasoning
features (status-vs-obligation separation, role-aware retrieval, audit). It
exists to prevent a recurring failure mode in legal GraphRAG: conflating a
*definitional* provision (e.g. MDR Article 2(30)) with an
*obligation-modifying* provision (e.g. MDR Article 5(5)) because both happen
to be retrieved on the same query.

Each rule returns a ``(role, rule_id, confidence)`` triple, so every persisted
``provision_role`` carries auditable provenance. Provisions that do not match
any rule are tagged ``UNCLASSIFIED`` and may be classified later by LLM or
human reviewers — with provenance preserved in a separate property.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Closed taxonomy of legal roles
# ---------------------------------------------------------------------------
# NOTE: adding a new role requires (a) appending here, (b) a corresponding
# rule in this module, and (c) updating downstream consumers (agent context
# formatting, audit checks).

PROVISION_ROLE_TAXONOMY: tuple[str, ...] = (
    "DEFINES",          # Definitional provision ("'X' means Y")
    "EXEMPTS",          # Carve-out from obligations ("shall not apply")
    "EXTENDS_STATUS",   # Third-party becomes provider/manufacturer/etc.
    "OBLIGATION",       # Actor + modal "shall" (positive duty)
    "PROHIBITION",      # Explicit negative duty / banned activity
    "SCOPE",            # Defines what the regulation applies to
    "CLASSIFICATION",   # Risk-class / device-class assignment rules
    "PROCEDURAL",       # Describes a procedure (conformity assessment, etc.)
    "PENALTY",          # Sanctions / fines
    "INTERPRETIVE",     # Recital / citation / preamble (non-binding)
    "STRUCTURAL",       # Document / chapter / section container
    "UNCLASSIFIED",     # Fallback (LLM/human can fill in a later pass)
)

# ---------------------------------------------------------------------------
# Provenance source labels
# ---------------------------------------------------------------------------

PROVISION_ROLE_SOURCE_RULE = "rule"
PROVISION_ROLE_SOURCE_LLM = "llm"
PROVISION_ROLE_SOURCE_HUMAN = "human"

# ---------------------------------------------------------------------------
# Definitions articles per regulation
# ---------------------------------------------------------------------------
# Used by the DEFINES rule to catch sub-points inside a regulation's official
# definitions article that omit the leading quoted term. Extends the smaller
# map in ``domain/ontology/defined_terms.py`` with GDPR (Article 4).

DEFINITIONS_ARTICLE_IDS: dict[str, str] = {
    "32017R0745": "32017R0745_art_2",   # MDR Article 2
    "32017R0746": "32017R0746_art_2",   # IVDR Article 2
    "32024R1689": "32024R1689_art_3",   # AI Act Article 3
    "32016R0679": "32016R0679_art_4",   # GDPR Article 4
}

# ---------------------------------------------------------------------------
# Provision kinds that bypass text analysis
# ---------------------------------------------------------------------------

_STRUCTURAL_KINDS: frozenset[str] = frozenset({
    "document", "chapter", "section",
    "annex", "annex_chapter", "annex_section", "annex_subsection", "annex_part",
    "title", "enacting_terms", "final_provisions",
    "guidance_document", "guidance_section", "guidance_subsection",
})

_INTERPRETIVE_KINDS: frozenset[str] = frozenset({
    "recital", "citation", "preamble",
})

# ---------------------------------------------------------------------------
# Canonical actor subjects (used by OBLIGATION rule to require an actor + "shall")
# ---------------------------------------------------------------------------

# Keep in sync with ``actor_roles.CANONICAL_ACTOR_ROLES``: every canonical
# actor role must be matched by ``_ACTOR_SUBJECT_RE`` below. This list may also
# carry non-role obligation subjects (Member State, person, sponsor, …).
_ACTOR_SUBJECTS: tuple[str, ...] = (
    "provider", "providers",
    "manufacturer", "manufacturers",
    "product manufacturer", "product manufacturers",
    "importer", "importers",
    "distributor", "distributors",
    "deployer", "deployers",
    "controller", "controllers",
    "processor", "processors",
    "operator", "operators",
    "Member State", "Member States",
    "notified body", "notified bodies",
    "competent authority", "competent authorities",
    "national authority", "national authorities",
    "supervisory authority", "supervisory authorities",
    "market surveillance authority", "market surveillance authorities",
    "authorised representative", "authorised representatives",
    "authorized representative", "authorized representatives",
    "health institution", "health institutions",
    "user", "users",
    "data subject", "data subjects",
    "person", "persons",
    "economic operator", "economic operators",
    "responsible person", "responsible persons",
    "sponsor", "sponsors",
)

_ACTOR_SUBJECT_RE = re.compile(
    r"\b(?:" + "|".join(
        re.escape(a) for a in sorted(_ACTOR_SUBJECTS, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Compiled regex patterns for individual rules
# ---------------------------------------------------------------------------

# Tolerate leading numbering ("1. ", "(1) ") before the quoted term.
_DEFINES_TERM_MEANS_RE = re.compile(
    r"^\s*(?:\(\d+\)\s*|\d+[.)]\s*)?"
    r"['\u2018\u2019\u201a\u201b\u201c\u201d\"]"
    r"([^'\u2018\u2019\u201a\u201b\u201c\u201d\"]+)"
    r"['\u2018\u2019\u201a\u201b\u201c\u201d\"]\s+means\b",
    re.IGNORECASE,
)
_MEANS_ANYWHERE_RE = re.compile(r"\bmeans\b", re.IGNORECASE)

_PENALTY_TITLE_RE = re.compile(r"^\s*(penalt|sanction|fine)", re.IGNORECASE)
_PENALTY_TEXT_RE = re.compile(
    r"\b(?:administrative fines|administrative penalties|shall be punishable"
    r"|shall lay down (?:the )?(?:rules on )?penalties)\b",
    re.IGNORECASE,
)

_SCOPE_TITLE_RE = re.compile(r"^\s*scope\s*$", re.IGNORECASE)
_SCOPE_OPENING_RE = re.compile(
    r"^\s*(?:\(\d+\)\s*|\d+[.)]\s*)?"
    r"(?:this regulation|this directive)\s+"
    r"(?:applies to|does not apply|shall apply)",
    re.IGNORECASE,
)

_EXEMPTS_TITLE_RE = re.compile(r"\b(exemption|derogation|exception)s?\b", re.IGNORECASE)
_EXEMPTS_SHALL_NOT_APPLY_RE = re.compile(r"\bshall not apply\b", re.IGNORECASE)
_EXEMPTS_SCOPE_NEAR_RE = re.compile(
    r"\b(regulation|article|paragraph|chapter|section|provisions|requirements|obligations)\b",
    re.IGNORECASE,
)
_EXEMPTS_EXCEPTION_OF_RE = re.compile(r"\bwith the exception of\b", re.IGNORECASE)

_EXTENDS_STATUS_RE = re.compile(
    r"\bshall be (?:considered|deemed)\s+(?:to be\s+)?(?:a\s+|an\s+)?"
    r"(?:provider|manufacturer|importer|distributor|deployer|controller|processor|responsible person|legal manufacturer)\b",
    re.IGNORECASE,
)

_PROHIBITION_RE = re.compile(
    r"\b(?:is|are|shall be) prohibited\b"
    r"|\bshall not\s+(?:be\s+)?"
    r"(?:place|put|make available|allow|permit|introduce|offer|supply|use)",
    re.IGNORECASE,
)

_CLASSIFICATION_ASSIGN_RE = re.compile(
    r"\b(?:is|are|shall be) classified (?:as|in)\b",
    re.IGNORECASE,
)
_CLASSIFICATION_RULE_TITLE_RE = re.compile(r"^\s*Rule\s+\d+\b", re.IGNORECASE)

# Article 6(1)/(2) AI Act: "<subject> shall be considered to be high-risk".
# Article 6(3) ("shall NOT be considered to be high-risk") is naturally
# excluded because the negation sits between "shall" and "be", breaking the
# adjacency this pattern requires.
_CLASSIFICATION_DEEMED_HIGH_RISK_RE = re.compile(
    r"\bshall be (?:considered|deemed)\s+(?:to be\s+)?(?:a\s+|an\s+)?"
    r"high(?:-|\s)risk\b",
    re.IGNORECASE,
)

# Article 51(2) AI Act: "shall be presumed to have high impact capabilities"
# (the FLOPs presumption that triggers systemic-risk classification of a
# general-purpose AI model).
_CLASSIFICATION_PRESUMED_CAPABILITY_RE = re.compile(
    r"\b(?:shall be|is) presumed to have high(?:-|\s)impact capabilities\b",
    re.IGNORECASE,
)

_PROCEDURAL_TITLE_RE = re.compile(
    r"\b(procedure|procedures|conformity assessment)\b",
    re.IGNORECASE,
)
# NOTE: no phrase-based PROCEDURAL rule. Texts like "the provider shall
# undergo the conformity assessment procedure..." are OBLIGATIONS that
# happen to mention a procedure; the binding force is the duty, not the
# procedure name. PROCEDURAL is reserved for provisions whose *title*
# announces them as procedure descriptions (e.g. AI Act Article 43).

_SHALL_RE = re.compile(r"\bshall\b", re.IGNORECASE)
_SHALL_NOT_APPLY_LOOKAHEAD = re.compile(r"\s*not\s+apply\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleAssignment:
    """Result of classifying a single provision.

    ``confidence`` is 1.0 for deterministic rule matches with no inferential
    step, and lower (0.85-0.95) for rules that infer role from context. The
    LLM and human paths will use their own confidence scales when introduced.
    """
    role: str
    rule_id: str
    confidence: float


# ---------------------------------------------------------------------------
# Helpers for definitions-article membership (works on provision id alone)
# ---------------------------------------------------------------------------

def _provision_article_number(provision_id: str, celex: str) -> str | None:
    """Extract the article number from a provision id, if it has one.

    Handles both id formats used by the parser:
    - article-level / point-under-article:  ``{celex}_art_{N}[_pt_...]``
    - paragraph-level / point-under-para:   ``{celex}_{NNN}.{PPP}[_pt_...]``
    """
    if not provision_id.startswith(f"{celex}_"):
        return None
    rest = provision_id[len(celex) + 1:]

    m = re.match(r"^art_(\d+)", rest)
    if m:
        return m.group(1)

    m = re.match(r"^(\d{3})\.\d{3}", rest)
    if m:
        return str(int(m.group(1)))  # strip zero padding

    return None


def _is_in_definitions_article(provision_id: str, celex: str) -> bool:
    """True if the provision lives inside the regulation's definitions article."""
    def_id = DEFINITIONS_ARTICLE_IDS.get(celex)
    if not def_id:
        return False
    m = re.match(rf"^{re.escape(celex)}_art_(\d+)$", def_id)
    if not m:
        return False
    return _provision_article_number(provision_id, celex) == m.group(1)


# ---------------------------------------------------------------------------
# Individual rule predicates
# ---------------------------------------------------------------------------
# Each rule takes the full context kwargs and returns a RoleAssignment or None.
# Rules MUST be ordered from most specific to least specific in ``_RULES``.

def _rule_defines(text: str, title: str | None, kind: str,
                  provision_id: str, celex: str) -> RoleAssignment | None:
    if _DEFINES_TERM_MEANS_RE.search(text):
        return RoleAssignment("DEFINES", "defines.quoted_means.v1", 1.0)
    if _is_in_definitions_article(provision_id, celex) and _MEANS_ANYWHERE_RE.search(text):
        return RoleAssignment("DEFINES", "defines.definitions_article.v1", 0.95)
    return None


def _rule_penalty(text: str, title: str | None, **_) -> RoleAssignment | None:
    if title and _PENALTY_TITLE_RE.search(title):
        return RoleAssignment("PENALTY", "penalty.title.v1", 1.0)
    if _PENALTY_TEXT_RE.search(text):
        return RoleAssignment("PENALTY", "penalty.keyword.v1", 0.95)
    return None


def _rule_scope(text: str, title: str | None, **_) -> RoleAssignment | None:
    if title and _SCOPE_TITLE_RE.match(title):
        return RoleAssignment("SCOPE", "scope.title.v1", 1.0)
    if _SCOPE_OPENING_RE.match(text):
        return RoleAssignment("SCOPE", "scope.opening_phrase.v1", 1.0)
    return None


def _rule_exempts(text: str, title: str | None, **_) -> RoleAssignment | None:
    if title and _EXEMPTS_TITLE_RE.search(title):
        return RoleAssignment("EXEMPTS", "exempts.title.v1", 1.0)
    m = _EXEMPTS_SHALL_NOT_APPLY_RE.search(text)
    if m:
        start = max(m.start() - 120, 0)
        end = min(m.end() + 120, len(text))
        if _EXEMPTS_SCOPE_NEAR_RE.search(text[start:end]):
            return RoleAssignment("EXEMPTS", "exempts.shall_not_apply.v1", 0.95)
    if _EXEMPTS_EXCEPTION_OF_RE.search(text):
        return RoleAssignment("EXEMPTS", "exempts.exception_of.v1", 0.9)
    return None


def _rule_extends_status(text: str, **_) -> RoleAssignment | None:
    if _EXTENDS_STATUS_RE.search(text):
        return RoleAssignment("EXTENDS_STATUS", "extends_status.deemed_actor.v1", 1.0)
    return None


def _rule_prohibition(text: str, **_) -> RoleAssignment | None:
    if _PROHIBITION_RE.search(text):
        return RoleAssignment("PROHIBITION", "prohibition.explicit.v1", 0.95)
    return None


def _rule_classification(text: str, title: str | None, **_) -> RoleAssignment | None:
    if title and _CLASSIFICATION_RULE_TITLE_RE.match(title):
        return RoleAssignment("CLASSIFICATION", "classification.rule_title.v1", 1.0)
    if _CLASSIFICATION_ASSIGN_RE.search(text):
        return RoleAssignment("CLASSIFICATION", "classification.class_assignment.v1", 0.9)
    if _CLASSIFICATION_DEEMED_HIGH_RISK_RE.search(text):
        return RoleAssignment("CLASSIFICATION", "classification.deemed_high_risk.v1", 1.0)
    if _CLASSIFICATION_PRESUMED_CAPABILITY_RE.search(text):
        return RoleAssignment("CLASSIFICATION", "classification.presumed_capability.v1", 1.0)
    return None


def _rule_procedural(text: str, title: str | None, **_) -> RoleAssignment | None:
    if title and _PROCEDURAL_TITLE_RE.search(title):
        return RoleAssignment("PROCEDURAL", "procedural.title.v1", 0.95)
    return None


def _rule_obligation(text: str, **_) -> RoleAssignment | None:
    """OBLIGATION = at least one 'shall' (not part of 'shall not apply') with
    an actor subject within an 80-character window around it."""
    for m in _SHALL_RE.finditer(text):
        # Skip "shall not apply" — that pattern is owned by EXEMPTS (which
        # runs first, but defensively skip here too in case EXEMPTS missed it
        # because its scope-word window did not hit).
        if _SHALL_NOT_APPLY_LOOKAHEAD.match(text, m.end()):
            continue
        start = max(m.start() - 80, 0)
        end = min(m.end() + 80, len(text))
        if _ACTOR_SUBJECT_RE.search(text[start:end]):
            return RoleAssignment("OBLIGATION", "obligation.actor_shall.v1", 0.9)
    return None


# ---------------------------------------------------------------------------
# Rule pipeline (order matters: most specific first)
# ---------------------------------------------------------------------------

_RULES: tuple[Callable, ...] = (
    _rule_defines,         # 1. quoted-term-means is highly specific
    _rule_penalty,         # 2. title-anchored, distinct semantics
    _rule_scope,           # 3. title-anchored, distinct opening phrase
    _rule_exempts,         # 4. MUST precede OBLIGATION ("shall not apply")
    _rule_extends_status,  # 5. MUST precede OBLIGATION ("shall be considered")
    _rule_prohibition,     # 6. MUST precede OBLIGATION ("shall not <verb>")
    _rule_classification,  # 7. distinct semantics
    _rule_procedural,      # 8. distinct semantics
    _rule_obligation,      # 9. catch-all for actor + shall
)


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify_provision(
    *,
    text: str,
    kind: str,
    title: str | None = None,
    provision_id: str = "",
    celex: str = "",
) -> RoleAssignment:
    """Assign a single primary legal role to a provision.

    Rule pipeline runs in priority order; **first match wins**. Provisions
    with no matching rule return ``UNCLASSIFIED``, leaving room for an LLM or
    human classification pass to fill the gap (with separate provenance).
    """
    if kind in _STRUCTURAL_KINDS:
        return RoleAssignment("STRUCTURAL", "structural.container_kind.v1", 1.0)
    if kind in _INTERPRETIVE_KINDS:
        return RoleAssignment("INTERPRETIVE", "interpretive.recital_kind.v1", 1.0)

    text = (text or "").strip()
    if not text:
        return RoleAssignment("UNCLASSIFIED", "none.empty_text.v1", 1.0)

    title_norm = (title or "").strip() or None

    for rule in _RULES:
        result = rule(
            text=text, title=title_norm, kind=kind,
            provision_id=provision_id, celex=celex,
        )
        if result is not None:
            return result

    return RoleAssignment("UNCLASSIFIED", "none.no_rule_matched.v1", 1.0)
