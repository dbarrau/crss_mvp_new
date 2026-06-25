"""Pre-retrieval scope assessment — ask-first clarification.

Every EU compliance answer is conditioned on *who is asking*: a provider/
manufacturer, a deployer, an importer and a distributor carry entirely
different obligations for the very same product. Actor role is therefore the
backbone of the answer — and the audit loop's first check — yet it is the fact
users are least likely to state, because they do not know it is the decisive
variable. The burden of knowing "what CRSS needs" belongs on CRSS, which owns
the ontology of what makes a compliance question well-posed.

This module detects, deterministically (no LLM), when an obligation-type
question omits a *decisive* slot whose candidate values would fork the legal
analysis, and asks for it before answering rather than silently assuming.

The slot framework is extensible; the only active slot in this increment is the
actor role — the single highest-impact axis. Risk tier and lifecycle trigger
are natural follow-ons.

Gated by ``CRSS_CLARIFY`` (default on) at the integration layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from application._routing import _has_obligation_focus
from application.contracts import Scenario
from domain.legislation_catalog import (
    AI_ACT_CELEX,
    MDR_CELEX,
    IVDR_CELEX,
    GDPR_CELEX,
)
from domain.ontology.actor_roles import CANONICAL_ACTOR_ROLES

# Roles ordered by how likely an asker is to self-identify as one; oversight
# bodies (notified body, supervisory authority) come last. Options are derived
# from ``CANONICAL_ACTOR_ROLES`` so they cannot drift from the role registry —
# this ordering only curates presentation.
_ROLE_PRIORITY: tuple[str, ...] = (
    "provider",
    "manufacturer",
    "deployer",
    "user",
    "importer",
    "distributor",
    "authorised representative",
    "product manufacturer",
    "operator",
    "controller",
    "processor",
    "notified body",
    "supervisory authority",
)

# Cap so the clarifying question stays scannable.
_MAX_ROLE_OPTIONS: int = 6

# Human-readable framework names for the role-partitioned regulations. MDCG
# guidance CELEXes are intentionally absent — they carry no actor roles.
_CELEX_SHORT: dict[str, str] = {
    GDPR_CELEX: "GDPR",
    MDR_CELEX: "MDR",
    IVDR_CELEX: "IVDR",
    AI_ACT_CELEX: "EU AI Act",
}

# Routes where a missing actor role is already handled, or where asking is
# inappropriate (a corpus-coverage overview is role-agnostic).
_NO_CLARIFY_ROUTES: frozenset[str] = frozenset(
    {"definition_lookup", "provision_lookup", "community_summary_search"}
)


@dataclass(frozen=True)
class ClarificationOption:
    """One candidate value the user can pick to fill a decisive slot."""

    label: str            # display text, e.g. "Provider"
    value: str            # normalized role term, e.g. "provider"
    celexes: frozenset[str]
    frameworks: str       # readable frameworks recognizing it, e.g. "EU AI Act"


@dataclass(frozen=True)
class Clarification:
    """A single ask-first clarifying question for one decisive slot."""

    slot: str
    question: str
    rationale: str
    options: list[ClarificationOption] = field(default_factory=list)


@dataclass(frozen=True)
class ScopingResult:
    """Outcome of pre-retrieval scope assessment."""

    needs_clarification: bool
    clarification: Clarification | None = None


def _readable_frameworks(celexes: set[str]) -> str:
    """Join in-scope framework short names into readable prose."""
    names = [_CELEX_SHORT[c] for c in sorted(celexes) if c in _CELEX_SHORT]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _role_options(target_celexes: set[str]) -> list[ClarificationOption]:
    """Build the candidate-role options recognized in the in-scope regulations.

    Derived from ``CANONICAL_ACTOR_ROLES`` (the role registry) filtered to the
    regulations actually in scope, ordered by ``_ROLE_PRIORITY``, and capped.
    """
    options: list[ClarificationOption] = []
    for role in _ROLE_PRIORITY:
        recognized = CANONICAL_ACTOR_ROLES.get(role, frozenset())
        in_scope = recognized & target_celexes
        if not in_scope:
            continue
        options.append(
            ClarificationOption(
                label=role.title(),
                value=role,
                celexes=frozenset(in_scope),
                frameworks=_readable_frameworks(set(in_scope)),
            )
        )
        if len(options) >= _MAX_ROLE_OPTIONS:
            break
    return options


def assess_scope(scenario: Scenario) -> ScopingResult:
    """Decide whether to ask for a decisive missing slot before answering.

    Reads the typed :class:`~application.contracts.Scenario` produced by the
    detection stage. Ask-first fires only when *all* hold, so a defensible
    answer is never blocked by a needless question:

    * the route is not one where role is irrelevant or already handled;
    * the question is not a definition or explicit-provision lookup;
    * no actor role was detected (``scenario.has_role`` is False);
    * at least one role-partitioned regulation is in scope (to build options);
    * the question is obligation-focused (asks about duties/requirements);
    * at least two candidate roles genuinely fork the analysis.
    """
    if scenario.route_id in _NO_CLARIFY_ROUTES:
        return ScopingResult(False)
    if scenario.is_definition_question or scenario.explicit_refs:
        return ScopingResult(False)
    if scenario.has_role:
        return ScopingResult(False)
    if not scenario.target_celexes:
        return ScopingResult(False)
    if not _has_obligation_focus(scenario.question):
        return ScopingResult(False)

    options = _role_options(set(scenario.target_celexes))
    if len(options) < 2:
        return ScopingResult(False)

    frameworks = _readable_frameworks(set(scenario.target_celexes))
    where = f" under the {frameworks}" if frameworks else ""
    clarification = Clarification(
        slot="actor_role",
        question=(
            f"Which role are you asking about? Obligations{where} are assigned "
            "per actor role, and each role carries different duties for the same "
            "product."
        ),
        rationale=(
            "Actor role is the backbone of an EU compliance answer — a "
            "provider/manufacturer, deployer, importer and distributor carry "
            "different obligations for the very same product. Naming it lets "
            "CRSS cite the duties that actually apply to you instead of guessing."
        ),
        options=options,
    )
    return ScopingResult(True, clarification)


def render_clarification_markdown(clarification: Clarification) -> str:
    """Render a clarification as markdown for clients that don't parse the event.

    The agent emits a structured ``clarify`` event *and* a ``done`` event whose
    answer is this markdown, so a UI that only understands ``done`` still shows
    the question instead of hanging.
    """
    lines = [
        f"**Before I can answer — {clarification.question}**",
        "",
        clarification.rationale,
        "",
    ]
    for opt in clarification.options:
        suffix = f" — _{opt.frameworks}_" if opt.frameworks else ""
        lines.append(f"- **{opt.label}**{suffix}")
    lines.extend(
        [
            "",
            "_Reply with your role (or “not sure”) and I'll give the "
            "obligations that apply specifically to you._",
        ]
    )
    return "\n".join(lines)
