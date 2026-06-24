"""Typed read-path contracts — Phase 1 of the read-path rewrite.

The retriever and agent currently pass provisions and definitions as
``dict``-of-unknown-keys, and the question's understood scope as a bag of loose
local variables. That ad-hoc shape is what let the faithfulness corpus silently
drift (a field stopped being included and nothing caught it). These contracts
give the read path one authoritative definition of each object.

**Phase 1 is additive and zero-behaviour-change.** `Provision` and `Definition`
are *typed views over the canonical dict* — the dict remains the source of truth,
``to_dict()`` returns it unchanged, so existing dict-consuming code is unaffected.
New code reads typed properties instead of scattered ``.get("…")`` calls. Later
phases migrate the hot path onto these types and then harden them into owned
state. `Scenario` and `Evidence` are new aggregates with no current dict form.

Field names mirror the retriever's ``RETURN … AS`` aliases (see
``retrieval/graph_retriever.py``). Nothing here imports from the rest of
``application/`` so it cannot create import cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Provision — typed view over the canonical retriever dict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provision:
    """A retrieved provision. Thin typed view; ``raw`` is the source of truth."""

    raw: dict[str, Any]

    # --- identity ---
    @property
    def article_id(self) -> str | None:
        return self.raw.get("article_id") or self.raw.get("id") or self.raw.get("provision_id")

    @property
    def celex(self) -> str | None:
        return self.raw.get("celex")

    @property
    def article_ref(self) -> str | None:
        return self.raw.get("article_ref") or self.raw.get("display_ref")

    @property
    def identity(self) -> str:
        """Stable identity for dedup / baseline diffing.

        Prefers the canonical node id; falls back to ``celex|article_ref`` only
        when no id is present. A bare ``display_ref`` is never used as identity —
        annex sub-node display_refs are non-unique.
        """
        aid = self.article_id
        if aid:
            return str(aid)
        ref = self.raw.get("article_ref") or self.raw.get("ref") or "?"
        return f"{self.celex or '?'}|{ref}"

    # --- text ---
    @property
    def article_text(self) -> str:
        return str(self.raw.get("article_text") or self.raw.get("text") or "")

    @property
    def children(self) -> list[dict[str, Any]]:
        return self.raw.get("children") or []

    @property
    def matched_leaf_id(self) -> str | None:
        return self.raw.get("matched_leaf_id")

    # --- classification / provenance ---
    @property
    def binding_force(self) -> str | None:
        return self.raw.get("binding_force")

    @property
    def provision_role(self) -> str | None:
        return self.raw.get("provision_role")

    @property
    def regulation(self) -> str | None:
        return self.raw.get("regulation")

    @property
    def interpreting_guidance(self) -> list[dict[str, Any]]:
        return self.raw.get("interpreting_guidance") or []

    @property
    def interpreted_provisions(self) -> list[dict[str, Any]]:
        return self.raw.get("interpreted_provisions") or []

    def text_payload(self) -> str:
        """Quotable verbatim text — mirrors ``_faithfulness._provision_text``.

        Body + child texts + interpretive-link lines, exactly what the
        faithfulness corpus must contain. This contract is the single place that
        definition should live; the existing helper can delegate here later.
        """
        parts: list[str] = []
        body = self.raw.get("article_text") or self.raw.get("text") or ""
        if body:
            parts.append(str(body))
        for child in self.children:
            child_text = child.get("raw_text") or child.get("text") or ""
            if child_text:
                parts.append(str(child_text))
        for link_field in ("interpreting_guidance", "interpreted_provisions"):
            for link in self.raw.get(link_field) or []:
                link_text = link.get("text") or ""
                if link_text:
                    parts.append(str(link_text))
        return "\n".join(parts)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provision":
        return cls(raw=d)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical dict unchanged (lossless round-trip)."""
        return self.raw


# ---------------------------------------------------------------------------
# Definition — typed view over a find_by_term result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    """A formal/scoped defined-term definition. Typed view over the dict."""

    raw: dict[str, Any]

    @property
    def term(self) -> str | None:
        return self.raw.get("term")

    @property
    def term_normalized(self) -> str | None:
        return self.raw.get("term_normalized")

    @property
    def celex(self) -> str | None:
        return self.raw.get("celex")

    @property
    def article_ref(self) -> str | None:
        return self.raw.get("article_ref")

    @property
    def regulation(self) -> str | None:
        return self.raw.get("regulation")

    @property
    def definition_type(self) -> str | None:
        return self.raw.get("definition_type")

    @property
    def source_provision_id(self) -> str | None:
        return self.raw.get("source_provision_id")

    def text_payload(self) -> str:
        """Quotable text — mirrors ``_faithfulness._definition_text``."""
        return str(self.raw.get("definition_text") or self.raw.get("text") or "")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Definition":
        return cls(raw=d)

    def to_dict(self) -> dict[str, Any]:
        return self.raw


# ---------------------------------------------------------------------------
# Scenario — the understood scope of a question (new aggregate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Everything the deterministic "understand the question" stage produces.

    Today these are loose locals threaded through ``ask_stream`` (mentioned
    regs, target celexes, role specs, explicit refs, route, definition flag).
    Modelling them as one object is the spine the rewritten agent organises
    around: understand -> (clarify?) -> plan -> retrieve.
    """

    question: str
    mentioned_regs: frozenset[str] = frozenset()
    target_celexes: frozenset[str] = frozenset()
    role_specs: tuple[tuple[str, str], ...] = ()
    explicit_refs: tuple[str, ...] = ()
    route_id: str = ""
    is_definition_question: bool = False

    @property
    def has_role(self) -> bool:
        return bool(self.role_specs)

    @property
    def is_cross_regulation(self) -> bool:
        return len(self.target_celexes) > 1

    def in_scope(self, celex: str) -> bool:
        return celex in self.target_celexes


# ---------------------------------------------------------------------------
# Evidence — the retrieved bundle (new aggregate)
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """The retrieved context: provisions + definitions, with one dedup point.

    Replaces the ad-hoc per-call-site ``_merge_unique_provisions`` scattered
    across the five overlapping retrieval mechanisms (see PATCH_LEDGER A1-A6).
    """

    provisions: list[Provision] = field(default_factory=list)
    definitions: list[Definition] = field(default_factory=list)

    @classmethod
    def from_dicts(
        cls,
        provisions: list[dict[str, Any]] | None = None,
        definitions: list[dict[str, Any]] | None = None,
    ) -> "Evidence":
        return cls(
            provisions=[Provision.from_dict(p) for p in (provisions or []) if p],
            definitions=[Definition.from_dict(d) for d in (definitions or []) if d],
        )

    def provision_ids(self) -> list[str]:
        return [p.identity for p in self.provisions]

    def unique_provisions(self) -> list[Provision]:
        """Provisions deduped by identity, first occurrence wins (order kept)."""
        seen: set[str] = set()
        out: list[Provision] = []
        for p in self.provisions:
            ident = p.identity
            if ident in seen:
                continue
            seen.add(ident)
            out.append(p)
        return out

    def extend(self, other: "Evidence") -> "Evidence":
        """Merge another Evidence in, deduping provisions by identity."""
        have = set(self.provision_ids())
        for p in other.provisions:
            if p.identity not in have:
                have.add(p.identity)
                self.provisions.append(p)
        def_terms = {(d.term_normalized, d.celex) for d in self.definitions}
        for d in other.definitions:
            key = (d.term_normalized, d.celex)
            if key not in def_terms:
                def_terms.add(key)
                self.definitions.append(d)
        return self

    def provision_dicts(self) -> list[dict[str, Any]]:
        """Back to plain dicts for the legacy hot path (lossless)."""
        return [p.to_dict() for p in self.provisions]
