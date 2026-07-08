"""Invariants for the structured-output system prompt.

The structured path must not carry any inline `[cite:]`/`[quote:]` pointer
instruction (those competed with the marker/citations channel and produced empty
"[]" litter + lost citations), while still keeping the shared domain rules.
"""
import re

import pytest

from application._prompts import SYSTEM_PROMPT, structured_system_prompt


def test_structured_prompt_has_no_inline_pointer_tokens():
    sp = structured_system_prompt()
    assert "[cite: <id>]" not in sp
    assert "[quote: <id>]" not in sp
    assert "GROUNDED CITATION CONTRACT" not in sp


def test_structured_prompt_defines_the_marker_contract():
    sp = structured_system_prompt()
    assert "STRUCTURED OUTPUT MODE (mandatory citation contract)" in sp
    assert "[[marker]]" in sp
    # The rule that prevents the empty-"[]" litter must be present.
    assert "matching entry in `citations`" in sp


def test_structured_prompt_preserves_shared_domain_rules():
    sp = structured_system_prompt()
    for section in (
        "LEGAL FORCE AWARENESS",
        "CROSS-REGULATION AWARENESS",
        "BALANCED ANALYSIS",
    ):
        assert section in sp


def test_inline_prompt_still_uses_the_inline_contract():
    # The default streaming path is unchanged: it keeps the inline pointer contract.
    assert "GROUNDED CITATION CONTRACT" in SYSTEM_PROMPT
    assert "[cite: <id>]" in SYSTEM_PROMPT


def test_structured_prompt_raises_if_base_contract_block_missing(monkeypatch):
    # If the base prompt structure drifts so the contract block can't be found,
    # the swap must fail loudly rather than ship a conflicted prompt.
    monkeypatch.setattr(
        "application._prompts.SYSTEM_PROMPT", "no contract block here", raising=True
    )
    with pytest.raises(RuntimeError):
        structured_system_prompt()
