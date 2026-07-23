"""Unit tests for the LLM-assisted residual quote repair (strict tier, mode 2).

The LLM is faked throughout — these tests pin the *deterministic* guarantees:
a `replace` only enters the answer when it verifies as an exact corpus
substring, `demote` keeps the inner text byte-for-byte, `delete` leaves the
redaction marker, and every failure path (garbage JSON, API error, rejected
replacement) leaves the answer untouched for the caller's redaction pass.
"""
from __future__ import annotations

import pytest

from application._faithfulness import (
    FaithfulnessReport,
    Quote,
    _REDACTION_MARKER,
    check_faithfulness,
    repair_and_redact,
)
from application._faithfulness_repair import _parse_decision, llm_repair_residuals


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROVISIONS = [
    {
        "article_ref": "Article 5",
        "article_text": (
            "The placing on the market, the putting into service for this "
            "specific purpose, or the use of AI systems to infer emotions of "
            "a natural person in the areas of workplace and education "
            "institutions shall be prohibited."
        ),
    },
    {
        "article_ref": "Article 43",
        "article_text": (
            "Where a high-risk AI system is to undergo a substantial "
            "modification, the high-risk AI system shall undergo a new "
            "conformity assessment procedure."
        ),
    },
]

# Own-prose analysis wrapped in quote marks (the HQ_009/HQ_016 shape): grounds
# nowhere, and no source paraphrase-matches it.
_OWN_PROSE = (
    "While the definitions align across both regulations, the AI Act imposes "
    "additional transparency duties on deployers"
)
# A fake "legal quote" typed from memory (the HQ_036 shape).
_MEMORY_QUOTE = (
    "Providers of medium-risk systems shall maintain a register of all "
    "deployments carried out within the Union"
)


def _answer_with(inner: str) -> tuple[str, Quote]:
    """Build an answer holding one straight-quoted span + its Quote (marks included)."""
    answer = (
        f'Under [Article 5] the position is clear: "{inner}" — which resolves '
        "the question for the deploying hospital."
    )
    start = answer.index('"')
    end = answer.index('"', start + 1) + 1
    return answer, Quote(text=inner, start=start, end=end)


class _FakeResp:
    def __init__(self, content: str):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]


class _FakeChat:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    def complete(self, **_kw):
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResp(payload)


class _FakeClient:
    def __init__(self, *payloads):
        self.chat = _FakeChat(payloads)


# ---------------------------------------------------------------------------
# Action application + gates
# ---------------------------------------------------------------------------


def test_demote_strips_marks_and_keeps_text_byte_for_byte():
    answer, q = _answer_with(_OWN_PROSE)
    report = FaithfulnessReport(total_quotes=1, unverified=[q])
    client = _FakeClient('{"action": "demote", "replacement": "", "source": null}')

    out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)

    assert f'"{_OWN_PROSE}"' not in out          # marks gone
    assert _OWN_PROSE in out                     # text intact
    assert len(notes) == 1 and "own wording" in notes[0]
    # The demoted span is no longer a quote claim → checker is clean.
    assert check_faithfulness(out, _PROVISIONS).ok


def test_replace_applies_only_an_exact_corpus_copy():
    answer, q = _answer_with(_MEMORY_QUOTE)
    report = FaithfulnessReport(total_quotes=1, unverified=[q])
    exact = (
        "the use of AI systems to infer emotions of a natural person in the "
        "areas of workplace and education institutions shall be prohibited"
    )
    client = _FakeClient(
        '{"action": "replace", "replacement": "' + exact + '", "source": 1}'
    )

    out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)

    assert "“" + exact + "”" in out              # substituted, smart-quoted
    assert _MEMORY_QUOTE not in out
    assert notes and notes[0].startswith("quote corrected")
    assert check_faithfulness(out, _PROVISIONS).ok


def test_replace_on_misattributed_quote_also_repoints_the_citation():
    # Real Article 5 text quoted under an [Article 43] label: replacing only the
    # quote text would leave it displaced and re-flag on the final pass — the
    # citation label must be re-pointed to the chosen source too.
    exact = (
        "the use of AI systems to infer emotions of a natural person in the "
        "areas of workplace and education institutions shall be prohibited"
    )
    answer = (
        f'Under [Article 43] the rule is: "{exact}" — hence the prohibition.'
    )
    start = answer.index('"')
    end = answer.index('"', start + 1) + 1
    q = Quote(text=answer[start + 1 : end - 1], start=start, end=end)
    report = FaithfulnessReport(total_quotes=1, misattributed=[q])
    client = _FakeClient(
        '{"action": "replace", "replacement": "' + exact + '", "source": 1}'
    )

    out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)

    assert "“" + exact + "”" in out
    assert "[Article 5]" in out                  # label re-pointed…
    assert "[Article 43]" not in out             # …away from the wrong provision
    assert any(n.startswith("citation corrected") for n in notes)
    assert check_faithfulness(out, _PROVISIONS).ok


def test_replace_repoints_a_cite_after_label_too():
    # Cite-after layout: “…” (Article 43). The checker adjudicates against the
    # trailing label, so the repoint must edit that same label — before the
    # _citation_ref_span fix this shape was flaggable but unfixable (HQ_006's
    # chronic pattern).
    exact = (
        "the use of AI systems to infer emotions of a natural person in the "
        "areas of workplace and education institutions shall be prohibited"
    )
    answer = f'The prohibition is explicit: "{exact}" (Article 43) — full stop.'
    start = answer.index('"')
    end = answer.index('"', start + 1) + 1
    q = Quote(text=answer[start + 1 : end - 1], start=start, end=end)
    report = FaithfulnessReport(total_quotes=1, misattributed=[q])
    client = _FakeClient(
        '{"action": "replace", "replacement": "' + exact + '", "source": 1}'
    )

    out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)

    assert "(Article 5)" in out
    assert "Article 43" not in out
    assert any(n.startswith("citation corrected") for n in notes)
    assert check_faithfulness(out, _PROVISIONS).ok


def test_replace_with_hallucinated_text_is_rejected():
    answer, q = _answer_with(_MEMORY_QUOTE)
    report = FaithfulnessReport(total_quotes=1, unverified=[q])
    client = _FakeClient(
        '{"action": "replace", "replacement": "Providers shall notify the '
        'Commission of every deployment within thirty days of go-live", '
        '"source": 1}'
    )

    out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)

    assert out == answer                         # nothing entered the answer
    assert notes == []


def test_delete_leaves_the_redaction_marker():
    answer, q = _answer_with(_MEMORY_QUOTE)
    report = FaithfulnessReport(total_quotes=1, unverified=[q])
    client = _FakeClient('{"action": "delete", "replacement": "", "source": null}')

    out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)

    assert _MEMORY_QUOTE not in out
    assert _REDACTION_MARKER in out
    assert notes and "removed" in notes[0]


def test_garbage_json_and_api_errors_leave_answer_untouched():
    answer, q = _answer_with(_MEMORY_QUOTE)
    report = FaithfulnessReport(total_quotes=1, unverified=[q])
    for client in (
        _FakeClient("I think you should probably rephrase this."),
        _FakeClient(RuntimeError("boom")),
        _FakeClient('{"action": "rewrite-everything"}'),
    ):
        out, notes = llm_repair_residuals(answer, report, _PROVISIONS, client=client)
        assert out == answer
        assert notes == []


def test_no_offenders_makes_no_client_calls():
    answer, _q = _answer_with(_OWN_PROSE)
    report = FaithfulnessReport(total_quotes=0)
    out, notes = llm_repair_residuals(
        answer, report, _PROVISIONS, client=_FakeClient()
    )
    assert out == answer and notes == []


def test_parse_decision_accepts_fenced_json_and_rejects_junk():
    assert _parse_decision('```json\n{"action": "demote"}\n```')["action"] == "demote"
    assert _parse_decision("no json here") is None
    assert _parse_decision('{"action": "explode"}') is None


# ---------------------------------------------------------------------------
# repair_and_redact redact_residuals=False (the strict-tier entry)
# ---------------------------------------------------------------------------


def test_repair_and_redact_can_keep_residuals_in_place():
    answer, q = _answer_with(_MEMORY_QUOTE)
    report = FaithfulnessReport(total_quotes=1, unverified=[q])

    kept, _residual, _notes = repair_and_redact(
        answer, report, _PROVISIONS, redact_residuals=False
    )
    assert _MEMORY_QUOTE in kept                 # offender still present
    assert _REDACTION_MARKER not in kept

    redacted, _residual2, _notes2 = repair_and_redact(answer, report, _PROVISIONS)
    assert _MEMORY_QUOTE not in redacted         # default behaviour unchanged
    assert _REDACTION_MARKER in redacted


# ---------------------------------------------------------------------------
# verify.py wiring: mode 2 engages the LLM tier; mode 1 is untouched
# ---------------------------------------------------------------------------


def test_strict_mode_demotes_instead_of_flagging(monkeypatch):
    from application.verify import _apply_faithfulness
    import application._faithfulness_repair as repair_mod

    answer, _q = _answer_with(_OWN_PROSE)

    def _fake_llm_repair(ans, report, provisions, definitions=None, **_kw):
        edits = sorted(
            report.unverified + report.misattributed,
            key=lambda x: x.start, reverse=True,
        )
        for quote in edits:
            ans = ans[: quote.start] + quote.text + ans[quote.end :]
        return ans, ["quotation marks removed from a span that was the answer's own wording"]

    monkeypatch.setattr(repair_mod, "llm_repair_residuals", _fake_llm_repair)

    out, report = _apply_faithfulness(
        answer, _PROVISIONS, [], faith_mode=2, question=None
    )
    assert report is not None and not report.ok  # original report keeps the defect
    assert "FAITHFULNESS FLAG" not in out        # but the answer was repaired…
    assert _OWN_PROSE in out                     # …with the text kept as prose
    assert "Auto-verified corrections" in out    # and the repair disclosed


def test_flag_mode_still_redacts_and_flags():
    from application.verify import _apply_faithfulness

    answer, _q = _answer_with(_OWN_PROSE)
    out, report = _apply_faithfulness(
        answer, _PROVISIONS, [], faith_mode=1, question=None
    )
    assert report is not None and not report.ok
    assert "FAITHFULNESS FLAG" in out
    assert f'"{_OWN_PROSE}"' not in out