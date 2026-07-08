"""Economy guard for grounded quotes: dedupe repeated quotes + soft length cap.

Structured output moved verbatim expansion into the renderer, hiding the cost of
quoting from the model so it over-quotes (whole sections, duplicated). These
deterministic levers restore economy where expansion now lives. No LLM / no I/O.
"""
import pytest

from application._grounded_citation import (
    _cap_quote_text,
    build_pointer_index,
    quote_char_cap,
    resolve_pointers,
)
from application._grounded_answer import (
    Citation,
    GroundedAnswer,
    render_grounded_answer,
)


def _big_index():
    long_body = (
        "First operative sentence establishing the rule. "
        + "Filler clause number %d that pads the provision well beyond the cap. " % 0
        + " ".join(f"Extra clause {i}." for i in range(120))
    )
    return build_pointer_index(
        [
            {
                "article_id": "REG_art_1",
                "article_ref": "Article 1",
                "regulation": "REG",
                "binding_force": "binding",
                "article_text": long_body,
                "children": [],
            },
            {
                "article_id": "REG_art_2",
                "article_ref": "Article 2",
                "regulation": "REG",
                "binding_force": "binding",
                "article_text": "Short operative clause.",
                "children": [],
            },
        ]
    )


# --- cap -------------------------------------------------------------------


def test_cap_truncates_at_sentence_boundary_with_elision():
    text = "First sentence. " + "x" * 2000
    out = _cap_quote_text(text, 600)
    assert out.endswith("[…]")
    assert len(out) <= 640  # cap + a little slack for the elision marker
    assert out.startswith("First sentence.")


def test_cap_zero_disables():
    text = "a" * 5000
    assert _cap_quote_text(text, 0) == text


def test_short_quote_is_untouched():
    assert _cap_quote_text("Short operative clause.", 600) == "Short operative clause."


def test_quote_char_cap_env_override(monkeypatch):
    monkeypatch.setenv("CRSS_QUOTE_CHAR_CAP", "50")
    assert quote_char_cap() == 50
    monkeypatch.setenv("CRSS_QUOTE_CHAR_CAP", "not-an-int")
    assert quote_char_cap() == 600  # falls back to default


def test_inline_render_caps_long_quote(monkeypatch):
    monkeypatch.setenv("CRSS_QUOTE_CHAR_CAP", "600")
    out = resolve_pointers("Rule: [quote: REG_art_1]", _big_index())
    assert "[…]" in out.text
    # A capped 600-char quote is far shorter than the ~7k-char source body.
    assert len(out.text) < 1200


# --- dedupe ----------------------------------------------------------------


def test_inline_repeat_quote_downgraded_to_cite():
    idx = _big_index()
    out = resolve_pointers(
        "IIa: [quote: REG_art_2]. IIb: [quote: REG_art_2]. III: [quote: REG_art_2].",
        idx,
    )
    # Verbatim rendered once; the two repeats become the reference.
    assert out.text.count("> Short operative clause.") == 1
    assert out.quoted_ids == ["REG_art_2"]
    assert out.deduped_ids == ["REG_art_2", "REG_art_2"]
    assert out.text.count("Article 2 REG") == 2


def test_structured_repeat_quote_downgraded_to_cite():
    idx = _big_index()
    ans = GroundedAnswer(
        body="One [[a]], two [[b]], three [[c]].",
        citations=[
            Citation(marker="a", node_id="REG_art_2", mode="quote"),
            Citation(marker="b", node_id="REG_art_2", mode="quote"),
            Citation(marker="c", node_id="REG_art_2", mode="quote"),
        ],
    )
    out = render_grounded_answer(ans, idx)
    assert out.text.count("> Short operative clause.") == 1
    assert out.quoted_ids == ["REG_art_2"]
    assert out.deduped_ids == ["REG_art_2", "REG_art_2"]


def test_structured_render_caps_long_quote():
    idx = _big_index()
    ans = GroundedAnswer(
        body="See [[q]].",
        citations=[Citation(marker="q", node_id="REG_art_1", mode="quote")],
    )
    out = render_grounded_answer(ans, idx)
    assert "[…]" in out.text
    assert out.text.startswith("See > First operative sentence") or "First operative sentence" in out.text
