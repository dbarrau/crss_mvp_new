"""The auditor must be told that a controlling amendment in the retrieved context
is current, in-force law — otherwise a mid-tier model whose training predates the
amending act (e.g. the 2026 Omnibus) "corrects" real inserted paragraphs
(Article 6(1a)-(1c)) as fabrication and can trigger a needless revision. The
directive is injected ONLY when the amendment marker is present, so a normal
audit is byte-for-byte unchanged.
"""
from application._audit import _audit_answer

_PASS_JSON = (
    '{"initial_status_correct": true, "primary_route_correct": true, '
    '"issues": [], "missing_provision_refs": [], "missing_topics": [], '
    '"verdict": "PASS"}'
)


class _FakeChat:
    def __init__(self):
        self.captured = None

    def complete(self, *, model, temperature, messages):
        self.captured = messages
        return type("R", (), {"choices": [
            type("C", (), {"message": type("M", (), {"content": _PASS_JSON})()})()
        ]})()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def _audit_user_message(context: str) -> str:
    client = _FakeClient()
    _audit_answer("q?", context, "draft answer", client, model="mistral-medium-latest")
    return client.chat.captured[1]["content"]  # [0]=system, [1]=user


def test_amendment_directive_injected_when_marker_present():
    ctx = ("⚠ AMENDING PROVISION — CONTROLLING (supersedes the original wording of "
           "Article 6; source: Regulation (EU) 2026/1744):\nArticle 6(1a) …")
    msg = _audit_user_message(ctx)
    assert "AMENDMENTS — READ FIRST" in msg
    assert "AUTHORITATIVE over your own training" in msg
    assert "do NOT flag it as non-existent" in msg


def test_no_amendment_directive_without_marker():
    msg = _audit_user_message("Article 6 of the AI Act sets the classification rules.")
    assert "AMENDMENTS — READ FIRST" not in msg
    assert "AUTHORITATIVE over your own training" not in msg
