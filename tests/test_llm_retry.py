"""Unit tests for transient-failure handling around Mistral streaming calls.

Mistral intermittently returns 5xx / 429 for a request that succeeds moments
later.  ``_stream_chat_with_retry`` retries those transient failures with
backoff, but never retries after a token has already been emitted (which would
duplicate the partial answer on the client) and never retries a deterministic
4xx.  No live LLM is required — a fake client scripts the failures.
"""
from __future__ import annotations

import pytest

import application.agent as agent
from application.agent import _is_retryable_llm_error, _stream_chat_with_retry


class _FakeSDKError(Exception):
    """Mirror of mistralai SDKError: carries a numeric ``status_code``."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(message or f"API error occurred: Status {status_code}. Body: ...")


# ── stream scaffolding: mimic chunk.data.choices[0].delta.content ───────────


class _FakeStreamCtx:
    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for d in self._deltas:
            delta = type("D", (), {"content": d})
            choice = type("C", (), {"delta": delta})
            data = type("Data", (), {"choices": [choice]})
            yield type("Chunk", (), {"data": data})


class _MidStreamFailCtx(_FakeStreamCtx):
    """Yields one delta, then fails — models a mid-stream 5xx."""

    def __iter__(self):
        yield next(iter(_FakeStreamCtx([self._deltas[0]])))
        raise _FakeSDKError(503)


class _FakeChat:
    def __init__(self, script):
        self._script = script
        self.calls = 0

    def stream(self, **_kw):
        action = self._script[self.calls]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        return action


class _FakeClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make backoff instant so the suite stays fast."""
    monkeypatch.setattr(agent.time, "sleep", lambda _s: None)


def _drain(client):
    return "".join(
        _stream_chat_with_retry(client, model="m", messages=[], temperature=0.1)
    )


# ── classification ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 599])
def test_retryable_status_codes(code):
    assert _is_retryable_llm_error(_FakeSDKError(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_non_retryable_status_codes(code):
    assert _is_retryable_llm_error(_FakeSDKError(code)) is False


def test_retryable_string_fallback_when_no_status_code():
    # A transport error with no status_code attribute -> sniff the message.
    assert _is_retryable_llm_error(RuntimeError("... Status 503 Service unavailable")) is True
    assert _is_retryable_llm_error(RuntimeError("connection ok 200")) is False


# ── streaming retry behaviour ────────────────────────────────────────────────


def test_transient_failure_then_success_retries_and_yields():
    client = _FakeClient([_FakeSDKError(503), _FakeStreamCtx(["Hel", "lo"])])
    assert _drain(client) == "Hello"
    assert client.chat.calls == 2  # failed once, succeeded on retry


def test_non_retryable_error_raises_immediately():
    client = _FakeClient([_FakeSDKError(400)])
    with pytest.raises(_FakeSDKError):
        _drain(client)
    assert client.chat.calls == 1  # no retry on a 4xx


def test_no_retry_after_first_token_emitted():
    # Once a delta has streamed, restarting would duplicate output on the client,
    # so a mid-stream failure must propagate rather than retry.
    client = _FakeClient([_MidStreamFailCtx(["partial"])])
    got: list[str] = []
    with pytest.raises(_FakeSDKError):
        for delta in _stream_chat_with_retry(
            client, model="m", messages=[], temperature=0.1
        ):
            got.append(delta)
    assert got == ["partial"]
    assert client.chat.calls == 1


def test_persistent_transient_failure_exhausts_retries():
    client = _FakeClient([_FakeSDKError(503)] * agent._LLM_MAX_RETRIES)
    with pytest.raises(_FakeSDKError):
        _drain(client)
    assert client.chat.calls == agent._LLM_MAX_RETRIES