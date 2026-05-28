from application.agent import ask_with_trace


def test_ask_with_trace_returns_answer_and_audit_trace(monkeypatch):
    events = [
        {
            "type": "audit",
            "trace": {
                "route": {"id": "provision_lookup"},
                "sufficiency": {"ok": True},
            },
        },
        {
            "type": "done",
            "answer": "Grounded answer",
        },
    ]

    def fake_ask_stream(question, retriever, k=20):
        assert question == "What does Article 26 require?"
        assert retriever == "fake-retriever"
        assert k == 7
        yield from events

    monkeypatch.setattr("application.agent.ask_stream", fake_ask_stream)

    result = ask_with_trace("What does Article 26 require?", "fake-retriever", k=7)

    assert result == {
        "answer": "Grounded answer",
        "audit_trace": {
            "route": {"id": "provision_lookup"},
            "sufficiency": {"ok": True},
        },
    }
