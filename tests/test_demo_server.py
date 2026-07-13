import importlib.util
from pathlib import Path


def _load_demo_server_module():
    server_path = Path(__file__).resolve().parents[1] / "demo" / "server.py"
    spec = importlib.util.spec_from_file_location("crss_demo_server", server_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_api_ask_returns_answer_and_audit_trace(monkeypatch):
    server = _load_demo_server_module()

    monkeypatch.setattr(
        server,
        "ask_with_trace",
        lambda question, retriever, k=5, history=None: {
            "answer": f"Answer for: {question}",
            "audit_trace": {
                "route": {"id": "definition_lookup"},
                "sufficiency": {"ok": True},
            },
        },
    )
    server.retriever = object()

    client = server.app.test_client()
    response = client.post("/api/ask", json={"question": "What is a provider?", "k": 4})

    assert response.status_code == 200
    payload = response.get_json()
    # The pilot-logging layer stamps a fresh interaction_id per request; the
    # test only pins the answer/trace contract, not the generated id.
    assert payload.pop("interaction_id", None)
    assert payload == {
        "answer": "Answer for: What is a provider?",
        "audit_trace": {
            "route": {"id": "definition_lookup"},
            "sufficiency": {"ok": True},
        },
    }
