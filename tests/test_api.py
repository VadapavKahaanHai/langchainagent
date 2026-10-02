"""API and actual agent-loop checks using local model responses, never Gemini."""

import pandas as pd
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.main import app
from app.api import endpoints
from app.agent import agent_service, memory
from app.tools import filing_review

client = TestClient(app)


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_routes_and_direct_tool(monkeypatch):
    monkeypatch.setattr(endpoints, "load_filings", lambda: pd.DataFrame(index=["filing_001"]))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/api/v1/agent/filings").json() == ["filing_001"]
    assert [t["name"] for t in client.get("/api/v1/agent/tools").json()] == ["review_filing"]
    monkeypatch.setattr(filing_review, "review_filing_data", lambda record_id: {"record_id": record_id, "predicted_label": "yes"})
    response = client.post("/api/v1/agent/tools/execute/review_filing", json={"record_id": "filing_001"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert client.post("/api/v1/agent/tools/execute/python_repl", json={}).status_code == 404
    assert client.post("/api/v1/agent/tools/execute/review_filing", json={}).status_code == 400
    assert client.get("/api/v1/agent/providers").status_code == 404
    for payload in [{"query": " "}, {"query": "hello", "provider": "mock"}, {"query": "hello", "agent_type": "react"}]:
        assert client.post("/api/v1/agent/query", json=payload).status_code == 422


def test_agent_tool_loop_and_history(monkeypatch):
    seen = []
    def review(record_id):
        seen.append(record_id)
        return {"record_id": record_id, "predicted_label": "yes", "excerpts": []}
    monkeypatch.setattr(filing_review, "review_filing_data", review)
    llm = ToolCallingModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "review_filing", "args": {"record_id": "filing_001"}, "id": "test-call", "type": "tool_call"}]),
        AIMessage(content=[{"type": "text", "text": "Exploratory review."}, {"type": "text", "text": "Not proof of fraud."}]),
    ])
    monkeypatch.setattr(agent_service, "create_llm", lambda **kwargs: llm)
    memory.clear_session_history("test-review")
    response = client.post("/api/v1/agent/query", json={"query": "Review filing_001", "session_id": "test-review"})
    assert response.status_code == 200, response.text
    assert seen == ["filing_001"]
    result = response.json()
    assert result["response"] == "Exploratory review.\nNot proof of fraud."
    assert result["steps"][0]["tool"] == "review_filing"
    assert result["provider_used"] == "gemini"
    history = client.get("/api/v1/agent/history/test-review").json()
    assert history["message_count"] == 2
    assert history["messages"][1]["content"] == result["response"]
    assert client.delete("/api/v1/agent/history/test-review").json()["cleared"]
    assert not client.delete("/api/v1/agent/history/test-review").json()["cleared"]


def test_execution_errors_are_not_success(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("simulated provider failure")
    monkeypatch.setattr(agent_service, "create_llm", fail)
    response = client.post("/api/v1/agent/query", json={"query": "Review filing_001", "session_id": "failed-review"})
    assert response.status_code == 500
    assert not memory.get_session_history("failed-review").messages
    memory.clear_session_history("failed-review")
