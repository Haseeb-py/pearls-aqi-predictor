from fastapi.testclient import TestClient

from pearls_aqi.api.main import app


def test_copilot_rejects_blank_message():
    response = TestClient(app).post("/api/v1/copilot/chat", json={"message": ""})
    assert response.status_code == 422


def test_copilot_uses_grounded_tools(monkeypatch):
    monkeypatch.setattr("pearls_aqi.api.main.chat", lambda message, cities, history: {"answer": "grounded", "tools_used": ["get_current_aqi"], "evidence": {"lahore": {}}, "provider": "deterministic_grounded"})
    response = TestClient(app).post("/api/v1/copilot/chat", json={"message": "Lahore AQI"})
    assert response.status_code == 200
    assert response.json()["provider"] == "deterministic_grounded"

def test_copilot_unexpected_error_returns_safe_response(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("unexpected provider failure")

    monkeypatch.setattr("pearls_aqi.api.main.chat", fail)

    response = TestClient(app).post(
        "/api/v1/copilot/chat",
        json={"message": "Tell me a joke about Lahore"},
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "service_fallback"
    assert response.json()["tools_used"] == []
