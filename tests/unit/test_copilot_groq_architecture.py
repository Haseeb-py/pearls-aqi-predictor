"""Production-mode contract tests for Groq Copilot orchestration."""

import json
from types import SimpleNamespace

from pearls_aqi import copilot


def _groq_response(content):
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"choices": [{"message": {"content": content}}]},
    )


def _status(city):
    return {
        "city": city, "aqi": 88.0, "category": "Moderate",
        "observed_at_utc": "2026-01-01T00:00:00+00:00", "is_stale": False,
        "aqi_change_24h": 0.0,
    }


def _forecast(city):
    return {
        "city": city, "issued_at_utc": "2026-01-01T00:00:00+00:00", "is_stale": False,
        "forecasts": [
            {"horizon_hours": 24, "aqi": 90.0, "category": "Moderate"},
            {"horizon_hours": 48, "aqi": 92.0, "category": "Moderate"},
            {"horizon_hours": 72, "aqi": 95.0, "category": "Moderate"},
        ],
    }


def _enable_production(monkeypatch):
    monkeypatch.setattr(copilot.settings, "APP_ENV", "production")
    monkeypatch.setattr(copilot.settings, "GROQ_API_KEY", "test-key")


def test_production_turn_calls_groq_for_crisis_intent_and_final_response(monkeypatch):
    _enable_production(monkeypatch)
    calls = []
    replies = iter([
        '{"risk": false}',
        '{"intent":"current_forecast","cities":["lahore"],"history_hours":72,"horizon_hours":24,"needs_city_clarification":false}',
        "Lahore is in the Moderate range at 88 AQI. Tomorrow is forecast near 90 AQI.",
    ])

    def fake_post(_url, **kwargs):
        calls.append(kwargs["json"])
        return _groq_response(next(replies))

    monkeypatch.setattr(copilot.requests, "post", fake_post)
    monkeypatch.setitem(copilot.TOOLS, "get_current_aqi", _status)
    monkeypatch.setitem(copilot.TOOLS, "get_aqi_forecast", _forecast)
    result = copilot.chat("What's Lahore's current AQI?", history=["Earlier message", "Another", "Recent", "Ignored"])

    assert result["provider"] == "groq_grounded"
    assert result["answer"].startswith("Lahore is in the Moderate")
    assert result["tools_used"] == ["get_current_aqi", "get_aqi_forecast"]
    assert len(calls) == 3
    assert json.loads(calls[0]["messages"][1]["content"])["recent_messages"] == ["Another", "Recent", "Ignored"]
    assert "Classify the user's meaning" in calls[1]["messages"][0]["content"]
    assert "tool_evidence" in calls[2]["messages"][1]["content"]


def test_semantic_crisis_result_stops_before_intent_or_tools(monkeypatch):
    _enable_production(monkeypatch)
    calls = []

    def fake_post(_url, **kwargs):
        calls.append(kwargs["json"])
        return _groq_response('{"risk": true}')

    monkeypatch.setattr(copilot.requests, "post", fake_post)
    result = copilot.chat("I feel like giving up, honestly", history=["I have felt hopeless all day."])

    assert result["provider"] == "safety_response"
    assert result["tools_used"] == []
    assert len(calls) == 1


def test_production_history_route_never_runs_forecast(monkeypatch):
    _enable_production(monkeypatch)
    replies = iter([
        '{"risk": false}',
        '{"intent":"history","cities":["karachi"],"history_hours":168,"horizon_hours":24,"needs_city_clarification":false}',
        "Karachi improved from 170 to 150 AQI over the available history.",
    ])
    monkeypatch.setattr(copilot.requests, "post", lambda _url, **_kwargs: _groq_response(next(replies)))
    monkeypatch.setitem(copilot.TOOLS, "get_aqi_history", lambda city, hours: {
        "city": city, "hours": hours, "is_stale": False,
        "points": [{"time": "2026-01-01T00:00:00+00:00", "aqi": 170.0}, {"time": "2026-01-02T00:00:00+00:00", "aqi": 150.0}],
    })
    monkeypatch.setitem(copilot.TOOLS, "get_aqi_forecast", lambda _city: (_ for _ in ()).throw(AssertionError("forecast must not run")))
    result = copilot.chat("What was Karachi's AQI last week?")

    assert result["tools_used"] == ["get_aqi_history"]
    assert result["evidence"]["history"]["hours"] == 168


def test_production_ambiguous_route_generates_clarification(monkeypatch):
    _enable_production(monkeypatch)
    replies = iter([
        '{"risk": false}',
        '{"intent":"clarify_city","cities":[],"history_hours":72,"horizon_hours":24,"needs_city_clarification":true}',
        "Which supported city would you like me to check?",
    ])
    monkeypatch.setattr(copilot.requests, "post", lambda _url, **_kwargs: _groq_response(next(replies)))
    result = copilot.chat("the one with the worst air")

    assert result["provider"] == "groq_grounded"
    assert result["tools_used"] == []
    assert result["answer"] == "Which supported city would you like me to check?"

def test_production_semantic_crisis_classifier_catches_new_indirect_phrasings(monkeypatch):
    _enable_production(monkeypatch)
    messages = [
        "I feel like giving up, honestly",
        "Nobody would notice or care either way",
        "I just want to disappear",
        "I keep thinking it would be easier if I just wasn't around",
    ]
    for message in messages:
        calls = []
        monkeypatch.setattr(
            copilot.requests,
            "post",
            lambda _url, **kwargs: calls.append(kwargs["json"]) or _groq_response('{"risk": true}'),
        )
        result = copilot.chat(message, history=["I have been feeling hopeless lately."])
        assert result["provider"] == "safety_response", message
        assert result["tools_used"] == [], message
        assert len(calls) == 1, message
