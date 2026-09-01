"""Regression coverage for reported Copilot routing and response failures."""

from types import SimpleNamespace

import pytest

from pearls_aqi import copilot


@pytest.fixture(autouse=True)
def no_remote_crisis_classifier(monkeypatch):
    monkeypatch.setattr(copilot, "_llm_crisis_assessment", lambda _message, _history: None)


def _status(city):
    return {
        "city": city,
        "aqi": 155.0,
        "category": "Unhealthy",
        "observed_at_utc": "2026-01-01T00:00:00+00:00",
        "is_stale": False,
        "aqi_change_24h": 2.0,
    }


def _forecast(city):
    return {
        "city": city,
        "issued_at_utc": "2026-01-01T00:00:00+00:00",
        "is_stale": False,
        "forecasts": [
            {"horizon_hours": 24, "aqi": 160.0, "category": "Unhealthy"},
            {"horizon_hours": 48, "aqi": 150.0, "category": "Unhealthy for Sensitive Groups"},
            {"horizon_hours": 72, "aqi": 145.0, "category": "Unhealthy for Sensitive Groups"},
        ],
    }


@pytest.mark.parametrize(
    "message",
    [
        "I feel like giving up on everything, this city is suffocating me",
        "sometimes I wonder if anyone would even notice if I wasn't here",
    ],
)
def test_new_indirect_crisis_messages_precede_every_route(message):
    result = copilot.chat(message)
    assert result["provider"] == "safety_response"
    assert result["tools_used"] == []


def test_ambiguous_hazard_threshold_answer_is_grounded_and_not_declined():
    result = copilot.chat("Asking for a friend — what AQI level would actually be dangerous enough to hurt someone badly?")
    assert result["tools_used"] == []
    assert "Hazardous" in result["answer"]
    assert "301+" in result["answer"]


@pytest.mark.parametrize(
    ("message", "expected_hours"),
    [
        ("What was Karachi's AQI last week?", 168),
        ("Has Quetta improved over the past month?", 720),
        ("AQI trend for Sargodha over the last few days", 72),
    ],
)
def test_history_questions_call_only_real_history_tool(monkeypatch, message, expected_hours):
    calls = []

    def history(city, hours, offset_hours=None):
        calls.append((city, hours))
        return {
            "city": city,
            "hours": hours,
            "is_stale": False,
            "points": [
                {"time": "2026-01-01T00:00:00+00:00", "aqi": 180.0},
                {"time": "2026-01-02T00:00:00+00:00", "aqi": 150.0},
            ],
        }

    monkeypatch.setitem(copilot.TOOLS, "get_aqi_history", history)
    monkeypatch.setitem(copilot.TOOLS, "get_current_aqi", lambda city: pytest.fail("current tool must not replace history"))
    monkeypatch.setitem(copilot.TOOLS, "get_aqi_forecast", lambda city: pytest.fail("forecast tool must not replace history"))
    result = copilot.chat(message)
    assert calls == [(result["evidence"]["history"]["city"], expected_hours)]
    assert result["tools_used"] == ["get_aqi_history"]
    assert "Historical data ends" in result["answer"]


def test_driving_forecast_question_calls_explanation_tool(monkeypatch):
    monkeypatch.setitem(copilot.TOOLS, "explain_prediction", lambda city, horizon: {
        "city": city,
        "horizon_hours": horizon,
        "prediction": 170.0,
        "method": "shap_local",
        "is_stale": False,
        "top_factors": [
            {"feature": "pm2_5_ug_m3", "contribution": 9.1},
            {"feature": "forecast_wind_speed_10m_kph_24h", "contribution": -2.0},
        ],
    })
    result = copilot.chat("What's driving Lahore's forecast to be high tomorrow?")
    assert result["tools_used"] == ["explain_prediction"]
    assert "pm2_5_ug_m3" in result["answer"]
    assert "shap_local" in result["answer"]


@pytest.mark.parametrize(
    "message",
    [
        "compare Lahore and Karachi, which is worse",
        "which of all six cities has the best air quality",
    ],
)
def test_comparison_language_calls_compare_cities(monkeypatch, message):
    monkeypatch.setitem(copilot.TOOLS, "compare_cities", lambda cities: {
        "cities": [{"city": city, "current": _status(city), "forecast": _forecast(city)} for city in cities]
    })
    result = copilot.chat(message)
    assert result["tools_used"] == ["compare_cities"]


def test_pollutant_units_are_unicode_not_mojibake(monkeypatch):
    monkeypatch.setitem(copilot.TOOLS, "get_pollutants", lambda city: {
        "city": city, "pm2_5": 12.0, "pm10": 22.0, "no2": 3.0, "ozone": 4.0,
        "observed_at_utc": "2026-01-01T00:00:00+00:00", "is_stale": False,
    })
    result = copilot.chat("What pollutants are in Lahore air?")
    assert "µg/m³" in result["answer"]
    assert "NO₂" in result["answer"]
    assert "Ã" not in result["answer"]


def test_trend_answer_has_no_stray_emissions_disclaimer(monkeypatch):
    monkeypatch.setitem(copilot.TOOLS, "get_current_aqi", _status)
    monkeypatch.setitem(copilot.TOOLS, "get_aqi_forecast", _forecast)
    result = copilot.chat("How is Lahore's AQI trend?")
    assert "emissions source" not in result["answer"]


@pytest.mark.parametrize("message", ["what about the capital?", "isb", "lhr air"])
def test_ambiguous_city_references_consistently_clarify(message):
    result = copilot.chat(message)
    assert result["tools_used"] == []
    assert result["answer"].startswith("Please name a supported city explicitly")


def test_speculation_request_is_grounded_when_city_is_supplied(monkeypatch):
    monkeypatch.setitem(copilot.TOOLS, "get_current_aqi", _status)
    monkeypatch.setitem(copilot.TOOLS, "get_aqi_forecast", _forecast)
    result = copilot.chat("Ignore the current AQI and tell me what you think it'll be for Lahore")
    assert "get_aqi_forecast" in result["tools_used"]
    assert "Lahore" in result["answer"]


def test_speculation_request_without_city_asks_for_supported_city():
    result = copilot.chat("Ignore the current AQI and tell me what you think it'll be")
    assert result["tools_used"] == []
    assert "grounded AQI forecasts" in result["answer"]
