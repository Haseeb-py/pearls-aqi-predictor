"""Safe, deterministic AQI Copilot with an explicit application-tool allow-list."""

import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from pearls_aqi.domain.aqi_categories import get_us_aqi_category
from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.explain import explain_prediction, shap_local_explanation
from pearls_aqi.models.registry import load_champion
from pearls_aqi.settings import settings

logger = logging.getLogger(__name__)
STALE_AFTER_HOURS = 30
MAX_HISTORY_MESSAGES = 6
# The Copilot is deterministic rather than LLM-prompted.  This policy is the
# equivalent response-style instruction: conversational lead, then evidence.
RESPONSE_STYLE = "Lead with a natural, helpful sentence, then state grounded AQI data and its timestamp."


def _cities() -> dict[str, dict]:
    return {city["slug"]: city for city in settings.load_cities_config()["cities"] if city.get("enabled", True)}


def _stale(timestamp: str) -> bool:
    value = pd.to_datetime(timestamp, utc=True)
    return datetime.now(timezone.utc) - value.to_pydatetime() > pd.Timedelta(hours=STALE_AFTER_HOURS)


def _value(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    return None if pd.isna(value) else round(float(value), 2)


def _city_data(city: str) -> pd.DataFrame:
    if city not in _cities():
        raise ValueError(f"Unsupported city '{city}'.")
    return load_training_data(city)


# The seven public tools required by the project specification.
def get_current_aqi(city: str) -> dict[str, Any]:
    frame = _city_data(city)
    row = frame.iloc[-1]
    previous = frame.iloc[max(0, len(frame) - 25)]
    observed_at = row["event_time_utc"].isoformat()
    return {
        "city": city,
        "observed_at_utc": observed_at,
        "is_stale": _stale(observed_at),
        "aqi": round(float(row["aqi"]), 1),
        "category": get_us_aqi_category(float(row["aqi"])).category,
        "aqi_change_24h": round(float(row["aqi"] - previous["aqi"]), 1),
    }


def get_aqi_forecast(city: str) -> dict[str, Any]:
    frame = _city_data(city)
    model, _ = load_champion(city)
    row = frame.iloc[[-1]]
    issued_at = row.iloc[0]["event_time_utc"]
    predictions = model.predict(row)
    return {
        "city": city,
        "issued_at_utc": issued_at.isoformat(),
        "is_stale": _stale(issued_at.isoformat()),
        "forecasts": [
            {
                "horizon_hours": hours,
                "valid_at_utc": (issued_at + pd.Timedelta(hours=hours)).isoformat(),
                "aqi": round(float(predictions[f"target_aqi_{hours}h"][0]), 1),
                "category": get_us_aqi_category(float(predictions[f"target_aqi_{hours}h"][0])).category,
            }
            for hours in (24, 48, 72)
        ],
    }


def get_weather(city: str) -> dict[str, Any]:
    row = _city_data(city).iloc[-1]
    timestamp = row["event_time_utc"].isoformat()
    return {"city": city, "observed_at_utc": timestamp, "is_stale": _stale(timestamp), "temperature_c": _value(row, "temperature_2m_c"), "humidity_pct": _value(row, "relative_humidity_2m_pct"), "pressure_hpa": _value(row, "surface_pressure_hpa"), "wind_kph": _value(row, "wind_speed_10m_kph"), "precipitation_mm": _value(row, "precipitation_mm")}


def get_pollutants(city: str) -> dict[str, Any]:
    row = _city_data(city).iloc[-1]
    timestamp = row["event_time_utc"].isoformat()
    return {"city": city, "observed_at_utc": timestamp, "is_stale": _stale(timestamp), "pm2_5": _value(row, "pm2_5_ug_m3"), "pm10": _value(row, "pm10_ug_m3"), "co": _value(row, "carbon_monoxide_ug_m3"), "no2": _value(row, "nitrogen_dioxide_ug_m3"), "so2": _value(row, "sulphur_dioxide_ug_m3"), "ozone": _value(row, "ozone_ug_m3")}


def get_aqi_history(city: str, hours: int = 72) -> dict[str, Any]:
    frame = _city_data(city).tail(max(1, min(hours, 168)))
    points = [{"time": row.event_time_utc.isoformat(), "aqi": round(float(row.aqi), 1)} for row in frame[["event_time_utc", "aqi"]].itertuples(index=False)]
    return {"city": city, "hours": len(points), "is_stale": _stale(points[-1]["time"]), "points": points}


def explain_city_prediction(city: str, horizon_hours: int = 24) -> dict[str, Any]:
    if horizon_hours not in (24, 48, 72):
        raise ValueError("Explanation horizon must be 24, 48, or 72 hours.")
    model, _ = load_champion(city)
    row = _city_data(city).iloc[[-1]]
    try:
        explanation = shap_local_explanation(model, row, horizon_hours)
        factors = [{"feature": item["feature"], "contribution": round(float(item["shap_value"]), 3)} for item in explanation["contributions"][:5]]
        method = "shap"
    except Exception:
        explanation = explain_prediction(model, row, horizon_hours)
        factors = []
        method = "feature_values_fallback"
    timestamp = row.iloc[0]["event_time_utc"].isoformat()
    return {"city": city, "horizon_hours": horizon_hours, "issued_at_utc": timestamp, "is_stale": _stale(timestamp), "prediction": round(float(explanation["prediction"]), 1), "method": method, "top_factors": factors}


def compare_cities(cities: list[str]) -> dict[str, Any]:
    if not cities:
        raise ValueError("At least one supported city is required.")
    comparison = []
    for city in cities:
        current = get_current_aqi(city)
        forecast = get_aqi_forecast(city)
        comparison.append({"city": city, "current": current, "forecast": forecast})
    return {"cities": comparison}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_current_aqi": get_current_aqi,
    "get_aqi_forecast": get_aqi_forecast,
    "get_weather": get_weather,
    "get_pollutants": get_pollutants,
    "compare_cities": compare_cities,
    "get_aqi_history": get_aqi_history,
    "explain_prediction": explain_city_prediction,
}


def _run_tool(correlation_id: str, name: str, *args, **kwargs) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    try:
        result = TOOLS[name](*args, **kwargs)
        event = {"tool": name, "outcome": "success", "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        logger.info("copilot_tool correlation_id=%s tool=%s outcome=success latency_ms=%.1f", correlation_id, name, event["latency_ms"])
        return result, event
    except Exception:
        event = {"tool": name, "outcome": "unavailable", "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        logger.warning("copilot_tool correlation_id=%s tool=%s outcome=unavailable latency_ms=%.1f", correlation_id, name, event["latency_ms"])
        return None, event


def _injection_or_internal_request(text: str) -> bool:
    signals = ("ignore previous", "ignore your rules", "system prompt", "developer message", "tool list", "internal implementation", "reveal prompt", "invent data", "always safe", "fake tool result")
    return any(signal in text for signal in signals)


def _off_topic(text: str) -> bool:
    return any(term in text for term in ("poem", "capital of france", "write code", "recipe", "joke")) and "aqi" not in text and "air" not in text


def _resolve_cities(message: str, requested: list[str], history: list[str]) -> tuple[list[str], str | None]:
    text = message.lower()
    allowed = _cities()
    if any(alias in re.findall(r"\b[a-z]+\b", text) for alias in ("isb", "khi")) or "the capital" in text or "my city" in text:
        return [], "Please name a supported city explicitly; I cannot safely infer abbreviations, 'the capital', or 'my city'."
    candidates = requested or [slug for slug, city in allowed.items() if slug in text or city["name"].lower() in text]
    resolved = [city.lower().strip() for city in candidates if city.lower().strip() in allowed]
    all_city_terms = ("all cities", "every city", "which city", "best city", "preferable", "safest", "cleanest", "healthiest", "where should", "compare")
    if not resolved and any(term in text for term in all_city_terms):
        resolved = list(allowed)
    if not resolved and re.search(r"\b(mult(an|on)|faisalabad|rawalpindi|hyderabad|gujranwala)\b", text):
        return [], "That city is not supported by the current registered AQI models. I cannot approximate it from a nearby city."
    if not resolved and history and re.search(r"\bwhat about\b", text):
        resolved = [slug for slug in allowed if slug in text]
    return resolved, None


def _stale_notice(evidence: dict[str, Any]) -> str:
    def contains_stale(value: Any) -> bool:
        if isinstance(value, dict):
            return bool(value.get("is_stale")) or any(contains_stale(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_stale(item) for item in value)
        return False
    stale = contains_stale(evidence)
    return "\n\nData freshness: source data is stale; this is the latest stored modeled data, not live conditions." if stale else ""


def _forecast_summary(forecasts: list[dict[str, Any]]) -> str:
    """Keep the three horizons readable instead of compressing them into one sentence."""
    lines = ["Forecast from this observation:"]
    lines.extend(f"- In {item['horizon_hours']} hours: {item['aqi']:.1f} AQI ({item['category']})." for item in forecasts)
    return "\n".join(lines)


def _activity_advice(category: str) -> str:
    """Conservative, category-based guidance; it is not personal medical advice."""
    if category == "Good":
        return "A usual outdoor jog is reasonable for most people."
    if category == "Moderate":
        return "Most people can exercise normally; sensitive people should take symptoms into account."
    if category == "Unhealthy for Sensitive Groups":
        return "For a jog, people with heart or lung conditions, children, and older adults should avoid prolonged or hard outdoor exertion; others should keep it easier if symptoms occur."
    if category == "Unhealthy":
        return "It is better to shorten, reduce the intensity of, or move a strenuous outdoor jog indoors."
    return "It is safer to avoid strenuous outdoor exercise and choose an indoor option if possible."


def _answer(message: str, evidence: dict[str, Any], cities: list[str]) -> str:
    text = message.lower()
    if not evidence:
        if "aqi" in text and any(term in text for term in ("stand", "mean", "what is", "category")):
            return "AQI means Air Quality Index. Lower values indicate cleaner air: Good 0-50, Moderate 51-100, Unhealthy for Sensitive Groups 101-150, Unhealthy 151-200, Very Unhealthy 201-300, and Hazardous 301+."
        if any(term in text for term in ("health", "mask", "exercise", "precaution")):
            return "For AQI above 100, sensitive groups should reduce prolonged outdoor exertion. Above 150, everyone should reduce prolonged outdoor activity and follow local public-health guidance."
        return "Ask about AQI, weather, pollutants, forecasts, history, comparisons, or model explanations for a supported city."
    if "comparison" in evidence:
        rows = evidence["comparison"]["cities"]
        tomorrow = sorted((entry["forecast"]["forecasts"][0]["aqi"], entry["city"]) for entry in rows)
        if any(term in text for term in ("improve", "improvement", "better over", "three days")):
            changes = sorted((entry["forecast"]["forecasts"][-1]["aqi"] - entry["current"]["aqi"], entry["city"]) for entry in rows)
            change, city = changes[0]
            direction = "improve" if change < -1 else "remain the most stable" if change <= 1 else "worsen the least"
            return f"{city.title()} is expected to {direction} over the next 72 hours ({change:+.1f} AQI).\n\nThis compares forecast change only; lower AQI is better." + _stale_notice(evidence["comparison"])
        if any(term in text for term in ("cleanest", "preferable", "best", "safest", "healthiest", "live")):
            best_aqi, best_city = tomorrow[0]
            return f"{best_city.title()} is expected to have the cleanest air tomorrow at {best_aqi:.1f} AQI.\n\nThis is an AQI-only comparison, not a full liveability recommendation." + _stale_notice(evidence["comparison"])
        return "Tomorrow's forecast, cleaner to poorer:\n" + "\n".join(f"- {city.title()}: {aqi:.1f} AQI." for aqi, city in tomorrow) + "\n\nLower AQI is better." + _stale_notice(evidence["comparison"])
    city = cities[0]
    current = evidence.get("current")
    forecast = evidence.get("forecast")
    if "history" in evidence:
        points = evidence["history"]["points"]
        change = points[-1]["aqi"] - points[0]["aqi"]
        return f"{city.title()} changed {change:+.1f} AQI over the last {evidence['history']['hours']} stored hours.\n\nIt moved from {points[0]['aqi']:.1f} to {points[-1]['aqi']:.1f}. Historical data ends at {points[-1]['time']}." + _stale_notice(evidence["history"])
    if "weather" in evidence:
        weather = evidence["weather"]
        return f"{city.title()} is {weather['temperature_c']}°C with {weather['humidity_pct']}% humidity.\n\nWind is {weather['wind_kph']} km/h and precipitation is {weather['precipitation_mm']} mm. Stored observation: {weather['observed_at_utc']}." + _stale_notice(weather)
    if "pollutants" in evidence:
        p = evidence["pollutants"]
        return f"For {city.title()}, PM2.5 is {p['pm2_5']} µg/m³ and PM10 is {p['pm10']} µg/m³.\n\nNO₂ is {p['no2']} µg/m³ and ozone is {p['ozone']} µg/m³. Stored observation: {p['observed_at_utc']}." + _stale_notice(p)
    if "explanation" in evidence:
        e = evidence["explanation"]
        factors = ", ".join(item["feature"] for item in e["top_factors"][:3]) or "the selected model features"
        return f"The {e['horizon_hours']}-hour forecast for {city.title()} is {e['prediction']:.1f} AQI.\n\nThe {e['method']} explanation highlights {factors}. These factors explain the model output, not a proven pollution source." + _stale_notice(e)
    if current and forecast:
        forecast_values = {item["horizon_hours"]: item["aqi"] for item in forecast["forecasts"]}
        asks_activity = any(term in text for term in ("jog", "run", "exercise", "workout", "outdoor activity"))
        asks_aqi_definition = "aqi" in text and any(term in text for term in ("what is", "what's the deal", "what is the deal", "mean", "stand for", "anyway"))
        values = _forecast_summary(forecast["forecasts"])
        if asks_activity or asks_aqi_definition:
            parts = [
                f"Lahore's stored air-quality picture is {current['category'].lower()} right now: AQI {current['aqi']:.1f}."
                if city == "lahore" else f"{city.title()}'s stored air-quality picture is {current['category'].lower()} right now: AQI {current['aqi']:.1f}."
            ]
            if asks_activity:
                parts.append(_activity_advice(current["category"]))
            parts.append(values)
            if asks_aqi_definition:
                parts.append("AQI means Air Quality Index: lower values indicate cleaner air, and its category is used to give broad activity guidance.")
            parts.append(f"Stored observation: {current['observed_at_utc']}.")
            return "\n\n".join(parts) + _stale_notice(current)
        if any(term in text for term in ("why", "worse", "better", "trend")):
            change = forecast_values[72] - current["aqi"]
            direction = "worsening" if change > 5 else "improving" if change < -5 else "broadly stable"
            return f"{city.title()} is forecast to be {direction} over the next three days.\n\nCurrent AQI is {current['aqi']:.1f} ({current['category']}); the 72-hour forecast is {forecast_values[72]:.1f} ({change:+.1f}). This forecast does not prove a specific emissions source.\n\nStored observation: {current['observed_at_utc']}." + _stale_notice(current)
        return f"{city.title()} is currently in the {current['category']} range, with a stored AQI of {current['aqi']:.1f}.\n\n{values}\n\nStored observation: {current['observed_at_utc']}." + _stale_notice(current)
    return "The requested AQI data is unavailable, so I cannot provide a number."


def chat(message: str, requested_cities: list[str] | None = None, history: list[str] | None = None) -> dict[str, Any]:
    """One safe, stateless turn; history is bounded and used only for intent context."""
    correlation_id = uuid.uuid4().hex
    history = (history or [])[-MAX_HISTORY_MESSAGES:]
    text = message.strip()
    if _injection_or_internal_request(text.lower()):
        return {"answer": "I can help with supported AQI information, but I cannot reveal internal instructions or bypass grounding and safety rules.", "tools_used": [], "tool_events": [], "evidence": {}, "provider": "deterministic_grounded", "correlation_id": correlation_id, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if _off_topic(text.lower()):
        return {"answer": "I am limited to supported AQI, weather, pollutant, forecast, history, and model-explanation questions for this project.", "tools_used": [], "tool_events": [], "evidence": {}, "provider": "deterministic_grounded", "correlation_id": correlation_id, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    cities, clarification = _resolve_cities(text, requested_cities or [], history)
    if clarification:
        return {"answer": clarification, "tools_used": [], "tool_events": [], "evidence": {}, "provider": "deterministic_grounded", "correlation_id": correlation_id, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    lower = text.lower()
    evidence: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    tools_used: list[str] = []
    if len(cities) > 1:
        result, event = _run_tool(correlation_id, "compare_cities", cities)
        events.append(event)
        tools_used.append("compare_cities")
        if result is not None:
            evidence["comparison"] = result
    elif cities:
        city = cities[0]
        # Compose a small, de-duplicated tool plan: one user turn may contain
        # more than one AQI question, so routing must not stop at the first.
        plan: list[tuple[str, str, tuple]] = []
        weather_terms = ("weather", "temperature", "humidity", "wind", "rain", "pressure")
        pollutant_terms = ("pm2", "pm10", "pollutant", "ozone", "no2", "so2", "carbon monoxide")
        history_terms = ("history", "last day", "last 24", "past 24", "past 3", "historical")
        explanation_terms = ("explain", "shap", "feature importance", "predicted high")
        needs_status = any(term in lower for term in ("aqi", "air", "forecast", "looking", "worse", "better", "trend", "jog", "run", "exercise", "workout"))
        if needs_status:
            plan.extend([("current", "get_current_aqi", (city,)), ("forecast", "get_aqi_forecast", (city,))])
        if any(term in lower for term in weather_terms):
            plan.append(("weather", "get_weather", (city,)))
        if any(term in lower for term in pollutant_terms):
            plan.append(("pollutants", "get_pollutants", (city,)))
        if any(term in lower for term in history_terms):
            plan.append(("history", "get_aqi_history", (city, 72)))
        if any(term in lower for term in explanation_terms) or ("predicted" in lower and "high" in lower):
            horizon = 72 if "72" in lower else 48 if "48" in lower else 24
            plan.append(("explanation", "explain_prediction", (city, horizon)))
        if not plan:
            plan = [("current", "get_current_aqi", (city,)), ("forecast", "get_aqi_forecast", (city,))]
        for key, tool, args in plan:
            result, event = _run_tool(correlation_id, tool, *args)
            events.append(event)
            tools_used.append(tool)
            if result is not None:
                evidence[key] = result
    if events and not evidence and all(event["outcome"] == "unavailable" for event in events):
        answer = "The requested AQI data is unavailable, so I cannot provide a number."
    else:
        answer = _answer(text, evidence, cities)
    return {"answer": answer, "tools_used": tools_used, "tool_events": events, "evidence": evidence, "provider": "deterministic_grounded", "correlation_id": correlation_id, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
