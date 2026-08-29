"""Minimal AQI prediction API."""

from datetime import datetime, timedelta, timezone
import logging

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pearls_aqi.copilot import chat, warm_city_cache
from pearls_aqi.domain.aqi_categories import get_us_aqi_category
from pearls_aqi.domain.schemas import (
    CityConfig,
    CopilotChatRequest,
    CopilotChatResponse,
    ForecastHorizonOutput,
    ForecastResponse,
)
from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.registry import load_champion
from pearls_aqi.settings import settings

logger = logging.getLogger(__name__)

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]


app = FastAPI(title="Pearls AQI Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.ALLOWED_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_default_city_cache() -> None:
    """Warm Lahore feature cache without making startup depend on it."""
    try:
        warm_city_cache("lahore")
    except Exception:
        logger.exception("Default city cache warmup failed; continuing startup.")


def _enabled_cities() -> list[dict]:
    return [
        city
        for city in settings.load_cities_config()["cities"]
        if city.get("enabled", True)
    ]


def _city_config(city_slug: str) -> dict:
    for city in _enabled_cities():
        if city["slug"] == city_slug:
            return city
    raise HTTPException(status_code=404, detail=f"Unknown city: {city_slug}")


def _fetch_forecast_weather(city: dict, observed_at: datetime) -> dict[str, float]:
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": ",".join(FORECAST_WEATHER_VARIABLES),
        "forecast_days": 4,
        "timezone": "UTC",
    }

    response = requests.get(WEATHER_FORECAST_URL, params=params, timeout=12)
    response.raise_for_status()

    hourly = response.json().get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True)

    if times.empty:
        raise ValueError("Open-Meteo forecast response did not contain hourly timestamps.")

    features: dict[str, float] = {}

    for horizon in (24, 48, 72):
        target_time = observed_at + timedelta(hours=horizon)
        nearest_idx = int(abs(times - target_time).argmin())

        features[f"forecast_temperature_2m_c_{horizon}h"] = float(
            hourly["temperature_2m"][nearest_idx]
        )
        features[f"forecast_relative_humidity_2m_pct_{horizon}h"] = float(
            hourly["relative_humidity_2m"][nearest_idx]
        )
        features[f"forecast_surface_pressure_hpa_{horizon}h"] = float(
            hourly["surface_pressure"][nearest_idx]
        )
        features[f"forecast_wind_speed_10m_kph_{horizon}h"] = float(
            hourly["wind_speed_10m"][nearest_idx]
        )
        features[f"forecast_precipitation_mm_{horizon}h"] = float(
            hourly["precipitation"][nearest_idx]
        )

    return features


def _add_forecast_weather_features(
    latest: pd.DataFrame,
    city: dict,
    observed_at: datetime,
) -> pd.DataFrame:
    enriched = latest.copy()
    forecast_features = _fetch_forecast_weather(city, observed_at)

    for column, value in forecast_features.items():
        enriched[column] = value

    return enriched


def _prediction_value(
    predictions: dict,
    hours: int,
    city: str,
) -> float:
    possible_targets = (
        f"target_aqi+{hours}h",
        f"target_aqi_{hours}h",
    )
    target = next((name for name in possible_targets if name in predictions), None)

    if target is None:
        logger.error(
            "Prediction output missing horizon=%sh for city=%s. Available keys=%s",
            hours,
            city,
            list(predictions.keys()),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Prediction output missing {hours}h forecast for {city}.",
        )

    return float(predictions[target][0])


@app.get("/cities", response_model=list[CityConfig])
def cities() -> list[dict]:
    return _enabled_cities()


@app.get("/predict/{city}", response_model=ForecastResponse)
def predict(city: str) -> ForecastResponse:
    city = city.lower().strip()
    city_cfg = _city_config(city)

    try:
        model, metadata = load_champion(city)
    except FileNotFoundError as exc:
        logger.exception("No trained model available for city=%s", city)
        raise HTTPException(
            status_code=404,
            detail=f"No trained model available for {city}.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to load champion model for city=%s", city)
        raise HTTPException(
            status_code=503,
            detail=f"Model registry unavailable for {city}.",
        ) from exc

    try:
        latest = load_training_data(city).sort_values("event_time_utc").iloc[[-1]]
    except Exception as exc:
        logger.exception("Failed to load prediction data for city=%s", city)
        raise HTTPException(
            status_code=503,
            detail=f"Prediction data unavailable for {city}.",
        ) from exc

    observed_at = latest.iloc[0]["event_time_utc"].to_pydatetime()
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)

    try:
        latest = _add_forecast_weather_features(latest, city_cfg, observed_at)
    except Exception as exc:
        logger.exception("Failed to load forecast-weather data for city=%s", city)
        raise HTTPException(
            status_code=503,
            detail=f"Forecast-weather data unavailable for {city}; please try again shortly.",
        ) from exc

    try:
        predictions = model.predict(latest)
    except KeyError as exc:
        logger.exception("Prediction feature mismatch for city=%s", city)
        raise HTTPException(
            status_code=503,
            detail=f"Prediction feature mismatch for {city}.",
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed for city=%s", city)
        raise HTTPException(
            status_code=503,
            detail=f"Prediction failed for {city}.",
        ) from exc

    forecasts = []
    for hours in (24, 48, 72):
        value = _prediction_value(predictions, hours, city)
        category = get_us_aqi_category(value)

        forecasts.append(
            ForecastHorizonOutput(
                horizon_hours=hours,
                valid_at_utc=observed_at + timedelta(hours=hours),
                aqi=round(value, 1),
                category=category.category,
                is_hazardous=category.is_hazardous,
            )
        )

    return ForecastResponse(
        city=city,
        issued_at_utc=observed_at,
        latest_observation_at_utc=observed_at,
        current_aqi=float(latest.iloc[0]["aqi"]),
        aqi_standard=str(latest.iloc[0].get("aqi_standard", "us_aqi")),
        data_label=str(latest.iloc[0].get("data_label", "modeled air-quality data")),
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
        forecasts=forecasts,
        is_stale=datetime.now(timezone.utc) - observed_at > timedelta(hours=30),
    )


@app.post("/api/v1/copilot/chat", response_model=CopilotChatResponse)
def copilot_chat(request: CopilotChatRequest) -> dict:
    if not request.message.strip() or len(request.message) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Message must contain 1-1000 characters.",
        )

    try:
        return chat(request.message, request.cities, request.history)
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Copilot evidence unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Copilot evidence is temporarily unavailable.",
        ) from exc