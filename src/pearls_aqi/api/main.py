"""Minimal AQI prediction API."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pearls_aqi.copilot import chat
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

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
    """Prepare the dashboard's default city before accepting requests."""
    try:
        load_training_data("lahore")
    except Exception:
        pass


def _enabled_cities() -> list[dict]:
    return [
        city
        for city in settings.load_cities_config()["cities"]
        if city.get("enabled", True)
    ]


def _add_forecast_weather_features(
    latest: pd.DataFrame,
    city_config: dict,
) -> pd.DataFrame:
    """Add real Open-Meteo weather forecasts for target horizons."""
    response = requests.get(
        WEATHER_FORECAST_URL,
        params={
            "latitude": city_config["latitude"],
            "longitude": city_config["longitude"],
            "hourly": (
                "temperature_2m,relative_humidity_2m,surface_pressure,"
                "wind_speed_10m,precipitation"
            ),
            "forecast_days": 4,
            "timezone": "UTC",
        },
        timeout=12,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})

    if not hourly or "time" not in hourly:
        raise RuntimeError("Open-Meteo returned no hourly weather forecast.")

    weather = pd.DataFrame(hourly)
    weather["event_time_utc"] = pd.to_datetime(
        weather["time"],
        utc=True,
    )

    enriched = latest.copy()
    observed_at = pd.Timestamp(enriched.iloc[0]["event_time_utc"])

    feature_map = {
        "temperature_2m": "forecast_temperature_2m_c",
        "relative_humidity_2m": "forecast_relative_humidity_2m_pct",
        "surface_pressure": "forecast_surface_pressure_hpa",
        "wind_speed_10m": "forecast_wind_speed_10m_kph",
        "precipitation": "forecast_precipitation_mm",
    }

    for hours in (24, 48, 72):
        target_time = observed_at + pd.Timedelta(hours=hours)
        matching = weather.loc[weather["event_time_utc"] == target_time]

        if matching.empty:
            raise RuntimeError(
                f"Open-Meteo has no weather forecast for +{hours}h."
            )

        row = matching.iloc[0]
        for source_column, feature_prefix in feature_map.items():
            enriched[f"{feature_prefix}_{hours}h"] = float(row[source_column])

    return enriched


@app.get("/cities", response_model=list[CityConfig])
def cities() -> list[dict]:
    return _enabled_cities()


@app.get("/predict/{city}", response_model=ForecastResponse)
def predict(city: str) -> ForecastResponse:
    city_configs = {
        item["slug"]: item
        for item in _enabled_cities()
    }

    if city not in city_configs:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    try:
        model, metadata = load_champion(city)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model available for {city}.",
        ) from exc

    try:
        latest = load_training_data(city).iloc[[-1]]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction data unavailable for {city}.",
        ) from exc

    try:
        predictions = model.predict(latest)
    except KeyError as exc:
        if "forecast_" not in str(exc):
            raise HTTPException(
                status_code=503,
                detail=f"Prediction feature data unavailable for {city}.",
            ) from exc

        try:
            latest = _add_forecast_weather_features(latest, city_configs[city])
            predictions = model.predict(latest)
        except Exception as weather_exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Forecast-weather data unavailable for {city}; "
                    "please try again shortly."
                ),
            ) from weather_exc
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction feature data unavailable for {city}.",
        ) from exc

    observed_at = latest.iloc[0]["event_time_utc"].to_pydatetime()
    forecasts = []

    for hours in (24, 48, 72):
        value = float(predictions[f"target_aqi+{hours}h"][0])
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
        raise HTTPException(
            status_code=503,
            detail="Copilot evidence is temporarily unavailable.",
        ) from exc