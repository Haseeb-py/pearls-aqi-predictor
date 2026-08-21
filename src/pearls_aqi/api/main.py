"""Minimal AQI prediction API."""

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException

from pearls_aqi.domain.aqi_categories import get_us_aqi_category
from pearls_aqi.copilot import chat
from pearls_aqi.domain.schemas import CityConfig, CopilotChatRequest, CopilotChatResponse, ForecastHorizonOutput, ForecastResponse
from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.registry import load_champion
from pearls_aqi.settings import settings

app = FastAPI(title="Pearls AQI Predictor")


@app.on_event("startup")
def warm_default_city_cache() -> None:
    """Prepare the dashboard's default city before accepting requests."""
    try:
        load_training_data("lahore")
    except Exception:
        # Startup must remain available even when local data is temporarily absent.
        pass


def _enabled_cities() -> list[dict]:
    return [city for city in settings.load_cities_config()["cities"] if city.get("enabled", True)]


@app.get("/cities", response_model=list[CityConfig])
def cities() -> list[dict]:
    return _enabled_cities()


@app.get("/predict/{city}", response_model=ForecastResponse)
def predict(city: str) -> ForecastResponse:
    if city not in {item["slug"] for item in _enabled_cities()}:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")
    try:
        model, metadata = load_champion(city)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No trained model available for {city}.") from exc
    try:
        latest = load_training_data(city).iloc[[-1]]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction data unavailable for {city}.") from exc

    predictions = model.predict(latest)
    observed_at = latest.iloc[0]["event_time_utc"].to_pydatetime()
    horizons = (24, 48, 72)
    forecasts = []
    for hours in horizons:
        value = float(predictions[f"target_aqi_{hours}h"][0])
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
        raise HTTPException(status_code=422, detail="Message must contain 1-1000 characters.")
    try:
        return chat(request.message, request.cities, request.history)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Copilot evidence is temporarily unavailable.") from exc
