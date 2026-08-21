"""Contract tests for Open-Meteo adapters using respx HTTP mocking."""

import respx

from pearls_aqi.data.air_quality_provider import OpenMeteoAirQualityProvider
from pearls_aqi.data.weather_provider import OpenMeteoWeatherProvider


@respx.mock
def test_open_meteo_weather_contract():
    route = respx.get("https://archive-api.open-meteo.com/v1/archive").respond(
        status_code=200,
        json={
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "temperature_2m": [14.2, 13.8],
                "relative_humidity_2m": [65.0, 68.0],
                "surface_pressure": [1012.5, 1012.8],
                "wind_speed_10m": [8.5, 9.0],
                "precipitation": [0.0, 0.0],
            }
        },
    )

    provider = OpenMeteoWeatherProvider()
    df = provider.fetch_historical_weather(31.5204, 74.3587, "2025-01-01", "2025-01-01")

    assert route.called
    assert len(df) == 2
    assert "temperature_2m_c" in df.columns
    assert df["temperature_2m_c"].iloc[0] == 14.2


@respx.mock
def test_open_meteo_previous_run_weather_contract():
    route = respx.get("https://previous-runs-api.open-meteo.com/v1/forecast").respond(
        status_code=200,
        json={"hourly": {
            "time": ["2025-01-03T00:00"],
            "temperature_2m_previous_day2": [20.0],
            "relative_humidity_2m_previous_day2": [50.0],
            "surface_pressure_previous_day2": [1000.0],
            "wind_speed_10m_previous_day2": [10.0],
            "precipitation_previous_day2": [0.0],
        }},
    )
    df = OpenMeteoWeatherProvider().fetch_previous_run_weather(31.5, 74.3, "2025-01-03", "2025-01-03", 2)
    assert route.called
    assert df.loc[0, "forecast_temperature_2m_c_48h"] == 20.0


@respx.mock
def test_open_meteo_air_quality_contract():
    route = respx.get("https://air-quality-api.open-meteo.com/v1/air-quality").respond(
        status_code=200,
        json={
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "pm2_5": [45.2, 50.1],
                "pm10": [80.0, 85.0],
                "carbon_monoxide": [400.0, 420.0],
                "nitrogen_dioxide": [30.0, 32.0],
                "sulphur_dioxide": [10.0, 11.0],
                "ozone": [20.0, 22.0],
                "us_aqi": [124.0, 137.0],
            }
        },
    )

    provider = OpenMeteoAirQualityProvider()
    df = provider.fetch_historical_air_quality(31.5204, 74.3587, "2025-01-01", "2025-01-01")

    assert route.called
    assert len(df) == 2
    assert "aqi" in df.columns
    assert df["aqi"].iloc[0] == 124.0
    assert df["data_label"].iloc[0] == "modeled air-quality data"
