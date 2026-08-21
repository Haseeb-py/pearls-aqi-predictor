"""Integration test for Lahore vertical slice (Gate 0 & Milestone 1)."""

import pytest

from pearls_aqi.data.air_quality_provider import OpenMeteoAirQualityProvider
from pearls_aqi.data.cleaning import merge_and_clean_city_data
from pearls_aqi.data.validation import validate_observation_df
from pearls_aqi.data.weather_provider import OpenMeteoWeatherProvider
from pearls_aqi.models.train import train_and_evaluate_lahore_slice


@pytest.mark.integration
def test_lahore_vertical_slice_end_to_end():
    # 1. Fetch Lahore data for Jan 1 - Jan 7 (1 week slice for fast integration test)
    weather_provider = OpenMeteoWeatherProvider()
    aq_provider = OpenMeteoAirQualityProvider()

    lat, lon = 31.5204, 74.3587
    start_date, end_date = "2025-01-01", "2025-01-14"

    w_df = weather_provider.fetch_historical_weather(lat, lon, start_date, end_date)
    a_df = aq_provider.fetch_historical_air_quality(lat, lon, start_date, end_date)

    assert not w_df.empty, "Historical weather data should not be empty"
    assert not a_df.empty, "Historical air quality data should not be empty"

    # 2. Merge, Clean, and Validate
    merged_df = merge_and_clean_city_data(w_df, a_df, "lahore", lat, lon)
    validated_df = validate_observation_df(merged_df)
    assert len(validated_df) >= 300, "Should have 2 weeks of hourly data"

    # 3. Train & Evaluate Lahore slice
    results = train_and_evaluate_lahore_slice(validated_df)

    assert "persistence_baseline" in results
    assert "seasonal_naive_baseline" in results
    assert "ridge_model" in results

    # Verify metrics exist for target_aqi_24h
    ridge_24h = results["ridge_model"]["target_aqi_24h"]
    assert "mae" in ridge_24h
    assert "rmse" in ridge_24h
    assert "r2" in ridge_24h
    assert not pytest.approx(ridge_24h["mae"]) == 0.0
