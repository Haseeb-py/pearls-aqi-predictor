"""Unit tests for Pydantic domain models and US AQI categories."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pearls_aqi.domain.aqi_categories import get_us_aqi_category
from pearls_aqi.domain.schemas import QualityFlag, RawObservation, SourceRecordType


def test_us_aqi_category_boundaries():
    assert get_us_aqi_category(25).category == "Good"
    assert get_us_aqi_category(50).category == "Good"
    assert get_us_aqi_category(51).category == "Moderate"
    assert get_us_aqi_category(100).category == "Moderate"
    assert get_us_aqi_category(101).category == "Unhealthy for Sensitive Groups"
    assert get_us_aqi_category(150).category == "Unhealthy for Sensitive Groups"
    assert get_us_aqi_category(151).category == "Unhealthy"
    assert get_us_aqi_category(200).category == "Unhealthy"
    assert get_us_aqi_category(201).category == "Very Unhealthy"
    assert get_us_aqi_category(300).category == "Very Unhealthy"
    assert get_us_aqi_category(301).category == "Hazardous"
    assert get_us_aqi_category(301).is_hazardous is True
    assert get_us_aqi_category(500).category == "Hazardous"


def test_raw_observation_schema_valid():
    obs = RawObservation(
        city_slug="lahore",
        event_time_utc=datetime.now(timezone.utc),
        ingested_at_utc=datetime.now(timezone.utc),
        source_name="Open-Meteo",
        source_record_type=SourceRecordType.REANALYSIS,
        latitude=31.5204,
        longitude=74.3587,
        aqi=120.0,
        quality_flag=QualityFlag.VALID,
    )
    assert obs.city_slug == "lahore"
    assert obs.aqi == 120.0
    assert obs.data_label == "modeled air-quality data"


def test_raw_observation_schema_invalid_aqi():
    with pytest.raises(ValidationError):
        RawObservation(
            city_slug="lahore",
            event_time_utc=datetime.now(timezone.utc),
            ingested_at_utc=datetime.now(timezone.utc),
            source_name="Open-Meteo",
            source_record_type=SourceRecordType.REANALYSIS,
            latitude=31.5204,
            longitude=74.3587,
            aqi=1500.0,  # invalid (> 1000)
        )
