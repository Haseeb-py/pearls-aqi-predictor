"""Unit tests for observation data validation."""

import pandas as pd
import pytest

from pearls_aqi.data.validation import validate_observation_df
from pearls_aqi.domain.exceptions import DataValidationError


def test_validate_observation_df_valid():
    df = pd.DataFrame(
        {
            "city_slug": ["lahore", "lahore"],
            "event_time_utc": pd.to_datetime(
                ["2025-01-01 00:00:00+00:00", "2025-01-01 01:00:00+00:00"]
            ),
            "temperature_2m_c": [15.0, 16.0],
            "relative_humidity_2m_pct": [60.0, 65.0],
            "aqi": [120.0, 130.0],
        }
    )
    validated = validate_observation_df(df)
    assert len(validated) == 2


def test_validate_observation_df_duplicates():
    df = pd.DataFrame(
        {
            "city_slug": ["lahore", "lahore"],
            "event_time_utc": pd.to_datetime(
                ["2025-01-01 00:00:00+00:00", "2025-01-01 00:00:00+00:00"]
            ),
            "aqi": [120.0, 130.0],
        }
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_observation_df(df)


def test_validate_observation_df_out_of_bounds():
    df = pd.DataFrame(
        {
            "city_slug": ["lahore"],
            "event_time_utc": pd.to_datetime(["2025-01-01 00:00:00+00:00"]),
            "relative_humidity_2m_pct": [150.0],  # Invalid (> 100)
            "aqi": [120.0],
        }
    )
    with pytest.raises(DataValidationError, match="outside bounds"):
        validate_observation_df(df)
