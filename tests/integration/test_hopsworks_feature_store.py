"""Integration test verifying Hopsworks Feature Group and Feature View management."""

import pytest
import pandas as pd
from pearls_aqi.features.builder import build_features
from pearls_aqi.features.targets import build_targets
from pearls_aqi.features.store import (
    get_hopsworks_project,
    get_or_create_features_fg,
    get_or_create_features_fv,
    upsert_features_df,
)
from pearls_aqi.settings import settings


@pytest.mark.integration
def test_hopsworks_feature_group_and_view():
    if not settings.HOPSWORKS_API_KEY or not settings.HOPSWORKS_PROJECT:
        pytest.skip("Hopsworks credentials not set.")

    project = get_hopsworks_project()
    fs = project.get_feature_store()

    fg = get_or_create_features_fg(fs)
    assert fg is not None
    assert fg.name == "aqi_features_hourly_v2"

    # Create 1 sample feature row to register schema
    times = pd.date_range("2025-01-01 00:00:00", periods=100, freq="h", tz="UTC")
    raw_df = pd.DataFrame(
        {
            "city_slug": ["lahore"] * 100,
            "event_time_utc": times,
            "aqi": [150.0] * 100,
            "temperature_2m_c": [15.0] * 100,
            "relative_humidity_2m_pct": [60.0] * 100,
            "pm2_5_ug_m3": [50.0] * 100,
        }
    )
    feat_df = build_features(raw_df)
    full_df = build_targets(feat_df)

    count, msg = upsert_features_df(full_df)
    assert count == 100

    fv = get_or_create_features_fv(fs, fg=fg)
    assert fv is not None
    assert fv.name == "aqi_features_fv_v2"
