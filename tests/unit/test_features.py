"""Unit tests for feature builder and target construction."""

import numpy as np
import pandas as pd

from pearls_aqi.features.builder import build_features
from pearls_aqi.features.targets import build_targets


def test_build_features_and_targets():
    # Construct 100 hours of synthetic data for lahore
    times = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "city_slug": ["lahore"] * 100,
            "event_time_utc": times,
            "aqi": np.linspace(100, 200, 100),
            "temperature_2m_c": [15.0] * 100,
            "relative_humidity_2m_pct": [50.0] * 100,
            "pm2_5_ug_m3": [45.0] * 100,
        }
    )

    df_feat = build_features(df)
    df_full = build_targets(df_feat)

    # Check time features
    assert "hour" in df_full.columns
    assert "hour_sin" in df_full.columns
    assert "hour_cos" in df_full.columns

    # Check lag features
    assert "aqi_lag_1h" in df_full.columns
    assert "aqi_lag_24h" in df_full.columns
    # Check exact value at index 24: lag_24h should equal aqi[0]
    assert np.isclose(df_full.loc[24, "aqi_lag_24h"], df.loc[0, "aqi"])

    # Check targets
    assert "target_aqi_24h" in df_full.columns
    assert "target_aqi_48h" in df_full.columns
    assert "target_aqi_72h" in df_full.columns

    # Check exact target alignment at index 0: target_aqi_24h should equal aqi[24]
    assert np.isclose(df_full.loc[0, "target_aqi_24h"], df.loc[24, "aqi"])
    assert np.isclose(df_full.loc[0, "target_aqi_48h"], df.loc[48, "aqi"])
    assert np.isclose(df_full.loc[0, "target_aqi_72h"], df.loc[72, "aqi"])
