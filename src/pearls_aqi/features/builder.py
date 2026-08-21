"""Feature engineering with strict temporal ordering and zero leakage."""

import numpy as np
import pandas as pd

from pearls_aqi.domain.exceptions import FeatureEngineeringError


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build engineered feature set for each city in df.
    
    Expects df to contain: city_slug, event_time_utc, aqi, pollutants, and weather variables.
    """
    if df.empty:
        raise FeatureEngineeringError("Cannot build features on empty DataFrame.")

    df_out = df.copy()
    df_out["event_time_utc"] = pd.to_datetime(df_out["event_time_utc"], utc=True)

    # Sort chronologically per city to guarantee order
    df_out.sort_values(by=["city_slug", "event_time_utc"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    # 1. Calendar / Time Features
    df_out["hour"] = df_out["event_time_utc"].dt.hour
    df_out["day"] = df_out["event_time_utc"].dt.day
    df_out["month"] = df_out["event_time_utc"].dt.month
    df_out["day_of_week"] = df_out["event_time_utc"].dt.dayofweek
    df_out["is_weekend"] = (df_out["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12.0)

    # 2. Lags per city
    lag_hours = [1, 3, 6, 12, 24, 48, 72]
    for lag in lag_hours:
        df_out[f"aqi_lag_{lag}h"] = df_out.groupby("city_slug")["aqi"].shift(lag)

    # 3. Change rates (1h, 3h, 24h)
    df_out["aqi_change_rate_1h"] = df_out["aqi"] - df_out["aqi_lag_1h"]
    df_out["aqi_change_rate_3h"] = df_out["aqi"] - df_out["aqi_lag_3h"]
    df_out["aqi_change_rate_24h"] = df_out["aqi"] - df_out["aqi_lag_24h"]

    epsilon = 1e-5
    df_out["aqi_pct_change_1h"] = (df_out["aqi"] - df_out["aqi_lag_1h"]) / (df_out["aqi_lag_1h"] + epsilon)
    df_out["aqi_pct_change_3h"] = (df_out["aqi"] - df_out["aqi_lag_3h"]) / (df_out["aqi_lag_3h"] + epsilon)
    df_out["aqi_pct_change_24h"] = (df_out["aqi"] - df_out["aqi_lag_24h"]) / (df_out["aqi_lag_24h"] + epsilon)

    # 4. Rolling statistics per city (shift by 1 to exclude current row from historical window)
    window_hours = [3, 6, 12, 24]
    for w in window_hours:
        shifted_aqi = df_out.groupby("city_slug")["aqi"].shift(1)
        df_out[f"aqi_rolling_mean_{w}h"] = shifted_aqi.groupby(df_out["city_slug"]).transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )
        df_out[f"aqi_rolling_std_{w}h"] = shifted_aqi.groupby(df_out["city_slug"]).transform(
            lambda s: s.rolling(window=w, min_periods=1).std()
        ).fillna(0.0)

    return df_out
