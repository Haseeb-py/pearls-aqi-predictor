"""Target construction for 24h, 48h, 72h horizons and leakage assertions."""

import pandas as pd

from pearls_aqi.domain.exceptions import DataLeakageError, FeatureEngineeringError


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Construct target_aqi_24h, target_aqi_48h, target_aqi_72h for each row."""
    if df.empty:
        raise FeatureEngineeringError("Cannot build targets on empty DataFrame.")

    df_out = df.copy()
    df_out["event_time_utc"] = pd.to_datetime(df_out["event_time_utc"], utc=True)
    df_out.sort_values(by=["city_slug", "event_time_utc"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    df_out["target_aqi_24h"] = df_out.groupby("city_slug")["aqi"].shift(-24)
    df_out["target_aqi_48h"] = df_out.groupby("city_slug")["aqi"].shift(-48)
    df_out["target_aqi_72h"] = df_out.groupby("city_slug")["aqi"].shift(-72)

    return df_out


def assert_no_future_leakage(df: pd.DataFrame, cutoff_time: pd.Timestamp) -> bool:
    """Assert that all feature rows used for inference at cutoff_time have event_time_utc <= cutoff_time."""
    future_rows = df[df["event_time_utc"] > cutoff_time]
    if not future_rows.empty:
        raise DataLeakageError(
            f"Detected {len(future_rows)} rows with event_time_utc > cutoff_time ({cutoff_time})."
        )
    return True
