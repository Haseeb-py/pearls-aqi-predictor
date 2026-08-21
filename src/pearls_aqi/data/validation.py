"""Validation rules for raw and cleaned observation records."""

from typing import Dict, Tuple

import pandas as pd

from pearls_aqi.domain.exceptions import DataValidationError

# Plausible range definitions
RANGE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature_2m_c": (-50.0, 60.0),
    "relative_humidity_2m_pct": (0.0, 100.0),
    "surface_pressure_hpa": (500.0, 1100.0),
    "wind_speed_10m_kph": (0.0, 200.0),
    "precipitation_mm": (0.0, 500.0),
    "pm2_5_ug_m3": (0.0, 5000.0),
    "pm10_ug_m3": (0.0, 5000.0),
    "carbon_monoxide_ug_m3": (0.0, 50000.0),
    "nitrogen_dioxide_ug_m3": (0.0, 5000.0),
    "sulphur_dioxide_ug_m3": (0.0, 5000.0),
    "ozone_ug_m3": (0.0, 5000.0),
    "aqi": (0.0, 1000.0),
}


def validate_observation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame against required schema, duplicate keys, and range bounds."""
    if df.empty:
        raise DataValidationError("Input DataFrame is empty.")

    required_cols = ["city_slug", "event_time_utc", "aqi"]
    for col in required_cols:
        if col not in df.columns:
            raise DataValidationError(f"Missing required column: {col}")

    # Ensure event_time_utc is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["event_time_utc"]):
        raise DataValidationError("event_time_utc must be a datetime column.")

    # Check for duplicates on (city_slug, event_time_utc)
    dups = df.duplicated(subset=["city_slug", "event_time_utc"])
    if dups.any():
        num_dups = dups.sum()
        raise DataValidationError(f"Found {num_dups} duplicate entity-time keys.")

    # Check plausible range bounds
    for col, (min_val, max_val) in RANGE_BOUNDS.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            if invalid_mask.any():
                num_invalid = invalid_mask.sum()
                raise DataValidationError(
                    f"Column '{col}' has {num_invalid} values outside bounds [{min_val}, {max_val}]."
                )

    return df
