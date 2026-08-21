"""Data cleaning, alignment, gap filling, and metadata enrichment."""

from datetime import datetime, timezone

import pandas as pd

from pearls_aqi.domain.schemas import QualityFlag, SourceRecordType


def merge_and_clean_city_data(
    weather_df: pd.DataFrame,
    aq_df: pd.DataFrame,
    city_slug: str,
    latitude: float,
    longitude: float,
    source_record_type: SourceRecordType = SourceRecordType.REANALYSIS,
    max_impute_gap_hours: int = 3,
) -> pd.DataFrame:
    """Merge weather and air quality DataFrames, clean, fill short gaps, and add metadata."""
    if weather_df.empty and aq_df.empty:
        return pd.DataFrame()

    if not weather_df.empty and not aq_df.empty:
        merged = pd.merge(aq_df, weather_df, on="event_time_utc", how="outer")
    elif not aq_df.empty:
        merged = aq_df.copy()
    else:
        merged = weather_df.copy()

    merged["city_slug"] = city_slug
    merged["latitude"] = latitude
    merged["longitude"] = longitude
    merged["source_name"] = "Open-Meteo"
    merged["source_record_type"] = source_record_type.value
    merged["ingested_at_utc"] = datetime.now(timezone.utc)
    merged["aqi_standard"] = "us_aqi"
    merged["data_label"] = "modeled air-quality data"

    # Sort chronologically
    merged.sort_values(by="event_time_utc", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged["quality_flag"] = QualityFlag.VALID.value
    # Modeled pollutant series occasionally contain small negative numerical
    # artefacts. They are physically invalid, so preserve their provenance as
    # suspect and treat them as short gaps rather than passing bad values on.
    pollutant_cols = [c for c in ["pm2_5_ug_m3", "pm10_ug_m3", "carbon_monoxide_ug_m3", "nitrogen_dioxide_ug_m3", "sulphur_dioxide_ug_m3", "ozone_ug_m3"] if c in merged.columns]
    for col in pollutant_cols:
        invalid = merged[col] < 0
        merged.loc[invalid, col] = pd.NA
        merged.loc[invalid, "quality_flag"] = QualityFlag.SUSPECT.value

    # Impute short gaps in AQI and pollutants up to max_impute_gap_hours
    target_cols = [c for c in ["aqi", *pollutant_cols, "temperature_2m_c", "relative_humidity_2m_pct"] if c in merged.columns]
    for col in target_cols:
        was_null = merged[col].isna()
        merged[col] = merged[col].ffill(limit=max_impute_gap_hours)
        now_filled = was_null & merged[col].notna()
        if now_filled.any():
            merged.loc[now_filled & (merged["quality_flag"] == QualityFlag.VALID.value), "quality_flag"] = QualityFlag.IMPUTED.value

    return merged
