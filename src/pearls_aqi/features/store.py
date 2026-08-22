"""Hopsworks Feature Store integration module."""

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple
import pandas as pd
from pearls_aqi.domain.exceptions import FeatureStoreError
from pearls_aqi.features.builder import build_features
from pearls_aqi.features.targets import build_targets
from pearls_aqi.settings import settings


@lru_cache(maxsize=24)
def _cached_local_training_data(city_slug: Optional[str], source_key: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    """Load and engineer immutable local data once per CSV version and city."""
    df = pd.concat(
        [pd.read_csv(path, parse_dates=["event_time_utc"]) for path, _ in source_key],
        ignore_index=True,
    )
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
    if city_slug:
        df = df.loc[df["city_slug"] == city_slug].copy()
    if df.empty:
        raise FeatureStoreError("No training rows found for the requested city.")
    df = df.sort_values(["city_slug", "event_time_utc"]).drop_duplicates(
        ["city_slug", "event_time_utc"], keep="last"
    )
    required_engineered = {"aqi_lag_72h", "aqi_rolling_mean_24h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"}
    if required_engineered.issubset(df.columns):
        return df.reset_index(drop=True)
    return build_targets(build_features(df)).reset_index(drop=True)


def get_hopsworks_project() -> Any:
    """Login to Hopsworks using environment variables without printing secrets."""
    if not settings.HOPSWORKS_API_KEY:
        raise FeatureStoreError(
            "HOPSWORKS_API_KEY is not set in environment or .env file."
        )
    if not settings.HOPSWORKS_PROJECT:
        raise FeatureStoreError(
            "HOPSWORKS_PROJECT is not set in environment or .env file."
        )

    tmp_dir = tempfile.gettempdir()
    os.environ.setdefault("HOPSWORKS_TMP", tmp_dir)
    os.environ.setdefault("HOME", tmp_dir)

    try:
        import hopsworks

        project = hopsworks.login(
            host=settings.HOPSWORKS_HOST,
            project=settings.HOPSWORKS_PROJECT,
            api_key_value=settings.HOPSWORKS_API_KEY,
        )
        return project
    except Exception as exc:
        raise FeatureStoreError(
            f"Failed to connect to Hopsworks Feature Store: {exc}"
        ) from exc


def verify_hopsworks_connection() -> bool:
    """Smoke test Hopsworks connection and return True if successful."""
    project = get_hopsworks_project()
    if project is None:
        return False
    fs = project.get_feature_store()
    return fs is not None


def get_or_create_observations_fg(fs: Any) -> Any:
    """Get or create raw/clean observations feature group in Hopsworks."""
    try:
        return fs.get_or_create_feature_group(
            name="aqi_observations_v1",
            version=1,
            description="Normalized weather and air quality observations by city and time",
            primary_key=["city_slug", "event_time_utc", "source_name"],
            event_time="event_time_utc",
            online_enabled=False,
            time_travel_format="HUDI",
        )
    except Exception as exc:
        raise FeatureStoreError(f"Failed to create/get observations feature group: {exc}") from exc


def get_or_create_features_fg(fs: Any) -> Any:
    """Get or create engineered hourly features feature group in Hopsworks."""
    try:
        return fs.get_or_create_feature_group(
            name="aqi_features_hourly_v2",
            version=1,
            description="Model-ready point-in-time engineered AQI and weather features",
            primary_key=["city_slug", "event_time_utc"],
            event_time="event_time_utc",
            online_enabled=False,
            time_travel_format="HUDI",
        )
    except Exception as exc:
        raise FeatureStoreError(f"Failed to create/get features feature group: {exc}") from exc


def get_or_create_features_fv(fs: Any, fg: Optional[Any] = None) -> Any:
    """Get or create feature view over hourly engineered features in Hopsworks."""
    try:
        existing = fs.get_feature_view(name="aqi_features_fv_v2", version=1)
        if existing is not None:
            return existing
    except Exception:
        pass

    if fg is None:
        fg = get_or_create_features_fg(fs)

    try:
        query = fg.select_all()
        fs.create_feature_view(
            name="aqi_features_fv_v2",
            version=1,
            description="Feature view for training and inference datasets",
            query=query,
        )
        return fs.get_feature_view(name="aqi_features_fv_v2", version=1)
    except Exception:
        try:
            return fs.get_feature_view(name="aqi_features_fv_v2", version=1)
        except Exception as exc:
            raise FeatureStoreError(f"Failed to create or get feature view: {exc}") from exc


def upsert_features_df(df: pd.DataFrame) -> Tuple[int, str]:
    """Upsert engineered feature DataFrame into Hopsworks Feature Store."""
    if df.empty:
        return 0, "DataFrame is empty"

    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = get_or_create_features_fg(fs)

    df_upload = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_upload["event_time_utc"]):
        df_upload["event_time_utc"] = pd.to_datetime(df_upload["event_time_utc"], utc=True)
    # Keep these types aligned with the already-registered Hopsworks schema.
    # ``ingested_at_utc`` is metadata rather than the feature group's event time,
    # so it is intentionally stored there as an ISO-8601 string.
    if "ingested_at_utc" in df_upload:
        df_upload["ingested_at_utc"] = pd.to_datetime(
            df_upload["ingested_at_utc"], utc=True
        ).astype(str)
    if "is_weekend" in df_upload:
        df_upload["is_weekend"] = df_upload["is_weekend"].astype("int32")
    for column in df_upload.select_dtypes(include=["number"]).columns:
        if column not in {"hour", "day", "month", "day_of_week", "is_weekend"}:
            df_upload[column] = df_upload[column].astype(float)

    fg.insert(df_upload, overwrite=False)
    get_or_create_features_fv(fs, fg=fg)

    return len(df_upload), "Successfully upserted into aqi_features_hourly_v2"


def load_training_data(
    city_slug: Optional[str] = None,
    local_path: Optional[Path] = None,
    use_feature_store: bool = False,
) -> pd.DataFrame:
    """Load feature-view data, falling back to a local backfill CSV for development."""
    if use_feature_store:
        candidates = []
    elif local_path is not None:
        candidates = [Path(local_path)]
    else:
        artifacts = list((settings.BASE_DIR / "artifacts" / "data").glob("backfill_*.csv"))
        # Production backfills are cumulative. Prefer the complete artifact,
        # but retain multi-file assembly for small/dev artifacts and tests.
        largest = max(artifacts, key=lambda path: path.stat().st_size) if artifacts else None
        candidates = [largest] if largest and largest.stat().st_size > 1_000_000 else artifacts

    if candidates:
        source_key = tuple((str(path.resolve()), path.stat().st_mtime_ns) for path in candidates)
        return _cached_local_training_data(city_slug, source_key).copy()
    else:
        try:
            project = get_hopsworks_project()
            fv = get_or_create_features_fv(project.get_feature_store())
            df = fv.get_batch_data()
        except Exception as exc:
            raise FeatureStoreError("No local training artifact and Feature Store read failed.") from exc

    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
    if city_slug:
        df = df.loc[df["city_slug"] == city_slug].copy()
    if df.empty:
        raise FeatureStoreError("No training rows found for the requested city.")

    # Feature Store rows can originate from adjacent backfill runs. Rebuild
    # temporal features after all rows for a city are assembled so a lag does
    # not reset at an ingestion/batch boundary.
    df = df.sort_values(["city_slug", "event_time_utc"]).drop_duplicates(
        ["city_slug", "event_time_utc"], keep="last"
    )
    required_engineered = {"aqi_lag_72h", "aqi_rolling_mean_24h", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"}
    if required_engineered.issubset(df.columns):
        return df.reset_index(drop=True)
    return build_targets(build_features(df)).reset_index(drop=True)
