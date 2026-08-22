import pandas as pd

from pearls_aqi.features import store
from pearls_aqi.features.store import load_training_data


def test_load_training_data_from_local_artifact(tmp_path):
    artifact = tmp_path / "backfill.csv"
    pd.DataFrame(
        {
            "city_slug": ["lahore", "karachi"],
            "event_time_utc": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "aqi": [100, 80],
        }
    ).to_csv(artifact, index=False)

    result = load_training_data("lahore", artifact)

    assert len(result) == 1
    assert result.iloc[0]["city_slug"] == "lahore"


def test_loader_rebuilds_lags_across_persisted_batch_boundaries(tmp_path):
    artifact = tmp_path / "backfill.csv"
    times = pd.date_range("2025-06-01", periods=720, freq="h", tz="UTC")
    # Simulate a persisted source whose old batch-level feature columns were
    # all null. The raw AQI series remains intact and must drive reconstruction.
    pd.DataFrame(
        {
            "city_slug": "lahore",
            "event_time_utc": times,
            "aqi": range(720),
            "aqi_lag_24h": float("nan"),
            "aqi_lag_48h": float("nan"),
            "aqi_lag_72h": float("nan"),
        }
    ).to_csv(artifact, index=False)

    result = load_training_data("lahore", artifact)

    assert result["aqi_lag_24h"].notna().sum() == 696
    assert result["aqi_lag_48h"].notna().sum() == 672
    assert result["aqi_lag_72h"].notna().sum() == 648
    assert result.loc[72, "aqi_lag_72h"] == 0


def test_loader_combines_local_artifacts_before_city_filtering(tmp_path, monkeypatch):
    data_dir = tmp_path / "artifacts" / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {"city_slug": "lahore", "event_time_utc": ["2025-01-01T00:00:00Z"], "aqi": [100]}
    ).to_csv(data_dir / "backfill_lahore.csv", index=False)
    pd.DataFrame(
        {"city_slug": "karachi", "event_time_utc": ["2025-01-01T00:00:00Z"], "aqi": [80]}
    ).to_csv(data_dir / "backfill_karachi.csv", index=False)
    from pearls_aqi.settings import settings
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)

    result = load_training_data("lahore")

    assert result["city_slug"].tolist() == ["lahore"]


def test_hopsworks_upload_uses_registered_metadata_types(monkeypatch):
    class FeatureGroup:
        uploaded = None

        def insert(self, dataframe, overwrite=False):
            self.uploaded = dataframe

    class Project:
        def get_feature_store(self):
            return object()

    feature_group = FeatureGroup()
    monkeypatch.setattr(store, "get_hopsworks_project", lambda: Project())
    monkeypatch.setattr(store, "get_or_create_features_fg", lambda fs: feature_group)
    monkeypatch.setattr(store, "get_or_create_features_fv", lambda fs, fg: object())
    frame = pd.DataFrame(
        {
            "city_slug": ["lahore"],
            "event_time_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "ingested_at_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "is_weekend": [0],
            "aqi": [100],
        }
    )

    store.upsert_features_df(frame)

    assert feature_group.uploaded["ingested_at_utc"].dtype == object
    assert feature_group.uploaded["is_weekend"].dtype == "int32"
