import json

import joblib

from pearls_aqi.models import registry
from pearls_aqi.models.registry import _registry_metrics, load_champion, save_champion
from pearls_aqi.settings import settings


def test_local_model_registry_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    saved = save_champion("lahore", {"model": "test"}, "ridge_model", ["aqi"], {"mae": 1.2})
    model, metadata = load_champion("lahore")

    assert model == {"model": "test"}
    assert metadata["model_name"] == "ridge_model"
    assert saved["city_slug"] == "lahore"


def test_model_registry_uses_latest_hopsworks_artifact_when_local_missing(tmp_path, monkeypatch):
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    joblib.dump({"model": "remote"}, remote_dir / "champion.joblib")
    (remote_dir / "champion.json").write_text(json.dumps({"model_name": "remote"}), encoding="utf-8")

    class RemoteModel:
        version = 2

        def download(self):
            return str(remote_dir)

    class Project:
        class ModelRegistry:
            @staticmethod
            def get_models(name):
                return [RemoteModel()]

        @staticmethod
        def get_model_registry():
            return Project.ModelRegistry()

    monkeypatch.setattr(settings, "BASE_DIR", tmp_path / "empty")
    monkeypatch.setattr(registry, "get_hopsworks_project", lambda: Project())

    model, metadata = load_champion("lahore")

    assert model == {"model": "remote"}
    assert metadata["model_name"] == "remote"


def test_registry_metrics_are_flattened_to_numeric_values():
    metrics = _registry_metrics(
        {"per_horizon": {"target_aqi_24h": {"mae": 12.3, "r2": 0.4}}, "selection": {"alpha": 10}}
    )

    assert metrics == {
        "per_horizon_target_aqi_24h_mae": 12.3,
        "per_horizon_target_aqi_24h_r2": 0.4,
        "selection_alpha": 10.0,
    }
