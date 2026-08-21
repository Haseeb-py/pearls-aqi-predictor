from pearls_aqi.models.registry import load_champion, save_champion
from pearls_aqi.settings import settings


def test_local_model_registry_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    saved = save_champion("lahore", {"model": "test"}, "ridge_model", ["aqi"], {"mae": 1.2})
    model, metadata = load_champion("lahore")

    assert model == {"model": "test"}
    assert metadata["model_name"] == "ridge_model"
    assert saved["city_slug"] == "lahore"
