"""Local-first model registry with an optional Hopsworks upload."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

from pearls_aqi.features.store import get_hopsworks_project
from pearls_aqi.settings import settings


def _artifact_dir(city_slug: str) -> Path:
    return settings.BASE_DIR / "artifacts" / "models" / city_slug


def save_champion(
    city_slug: str,
    model: Any,
    model_name: str,
    feature_columns: Any,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a local champion artifact and its serving metadata."""
    artifact_dir = _artifact_dir(city_slug)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "champion.joblib"
    metadata_path = artifact_dir / "champion.json"
    joblib.dump(model, model_path)
    metadata = {
        "city_slug": city_slug,
        "model_name": model_name,
        "model_version": 1,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metadata_path": str(metadata_path), **metadata}


def load_champion(city_slug: str) -> Tuple[Any, Dict[str, Any]]:
    """Load the locally registered champion for inference."""
    artifact_dir = _artifact_dir(city_slug)
    model_path = artifact_dir / "champion.joblib"
    metadata_path = artifact_dir / "champion.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"No local champion registered for {city_slug}.")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def upload_champion_to_hopsworks(city_slug: str) -> bool:
    """Best-effort upload of the local artifact; local serving never depends on this."""
    try:
        project = get_hopsworks_project()
        registry = project.get_model_registry()
        artifact_dir = _artifact_dir(city_slug)
        metadata = json.loads((artifact_dir / "champion.json").read_text(encoding="utf-8"))
        remote_model = registry.python.create_model(
            name=f"{settings.MODEL_NAME}_{city_slug}",
            metrics=metadata["metrics"],
            description="AQI forecasting champion; local artifact is source of truth.",
        )
        remote_model.save(str(artifact_dir), keep_original_files=True)
        return True
    except Exception:
        return False
