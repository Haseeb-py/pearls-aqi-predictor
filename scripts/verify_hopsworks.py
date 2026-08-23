"""Verify live Hopsworks Feature Store and Model Registry access without secrets."""

from pearls_aqi.features.store import get_hopsworks_project, get_or_create_features_fg, get_or_create_features_fv
from pearls_aqi.settings import settings


def main() -> None:
    project = get_hopsworks_project()
    feature_store = project.get_feature_store()
    feature_group = get_or_create_features_fg(feature_store)
    feature_view = get_or_create_features_fv(feature_store, feature_group)
    registry = project.get_model_registry()
    print(f"Feature group: {feature_group.name} v{feature_group.version}")
    print(f"Feature view: {feature_view.name} v{feature_view.version}")
    for city in settings.load_cities_config()["cities"]:
        if city.get("enabled", True):
            models = registry.get_models(f"{settings.MODEL_NAME}_{city['slug']}")
            latest = max((int(model.version) for model in models), default=None)
            print(f"{city['slug']}: registry version {latest if latest is not None else 'missing'}")


if __name__ == "__main__":
    main()
