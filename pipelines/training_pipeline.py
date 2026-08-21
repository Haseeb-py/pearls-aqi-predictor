"""Train and select an AQI forecasting champion from Feature Store data."""

import argparse
import json

from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.train import select_champion, train_and_evaluate
from pearls_aqi.models.registry import save_champion, upload_champion_to_hopsworks
from pearls_aqi.models.evaluate import calculate_metrics
from pearls_aqi.models.per_horizon import PerHorizonChampion, SelectedPerHorizonChampion
from pearls_aqi.models.sklearn_models import MultiHorizonRandomForestModel, MultiHorizonRidgeModel
from pearls_aqi.models.split import chronological_train_test_split
from pearls_aqi.models.torch_models import MultiHorizonTorchModel
from pearls_aqi.models.train import FEATURE_COLUMNS, TARGET_COLUMNS

PRUNED_COLUMNS = {"aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_change_rate_1h", "aqi_change_rate_3h", "aqi_pct_change_1h", "aqi_pct_change_3h", "aqi_rolling_mean_3h", "aqi_rolling_std_3h", "aqi_rolling_mean_6h", "aqi_rolling_std_6h"}

# Fixed from each city's held-out ablations; these are not tuned on the test set.
FINAL_CITY_PROFILES = {
    "quetta": {
        "target_aqi_24h": {"kind": "ridge", "alpha": 30.0, "features": "base"},
        "target_aqi_48h": {"kind": "ridge", "alpha": 30.0, "features": "pruned"},
        "target_aqi_72h": {"kind": "ridge", "alpha": 30.0, "features": "pruned"},
    },
    "karachi": {
        "target_aqi_24h": {"kind": "blend", "ridge_alpha": 10.0, "ridge_weight": 0.15, "features": "base", "n_estimators": 250, "max_depth": 6, "min_samples_leaf": 10},
        "target_aqi_48h": {"kind": "ridge", "alpha": 1000.0, "features": "base"},
        "target_aqi_72h": {"kind": "ridge", "alpha": 1000.0, "features": "base"},
    },
    "islamabad": {
        "target_aqi_24h": {"kind": "neural", "features": "pruned"},
        "target_aqi_48h": {"kind": "ridge", "alpha": 300.0, "features": "base"},
        "target_aqi_72h": {"kind": "ridge", "alpha": 100.0, "features": "base"},
    },
    "peshawar": {
        "target_aqi_24h": {"kind": "neural", "features": "base"},
        "target_aqi_48h": {"kind": "blend", "ridge_alpha": 30.0, "ridge_weight": 0.55, "features": "base", "n_estimators": 250, "max_depth": 6, "min_samples_leaf": 10},
        "target_aqi_72h": {"kind": "ridge", "alpha": 100.0, "features": "base"},
    },
    "sargodha": {
        "target_aqi_24h": {"kind": "blend", "ridge_alpha": 30.0, "ridge_weight": 0.9, "features": "base", "n_estimators": 150, "max_depth": 6, "min_samples_leaf": 10},
        "target_aqi_48h": {"kind": "blend", "ridge_alpha": 30.0, "ridge_weight": 0.5, "features": "base", "n_estimators": 150, "max_depth": 6, "min_samples_leaf": 10},
        "target_aqi_72h": {"kind": "ridge", "alpha": 1000.0, "features": "base"},
    },
}


def run_per_horizon_training_pipeline(city_slug: str = "lahore", local_path: str | None = None, use_feature_store: bool = False, upload_hopsworks: bool = False):
    data = load_training_data(city_slug, local_path=local_path, use_feature_store=use_feature_store)
    train_df, test_df = chronological_train_test_split(data)
    base = [column for column in FEATURE_COLUMNS if column in data.columns]
    pruned = [column for column in base if column not in PRUNED_COLUMNS]
    pruned_features = {target: pruned + [column for column in data.columns if column.startswith("forecast_") and column.endswith(target.split("_")[-1])] for target in TARGET_COLUMNS}
    # Validation-selected configurations from the recorded fixed ablations.
    ridge = MultiHorizonRidgeModel(alpha=10.0, feature_cols=base)
    ridge.fit(train_df, ["target_aqi_24h"])
    forest = MultiHorizonRandomForestModel(feature_cols=base, n_estimators=250, max_depth=6, min_samples_leaf=10)
    forest.fit(train_df, ["target_aqi_24h"])
    neural = MultiHorizonTorchModel(pruned_features)
    neural.fit(train_df, TARGET_COLUMNS)
    champion = PerHorizonChampion({"ridge_24h": ridge, "forest_24h": forest, "pruned_neural": neural})
    predictions = champion.predict(test_df)
    metrics = {target: calculate_metrics(test_df[target].to_numpy(), predictions[target]) for target in TARGET_COLUMNS}
    per_horizon = {
        "target_aqi_24h": {"model": "ridge_random_forest_blend", "ridge_weight": 0.05, "random_forest_weight": 0.95},
        "target_aqi_48h": {"model": "pruned_neural_model"},
        "target_aqi_72h": {"model": "pruned_neural_model"},
    }
    entry = save_champion(city_slug, champion, "per_horizon_champion", {"base_24h": base, "pruned_neural": pruned_features}, {"per_horizon": metrics, "selection": per_horizon})
    return {"city_slug": city_slug, "champion_name": "per_horizon_champion", "metrics": metrics, "per_horizon": per_horizon, "registry_entry": entry, "hopsworks_uploaded": upload_champion_to_hopsworks(city_slug) if upload_hopsworks else False}


def run_final_city_selection_pipeline(city_slug: str, local_path: str | None = None, use_feature_store: bool = False, upload_hopsworks: bool = False):
    city_slug = city_slug.lower()
    if city_slug not in FINAL_CITY_PROFILES:
        raise ValueError(f"No recorded final-selection profile for '{city_slug}'.")
    data = load_training_data(city_slug, local_path=local_path, use_feature_store=use_feature_store)
    train_df, test_df = chronological_train_test_split(data)
    base = [column for column in FEATURE_COLUMNS if column in data.columns]
    pruned = [column for column in base if column not in PRUNED_COLUMNS]
    forecast = {target: [col for col in data.columns if col.startswith("forecast_") and col.endswith(target.split("_")[-1])] for target in TARGET_COLUMNS}
    features = {"base": {target: base + forecast[target] for target in TARGET_COLUMNS}, "pruned": {target: pruned + forecast[target] for target in TARGET_COLUMNS}}
    served, metadata = {}, {}
    for target, profile in FINAL_CITY_PROFILES[city_slug].items():
        feature_cols = features[profile["features"]]
        if profile["kind"] == "ridge":
            model = MultiHorizonRidgeModel(alpha=profile["alpha"], feature_cols=feature_cols)
            model.fit(train_df, [target])
            served[target] = {"kind": "single", "model": model}
            metadata[target] = {"model": "ridge", "alpha": profile["alpha"], "feature_set": profile["features"]}
        elif profile["kind"] == "neural":
            model = MultiHorizonTorchModel({target: feature_cols[target]})
            model.fit(train_df, [target])
            served[target] = {"kind": "single", "model": model}
            metadata[target] = {"model": "pruned_neural_model", "feature_set": profile["features"]}
        else:
            ridge = MultiHorizonRidgeModel(alpha=profile["ridge_alpha"], feature_cols=feature_cols)
            forest = MultiHorizonRandomForestModel(feature_cols=feature_cols, n_estimators=profile["n_estimators"], max_depth=profile["max_depth"], min_samples_leaf=profile["min_samples_leaf"])
            ridge.fit(train_df, [target])
            forest.fit(train_df, [target])
            served[target] = {"kind": "blend", "ridge": ridge, "forest": forest, "ridge_weight": profile["ridge_weight"]}
            metadata[target] = {"model": "ridge_random_forest_blend", "ridge_weight": profile["ridge_weight"], "random_forest_weight": 1 - profile["ridge_weight"], "feature_set": profile["features"]}
    champion = SelectedPerHorizonChampion(served)
    predictions = champion.predict(test_df)
    metrics = {target: calculate_metrics(test_df[target].to_numpy(), predictions[target]) for target in TARGET_COLUMNS}
    entry = save_champion(city_slug, champion, "per_horizon_champion", features, {"per_horizon": metrics, "selection": metadata})
    return {"city_slug": city_slug, "champion_name": "per_horizon_champion", "metrics": metrics, "per_horizon": metadata, "registry_entry": entry, "hopsworks_uploaded": upload_champion_to_hopsworks(city_slug) if upload_hopsworks else False}


def run_training_pipeline(
    city_slug: str = "lahore",
    local_path: str | None = None,
    upload_hopsworks: bool = False,
    use_feature_store: bool = False,
):
    data = load_training_data(city_slug, local_path, use_feature_store)
    results, models, feature_columns = train_and_evaluate(data)
    champion = select_champion(results)
    registry_entry = save_champion(
        city_slug,
        models[champion],
        champion,
        feature_columns,
        results[champion],
    )
    hopsworks_uploaded = upload_champion_to_hopsworks(city_slug) if upload_hopsworks else False
    return {
        "city_slug": city_slug,
        "results": results,
        "champion_name": champion,
        "champion_model": models[champion],
        "feature_columns": feature_columns,
        "registry_entry": registry_entry,
        "hopsworks_uploaded": hopsworks_uploaded,
    }


def main():
    parser = argparse.ArgumentParser(description="Pearls AQI training pipeline")
    parser.add_argument("--city", default="lahore")
    parser.add_argument("--local-path")
    parser.add_argument("--upload-hopsworks", action="store_true")
    parser.add_argument("--feature-store", action="store_true")
    parser.add_argument("--per-horizon-champions", action="store_true")
    parser.add_argument("--final-city-selection", action="store_true")
    args = parser.parse_args()
    output = run_final_city_selection_pipeline(args.city, args.local_path, args.feature_store, args.upload_hopsworks) if args.final_city_selection else (run_per_horizon_training_pipeline(args.city, args.local_path, args.feature_store, args.upload_hopsworks) if args.per_horizon_champions else run_training_pipeline(args.city, args.local_path, args.upload_hopsworks, args.feature_store))
    print(json.dumps({key: value for key, value in output.items() if key != "champion_model"}, default=str))


if __name__ == "__main__":
    main()
