"""Fixed, leakage-safe per-city model ablations on a chronological holdout."""

import argparse
import json
from pathlib import Path

import numpy as np

from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.evaluate import calculate_metrics
from pearls_aqi.models.sklearn_models import MultiHorizonRandomForestModel, MultiHorizonRidgeModel
from pearls_aqi.models.split import chronological_train_test_split
from pearls_aqi.models.train import (
    FEATURE_COLUMNS,
    FORECAST_WEATHER_COLUMNS,
    TARGET_COLUMNS,
    train_and_evaluate,
)

PRUNED_COLUMNS = {
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_change_rate_1h", "aqi_change_rate_3h",
    "aqi_pct_change_1h", "aqi_pct_change_3h", "aqi_rolling_mean_3h", "aqi_rolling_std_3h",
    "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
}
STRONG_REGULARIZATION = {
    target: {
        "ridge": {"alpha_grid": [30.0, 100.0, 300.0, 1000.0]},
        "random_forest": {"n_estimators": [150, 250], "max_depth": [3, 4, 6, 8], "min_samples_leaf": [10, 20, 40]},
        "tuning": {"rf_candidates": 8},
    }
    for target in ("target_aqi_48h", "target_aqi_72h")
}


def feature_sets(df):
    base = [col for col in FEATURE_COLUMNS if col in df.columns]
    return {target: base + [col for col in FORECAST_WEATHER_COLUMNS.get(target, []) if col in df.columns] for target in TARGET_COLUMNS}


def add_calibrated_blend(df, results, models, features):
    """Choose one Ridge/RF weight per horizon on an inner chronological validation window."""
    train_df, test_df = chronological_train_test_split(df)
    cut = int(len(train_df) * 0.85)
    inner_train, validation = train_df.iloc[:cut], train_df.iloc[cut:]
    # Configurations are fixed before the blend comparison; only its weight is
    # chosen on validation. This avoids another costly CV search per blend.
    ridge_inner = MultiHorizonRidgeModel(alpha=30.0, feature_cols=features)
    forest_inner = MultiHorizonRandomForestModel(feature_cols=features, n_estimators=150, max_depth=6, min_samples_leaf=10)
    ridge_inner.fit(inner_train, TARGET_COLUMNS)
    forest_inner.fit(inner_train, TARGET_COLUMNS)
    ridge_val, forest_val = ridge_inner.predict(validation), forest_inner.predict(validation)
    weights, blend_metrics = {}, {}
    ridge_test, forest_test = models["ridge_model"].predict(test_df), models["random_forest_model"].predict(test_df)
    for target in TARGET_COLUMNS:
        candidates = np.linspace(0.0, 1.0, 21)
        weight = min(candidates, key=lambda w: calculate_metrics(validation[target].to_numpy(), w * ridge_val[target] + (1 - w) * forest_val[target])["mae"])
        weights[target] = float(weight)
        blend_metrics[target] = calculate_metrics(test_df[target].to_numpy(), weight * ridge_test[target] + (1 - weight) * forest_test[target])
    results["ridge_random_forest_blend"] = blend_metrics
    results["blend_validation_weights"] = weights
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("pruning", "regularization", "blend"), required=True)
    parser.add_argument("--city", default="lahore")
    parser.add_argument("--local-path")
    args = parser.parse_args()
    df = load_training_data(args.city, args.local_path)
    features = feature_sets(df)
    if args.experiment == "pruning":
        features = {target: [col for col in cols if col not in PRUNED_COLUMNS] for target, cols in features.items()}
        results, _, _ = train_and_evaluate(df, feature_columns_override=features)
        details = {"removed_columns": sorted(PRUNED_COLUMNS)}
    elif args.experiment == "regularization":
        results, _, _ = train_and_evaluate(df, per_target_config=STRONG_REGULARIZATION)
        details = {"validation_search": STRONG_REGULARIZATION}
    else:
        # A blend uses only Ridge and RF.  Avoid retraining the neural model here;
        # its independent pruning/regularization experiment remains the comparison.
        train_df, test_df = chronological_train_test_split(df)
        ridge = MultiHorizonRidgeModel(alpha=30.0, feature_cols=features)
        forest = MultiHorizonRandomForestModel(feature_cols=features, n_estimators=150, max_depth=6, min_samples_leaf=10)
        ridge.fit(train_df, TARGET_COLUMNS)
        forest.fit(train_df, TARGET_COLUMNS)
        models = {"ridge_model": ridge, "random_forest_model": forest}
        results = {
            "ridge_model": {target: calculate_metrics(test_df[target].to_numpy(), ridge.predict(test_df)[target]) for target in TARGET_COLUMNS},
            "random_forest_model": {target: calculate_metrics(test_df[target].to_numpy(), forest.predict(test_df)[target]) for target in TARGET_COLUMNS},
            "training_diagnostics": {"configuration": "fixed_prevalidated_ridge_rf; blend weight selected on inner chronological validation"},
        }
        results = add_calibrated_blend(df, results, models, features)
        details = {"weights_selected_on": "last 15% of chronological training partition"}
    output = {"experiment": args.experiment, "details": details, "results": results}
    output_dir = Path("artifacts") / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.city}_{args.experiment}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
