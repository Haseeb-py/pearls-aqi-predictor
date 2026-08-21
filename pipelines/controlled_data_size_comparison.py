"""Compare one- versus three-year training with an identical final holdout."""

import json

from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.evaluate import calculate_metrics
from pearls_aqi.models.per_horizon import PerHorizonChampion
from pearls_aqi.models.sklearn_models import MultiHorizonRandomForestModel, MultiHorizonRidgeModel
from pearls_aqi.models.split import chronological_train_test_split
from pearls_aqi.models.torch_models import MultiHorizonTorchModel
from pearls_aqi.models.train import FEATURE_COLUMNS, TARGET_COLUMNS

PRUNED_COLUMNS = {"aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_change_rate_1h", "aqi_change_rate_3h", "aqi_pct_change_1h", "aqi_pct_change_3h", "aqi_rolling_mean_3h", "aqi_rolling_std_3h", "aqi_rolling_mean_6h", "aqi_rolling_std_6h"}


def fit_and_score(data, train, test):
    base = [column for column in FEATURE_COLUMNS if column in data.columns]
    pruned = [column for column in base if column not in PRUNED_COLUMNS]
    features = {target: pruned + [column for column in data.columns if column.startswith("forecast_") and column.endswith(target.split("_")[-1])] for target in TARGET_COLUMNS}
    ridge = MultiHorizonRidgeModel(alpha=10.0, feature_cols=base)
    ridge.fit(train, ["target_aqi_24h"])
    forest = MultiHorizonRandomForestModel(feature_cols=base, n_estimators=250, max_depth=6, min_samples_leaf=10)
    forest.fit(train, ["target_aqi_24h"])
    neural = MultiHorizonTorchModel(features)
    neural.fit(train, TARGET_COLUMNS)
    model = PerHorizonChampion({"ridge_24h": ridge, "forest_24h": forest, "pruned_neural": neural})
    predictions = model.predict(test)
    return {target: calculate_metrics(test[target].to_numpy(), predictions[target]) for target in TARGET_COLUMNS}


def main():
    one = load_training_data("lahore", "artifacts/data/backfill_2024-08-01_to_2025-08-01.csv")
    three = load_training_data("lahore", "artifacts/data/backfill_2022-08-01_to_2025-08-01.csv")
    three_train, three_test = chronological_train_test_split(three)
    cutoff = three_test["event_time_utc"].min()
    one_train = one.loc[one["event_time_utc"] < cutoff].copy()
    one_test = one.loc[one["event_time_utc"] >= cutoff].copy()
    # Require exact timestamp agreement; the target can legitimately be NaN at the final horizons.
    assert one_test["event_time_utc"].tolist() == three_test["event_time_utc"].tolist()
    output = {
        "shared_holdout": {"start": cutoff.isoformat(), "end": three_test["event_time_utc"].max().isoformat(), "rows": len(three_test)},
        "one_year_training_rows": len(one_train), "three_year_training_rows": len(three_train),
        "one_year": fit_and_score(one, one_train, one_test),
        "three_year": fit_and_score(three, three_train, three_test),
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
