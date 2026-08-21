"""Training orchestration and comparison against baseline models."""

from copy import deepcopy
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pearls_aqi.features.builder import build_features
from pearls_aqi.features.targets import build_targets
from pearls_aqi.models.baselines import PersistenceBaseline, SeasonalNaiveBaseline
from pearls_aqi.models.evaluate import calculate_metrics
from pearls_aqi.models.sklearn_models import MultiHorizonRandomForestModel, MultiHorizonRidgeModel
from pearls_aqi.models.split import chronological_train_test_split
from pearls_aqi.models.torch_models import MultiHorizonTorchModel
from pearls_aqi.settings import settings

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "temperature_2m_c", "relative_humidity_2m_pct", "surface_pressure_hpa",
    "wind_speed_10m_kph", "precipitation_mm",
    "pm2_5_ug_m3", "pm10_ug_m3", "carbon_monoxide_ug_m3",
    "nitrogen_dioxide_ug_m3", "sulphur_dioxide_ug_m3", "ozone_ug_m3",
    "aqi", "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    "aqi_change_rate_1h", "aqi_change_rate_3h", "aqi_change_rate_24h",
    "aqi_pct_change_1h", "aqi_pct_change_3h", "aqi_pct_change_24h",
    "aqi_rolling_mean_3h", "aqi_rolling_std_3h",
    "aqi_rolling_mean_6h", "aqi_rolling_std_6h",
    "aqi_rolling_mean_12h", "aqi_rolling_std_12h",
    "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
]

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
FORECAST_WEATHER_COLUMNS = {
    "target_aqi_48h": [
        "forecast_temperature_2m_c_48h", "forecast_relative_humidity_2m_pct_48h",
        "forecast_surface_pressure_hpa_48h", "forecast_wind_speed_10m_kph_48h",
        "forecast_precipitation_mm_48h",
    ],
    "target_aqi_72h": [
        "forecast_temperature_2m_c_72h", "forecast_relative_humidity_2m_pct_72h",
        "forecast_surface_pressure_hpa_72h", "forecast_wind_speed_10m_kph_72h",
        "forecast_precipitation_mm_72h",
    ],
}


def _metrics_by_target(test_df: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return {
        target: calculate_metrics(test_df[target].to_numpy(), predictions[target])
        for target in TARGET_COLUMNS
    }


def _tuned_models(
    train_df: pd.DataFrame,
    feature_columns: list[str] | Dict[str, list[str]],
    per_target_config: Dict[str, Dict[str, Any]] | None = None,
):
    config = settings.load_model_config()
    ridge = MultiHorizonRidgeModel(feature_cols=feature_columns)
    forest = MultiHorizonRandomForestModel(feature_cols=feature_columns)
    diagnostics = {}
    for target in TARGET_COLUMNS:
        target_config = deepcopy(config)
        for section, values in (per_target_config or {}).get(target, {}).items():
            target_config[section].update(values)
        labelled = train_df.loc[train_df[target].notna()].copy()
        if len(labelled) < 12:
            raise ValueError(f"Insufficient labelled rows for {target}: {len(labelled)}")
        cv = TimeSeriesSplit(n_splits=min(target_config["tuning"]["cv_splits"], max(2, len(labelled) // 4)))
        columns = feature_columns[target] if isinstance(feature_columns, dict) else feature_columns
        X, y = labelled[columns], labelled[target]
        ridge_search = GridSearchCV(
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge())]),
            {"ridge__alpha": target_config["ridge"]["alpha_grid"]}, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1,
        ).fit(X, y)
        rf_search = RandomizedSearchCV(
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("random_forest", RandomForestRegressor(random_state=42, n_jobs=1))]),
            {"random_forest__n_estimators": target_config["random_forest"]["n_estimators"], "random_forest__max_depth": target_config["random_forest"]["max_depth"], "random_forest__min_samples_leaf": target_config["random_forest"]["min_samples_leaf"]},
            n_iter=target_config["tuning"]["rf_candidates"], scoring="neg_mean_absolute_error", cv=cv, random_state=42, n_jobs=-1,
        ).fit(X, y)
        ridge.models[target] = ridge_search.best_estimator_
        forest.models[target] = rf_search.best_estimator_
        diagnostics[target] = {"training_rows": len(labelled), "complete_case_rows": int(labelled[columns].notna().all(axis=1).sum()), "feature_count": len(columns), "ridge_params": ridge_search.best_params_, "random_forest_params": rf_search.best_params_}
    return ridge, forest, diagnostics


def train_and_evaluate(
    df_full: pd.DataFrame,
    feature_columns_override: Dict[str, list[str]] | None = None,
    per_target_config: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], list[str]]:
    """Evaluate baselines, Ridge and Random Forest on engineered feature data."""
    base_features = [column for column in FEATURE_COLUMNS if column in df_full.columns]
    if not base_features:
        raise ValueError("No configured feature columns are present in training data.")
    avail_features = {
        target: base_features + [column for column in FORECAST_WEATHER_COLUMNS.get(target, []) if column in df_full.columns]
        for target in TARGET_COLUMNS
    }
    if feature_columns_override is not None:
        avail_features = feature_columns_override
    train_df, test_df = chronological_train_test_split(df_full, train_ratio=0.8)
    results = {}
    persistence = PersistenceBaseline()
    y_pred_pers = persistence.predict(test_df, target_col="aqi")
    results["persistence_baseline"] = {
        target: calculate_metrics(test_df[target].to_numpy(), y_pred_pers) for target in TARGET_COLUMNS
    }
    seasonal_naive = SeasonalNaiveBaseline()
    y_pred_seas = seasonal_naive.predict(test_df, lag_col="aqi_lag_24h")
    results["seasonal_naive_baseline"] = {
        target: calculate_metrics(test_df[target].to_numpy(), y_pred_seas) for target in TARGET_COLUMNS
    }
    ridge_model, rf_model, diagnostics = _tuned_models(train_df, avail_features, per_target_config)
    results["ridge_model"] = _metrics_by_target(test_df, ridge_model.predict(test_df))
    results["random_forest_model"] = _metrics_by_target(test_df, rf_model.predict(test_df))
    neural_model = MultiHorizonTorchModel(avail_features)
    neural_model.fit(train_df, TARGET_COLUMNS)
    results["neural_model"] = _metrics_by_target(test_df, neural_model.predict(test_df))

    trained_models = {"ridge_model": ridge_model, "random_forest_model": rf_model, "neural_model": neural_model}
    results["training_diagnostics"] = diagnostics
    return results, trained_models, sorted(set().union(*avail_features.values()))


def select_champion(results: Dict[str, Any]) -> str:
    """Select the lowest mean finite MAE among trainable models."""
    candidates = tuple(name for name in ("ridge_model", "random_forest_model", "neural_model") if name in results)
    return min(
        candidates,
        key=lambda name: sum(metrics["mae"] for metrics in results[name].values()) / len(TARGET_COLUMNS),
    )


def train_and_evaluate_lahore_slice(raw_merged_df: pd.DataFrame) -> Dict[str, Any]:
    """Backward-compatible vertical slice entry point for raw observations."""
    df_full = build_targets(build_features(raw_merged_df))
    results, _, _ = train_and_evaluate(df_full)

    return results
