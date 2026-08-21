"""Model-agnostic permutation explanations for AQI forecasts."""

from typing import Any

import pandas as pd
import numpy as np


def _components(model: Any, target: str) -> list[tuple[float, Any]]:
    """Resolve a single or blended per-horizon champion into model components."""
    if hasattr(model, "selections"):
        selection = model.selections[target]
        if selection["kind"] == "blend":
            weight = selection["ridge_weight"]
            return [(weight, selection["ridge"]), (1 - weight, selection["forest"])]
        return [(1.0, selection["model"])]
    if hasattr(model, "models") and "ridge_24h" in model.models:
        if target == "target_aqi_24h":
            return [(model.blend_weight_24h, model.models["ridge_24h"]), (1 - model.blend_weight_24h, model.models["forest_24h"])]
        return [(1.0, model.models["pruned_neural"])]
    return [(1.0, model)]


def _shap_inputs(model: Any, data: pd.DataFrame, horizon_hours: int):
    """Return fitted estimator and feature matrix after pipeline preprocessing."""
    target = f"target_aqi_{horizon_hours}h"
    pipeline = model.models[target]
    features = model.feature_cols[target] if isinstance(model.feature_cols, dict) else model.feature_cols
    transformed = pipeline[:-1].transform(data[features])
    return target, pipeline[-1], features, transformed


def shap_feature_importance(model: Any, data: pd.DataFrame, horizon_hours: int) -> list[dict]:
    """Primary SHAP global importance for Ridge and Random Forest pipelines."""
    import shap

    target = f"target_aqi_{horizon_hours}h"
    combined = {}
    for weight, component in _components(model, target):
        _, estimator, features, transformed = _shap_inputs(component, data, horizon_hours)
        sample = transformed[: min(len(transformed), 500)]
        values = shap.TreeExplainer(estimator).shap_values(sample) if hasattr(estimator, "estimators_") else shap.LinearExplainer(estimator, sample).shap_values(sample)
        for feature, score in zip(features, abs(values).mean(axis=0)):
            combined[feature] = combined.get(feature, 0.0) + weight * float(score)
    return [{"feature": feature, "importance": score} for feature, score in sorted(combined.items(), key=lambda item: item[1], reverse=True)]


def shap_local_explanation(model: Any, row: pd.DataFrame, horizon_hours: int) -> dict:
    """Primary SHAP local explanation for one forecast."""
    import shap

    target = f"target_aqi_{horizon_hours}h"
    combined = {}
    for weight, component in _components(model, target):
        _, estimator, features, transformed = _shap_inputs(component, row, horizon_hours)
        values = shap.TreeExplainer(estimator).shap_values(transformed)[0] if hasattr(estimator, "estimators_") else shap.LinearExplainer(estimator, transformed).shap_values(transformed)[0]
        for feature, value in zip(features, values):
            combined[feature] = combined.get(feature, 0.0) + weight * float(value)
    contributions = [{"feature": feature, "shap_value": value, "feature_value": float(row.iloc[0][feature])} for feature, value in sorted(combined.items(), key=lambda item: abs(item[1]), reverse=True)]
    return {"horizon_hours": horizon_hours, "prediction": float(model.predict(row)[target][0]), "contributions": contributions}


def global_feature_importance(model: Any, data: pd.DataFrame, horizon_hours: int) -> list[dict]:
    """Model-agnostic permutation fallback, including PyTorch and blends."""
    target = f"target_aqi_{horizon_hours}h"
    components = _components(model, target)
    metadata_cols = sorted(set().union(*[(component.feature_cols[target] if isinstance(component.feature_cols, dict) else component.feature_cols) for _, component in components]))
    valid = data.loc[data[target].notna()].tail(500).copy()
    if valid.empty:
        raise ValueError(f"No observed targets available for {target} explanation.")
    actual = valid[target].to_numpy(dtype=float)
    baseline_mae = float(np.mean(np.abs(actual - model.predict(valid)[target])))
    rng = np.random.default_rng(42)
    scores = []
    for column in metadata_cols:
        permuted = valid.copy()
        permuted[column] = rng.permutation(permuted[column].to_numpy())
        permuted_mae = float(np.mean(np.abs(actual - model.predict(permuted)[target])))
        scores.append(permuted_mae - baseline_mae)
    return [
        {"feature": name, "importance": float(score)}
        for name, score in sorted(zip(metadata_cols, scores), key=lambda item: item[1], reverse=True)
    ]


def explain_prediction(model: Any, row: pd.DataFrame, horizon_hours: int) -> dict:
    """Provide model/version-neutral local feature values for a future Copilot tool."""
    target = f"target_aqi_{horizon_hours}h"
    prediction = float(model.predict(row)[target][0])
    component = _components(model, target)[0][1]
    cols = component.feature_cols[target] if isinstance(component.feature_cols, dict) else component.feature_cols
    values = row.iloc[0][cols].to_dict()
    return {"horizon_hours": horizon_hours, "prediction": prediction, "feature_values": values}
