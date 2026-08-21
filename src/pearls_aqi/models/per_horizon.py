"""Serving wrapper for independently selected horizon models."""

from typing import Any, Dict

import numpy as np
import pandas as pd


class PerHorizonChampion:
    def __init__(self, models: Dict[str, Any], blend_weight_24h: float = 0.05):
        self.models = models
        self.blend_weight_24h = blend_weight_24h

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        ridge_prediction = self.models["ridge_24h"].predict(df)["target_aqi_24h"]
        forest_prediction = self.models["forest_24h"].predict(df)["target_aqi_24h"]
        neural_predictions = self.models["pruned_neural"].predict(df)
        return {
            "target_aqi_24h": self.blend_weight_24h * ridge_prediction + (1 - self.blend_weight_24h) * forest_prediction,
            "target_aqi_48h": neural_predictions["target_aqi_48h"],
            "target_aqi_72h": neural_predictions["target_aqi_72h"],
        }


class SelectedPerHorizonChampion:
    """Serve independently selected models, including an optional blend."""

    def __init__(self, selections: Dict[str, Dict[str, Any]]):
        self.selections = selections

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for target, selection in self.selections.items():
            if selection["kind"] == "blend":
                ridge = selection["ridge"].predict(df)[target]
                forest = selection["forest"].predict(df)[target]
                weight = selection["ridge_weight"]
                predictions[target] = weight * ridge + (1 - weight) * forest
            else:
                predictions[target] = selection["model"].predict(df)[target]
        return predictions
