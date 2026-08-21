"""Ridge regression preprocessing and modeling pipeline."""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MultiHorizonRidgeModel:
    """Preprocessed Ridge Regression model trained separately for each horizon."""

    def __init__(self, alpha: float = 1.0, feature_cols: List[str] = None):
        self.alpha = alpha
        self.feature_cols = feature_cols or []
        self.models: Dict[str, Pipeline] = {}

    def _columns_for(self, target: str) -> List[str]:
        return self.feature_cols[target] if isinstance(self.feature_cols, dict) else self.feature_cols

    def fit(self, train_df: pd.DataFrame, target_cols: List[str]):
        """Fit a Ridge pipeline for each target column in target_cols."""
        for target in target_cols:
            X_train = train_df[self._columns_for(target)]
            y_train = train_df[target]
            # Exclude rows where target is NaN
            valid_mask = y_train.notna()
            X_valid = X_train[valid_mask]
            y_valid = y_train[valid_mask]

            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=self.alpha)),
                ]
            )
            pipeline.fit(X_valid, y_valid)
            self.models[target] = pipeline

    def predict(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions for each target column."""
        predictions = {}
        for target, pipeline in self.models.items():
            predictions[target] = pipeline.predict(test_df[self._columns_for(target)])
        return predictions


class MultiHorizonRandomForestModel:
    """Random Forest trained independently for each forecasting horizon."""

    def __init__(
        self,
        feature_cols: List[str] = None,
        n_estimators: int = 100,
        max_depth: int | None = 12,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols or []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.models: Dict[str, Pipeline] = {}

    def _columns_for(self, target: str) -> List[str]:
        return self.feature_cols[target] if isinstance(self.feature_cols, dict) else self.feature_cols

    def fit(self, train_df: pd.DataFrame, target_cols: List[str]):
        for target in target_cols:
            X_train = train_df[self._columns_for(target)]
            valid = train_df[target].notna()
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "random_forest",
                        RandomForestRegressor(
                            n_estimators=self.n_estimators,
                            max_depth=self.max_depth,
                            min_samples_leaf=self.min_samples_leaf,
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            pipeline.fit(X_train.loc[valid], train_df.loc[valid, target])
            self.models[target] = pipeline

    def predict(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {target: pipeline.predict(test_df[self._columns_for(target)]) for target, pipeline in self.models.items()}
