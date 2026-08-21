"""Statistical naive and persistence baseline models."""

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Baseline predicting current AQI for all target horizons."""

    def predict(self, df: pd.DataFrame, target_col: str = "aqi") -> np.ndarray:
        """Predict current target_col value for target horizon."""
        return df[target_col].to_numpy()


class SeasonalNaiveBaseline:
    """Baseline predicting AQI from 24 hours ago (same hour yesterday)."""

    def predict(self, df: pd.DataFrame, lag_col: str = "aqi_lag_24h") -> np.ndarray:
        """Predict aqi_lag_24h value for target horizon."""
        if lag_col in df.columns:
            return df[lag_col].to_numpy()
        return df["aqi"].to_numpy()
