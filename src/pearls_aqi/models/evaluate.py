"""Evaluation metrics: MAE, RMSE, R2 per horizon and overall."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, R2 metrics for true and predicted targets."""
    # Filter out NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    yt = y_true[mask]
    yp = y_pred[mask]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))

    return {"mae": mae, "rmse": rmse, "r2": r2}
