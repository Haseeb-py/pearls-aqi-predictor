"""Unit tests for models, splitting, baselines, and metrics."""

import numpy as np
import pandas as pd

from pearls_aqi.models.evaluate import calculate_metrics
from pearls_aqi.models.sklearn_models import MultiHorizonRandomForestModel, MultiHorizonRidgeModel
from pearls_aqi.models.train import select_champion
from pearls_aqi.models.torch_models import MultiHorizonTorchModel
from pearls_aqi.models.split import chronological_train_test_split


def test_calculate_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])
    res = calculate_metrics(y_true, y_pred)

    assert "mae" in res
    assert "rmse" in res
    assert "r2" in res
    assert res["mae"] > 0
    assert res["r2"] <= 1.0


def test_chronological_train_test_split():
    times = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"event_time_utc": times, "val": range(10)})

    train_df, test_df = chronological_train_test_split(df, train_ratio=0.8)
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert train_df["event_time_utc"].max() < test_df["event_time_utc"].min()


def test_multi_horizon_ridge_model():
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "target_aqi_24h": np.random.randn(50) * 10 + 100,
            "target_aqi_48h": np.random.randn(50) * 10 + 105,
        }
    )

    model = MultiHorizonRidgeModel(alpha=1.0, feature_cols=["feature1", "feature2"])
    model.fit(df.iloc[:40], target_cols=["target_aqi_24h", "target_aqi_48h"])

    preds = model.predict(df.iloc[40:])
    assert "target_aqi_24h" in preds
    assert len(preds["target_aqi_24h"]) == 10


def test_multi_horizon_random_forest_model():
    df = pd.DataFrame({"feature": range(20), "target_aqi_24h": range(20)})
    model = MultiHorizonRandomForestModel(feature_cols=["feature"], n_estimators=5)
    model.fit(df.iloc[:15], ["target_aqi_24h"])
    assert len(model.predict(df.iloc[15:])["target_aqi_24h"]) == 5


def test_select_champion_uses_mean_mae():
    results = {
        "ridge_model": {target: {"mae": 2.0} for target in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]},
        "random_forest_model": {target: {"mae": 1.0} for target in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]},
    }
    assert select_champion(results) == "random_forest_model"


def test_multi_horizon_torch_model_uses_train_validation_split():
    df = pd.DataFrame({
        "feature1": np.linspace(0, 1, 50),
        "feature2": np.linspace(1, 0, 50),
    })
    df["target_aqi_24h"] = 50 + 20 * df["feature1"] - 5 * df["feature2"]
    model = MultiHorizonTorchModel(["feature1", "feature2"], epochs=4, hidden_size=8)
    model.fit(df.iloc[:40], ["target_aqi_24h"])
    prediction = model.predict(df.iloc[40:])["target_aqi_24h"]
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()
