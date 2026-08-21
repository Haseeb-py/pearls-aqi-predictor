import pandas as pd

from pearls_aqi.models.explain import explain_prediction
from pearls_aqi.models.sklearn_models import MultiHorizonRidgeModel


def test_explain_prediction_returns_horizon_and_features():
    data = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target_aqi_24h": [2.0, 3.0, 4.0]})
    model = MultiHorizonRidgeModel(feature_cols=["feature"])
    model.fit(data, ["target_aqi_24h"])
    explanation = explain_prediction(model, data.iloc[[-1]], 24)
    assert explanation["horizon_hours"] == 24
    assert "feature" in explanation["feature_values"]
