# Pearls AQI Predictor — Final Report

## Objective

Predict US AQI 24, 48, and 72 hours ahead for six Pakistani cities using an
automated, serverless-ready ML pipeline.

## Data and feature pipeline

Open-Meteo/CAMS provides hourly pollutant and weather observations/forecasts.
The pipeline validates and combines PM2.5, PM10, CO, NO2, SO2, ozone,
temperature, humidity, pressure, wind, and precipitation. It derives calendar,
cyclical-time, AQI lag, AQI change-rate, percentage-change, rolling-statistic,
and future-weather features. Historical backfill provides the training set;
the hourly GitHub Action updates the Hopsworks feature group
`aqi_features_hourly_v2`.

## Modelling

Chronological splits prevent future leakage. Ridge Regression, regularized
Random Forest, and a PyTorch feed-forward neural network were evaluated with
MAE, RMSE, and R². Selection is per horizon because short and long forecasts
have different error patterns. Model artifacts include the selected feature
sets and metrics, and are stored locally plus Hopsworks Model Registry.

## Latest held-out evaluation

| City | 24h R² | 48h R² | 72h R² |
|---|---:|---:|---:|
| Islamabad | 0.698 | 0.445 | 0.378 |
| Karachi | 0.545 | 0.243 | 0.113 |
| Lahore | 0.521 | 0.149 | 0.195 |
| Peshawar | 0.647 | 0.450 | 0.391 |
| Quetta | 0.434 | 0.122 | 0.002 |
| Sargodha | 0.687 | 0.432 | 0.352 |

Longer horizons are intrinsically harder; results are reported rather than
artificially tuned against the held-out test period. Full MAE/RMSE/R² metadata
is stored beside every champion model.

## Application and explainability

FastAPI exposes city and three-horizon prediction endpoints. Streamlit presents
current AQI, forecasts, trend/history, comparison, hazardous-AQI alerts, model
metrics, SHAP global/local explanations (with a safe fallback), and a grounded
AQI Copilot. The Copilot only uses application tools for measured values and
flags stale stored data.

## Automation and deployment

GitHub Actions runs hourly feature ingestion and daily training with concurrency
guards. The repository is configured for FastAPI on Cloud Run and the UI on
Streamlit Community Cloud; both use Hopsworks secrets in their host secret
managers. See `docs/deployment.md`.

## Limitations

Open-Meteo/CAMS is modeled environmental data rather than a certified local
monitor. Forecasts are estimates, not medical advice or source attribution.
Performance should be monitored over future seasons and retraining should be
reviewed if data quality or error distributions change.
