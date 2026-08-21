# Pearls AQI Predictor — Project Explanation

## What the system does

Pearls AQI Predictor forecasts US AQI 24, 48, and 72 hours ahead for Karachi, Lahore, Islamabad, Peshawar, Quetta, and Sargodha. It uses Open-Meteo/CAMS modeled air-quality and weather data, clearly labelled as modeled data rather than official monitoring-station observations.

## Data and pipeline

The local backfill covers **2022-08-01 to 2025-08-01**: 26,328 hourly rows per city. Open-Meteo supplies US AQI, PM2.5, PM10, CO, NO2, SO2, ozone, temperature, humidity, pressure, wind, and precipitation.

```text
Open-Meteo → cleaning/validation → engineered features → local CSV + Hopsworks
→ chronological model training → local/Hopsworks registry → FastAPI → Streamlit
```

Features include calendar cycles, current weather/pollutants/AQI, AQI lags from 1 to 72 hours, change rates, rolling statistics, and target-time weather forecast features for 48h/72h. Data is sorted by city and UTC timestamp before lag/rolling calculations, preventing batch-boundary resets and future leakage.

## Training and evaluation

The earliest 80% of each city series trains and validates models; the newest 20% is held out chronologically. The project compares persistence and seasonal-naive baselines with Ridge Regression, Random Forest, and a compact PyTorch feed-forward neural network. Ridge/RF hyperparameters and blend weights are selected only on chronological validation data.

MAE and RMSE should be minimized. Positive R² means a model beats predicting the holdout mean. Each horizon can use a different champion because its predictability and ideal regularization differ.

## Registered champions

| City | 24h champion (R²) | 48h champion (R²) | 72h champion (R²) |
|---|---|---|---|
| Lahore | Ridge/RF blend (0.521) | Pruned neural (0.149) | Pruned neural (0.195) |
| Karachi | Ridge/RF blend (0.545) | Ridge (0.243) | Ridge (0.113) |
| Islamabad | Pruned neural (0.698) | Ridge (0.445) | Ridge (0.378) |
| Peshawar | Neural (0.647) | Ridge/RF blend (0.450) | Ridge (0.391) |
| Quetta | Ridge (0.434) | Pruned Ridge (0.122) | Pruned Ridge (0.002) |
| Sargodha | Ridge/RF blend (0.687) | Ridge/RF blend (0.432) | Ridge (0.352) |

## Feature Store, API, dashboard, and automation

Local development is fully offline-capable: data is under `artifacts/data/`; per-city model artifacts are under `artifacts/models/<city>/`. FastAPI starts and serves from those local artifacts without connecting to Hopsworks. Hopsworks is optional and uses Feature Group `aqi_features_hourly_v2:1`, Feature View `aqi_features_fv_v2:1`, plus best-effort registry upload. The live Hopsworks connection test passes with configured credentials.

Run the API:

```powershell
.\venv\Scripts\uvicorn.exe api.main:app --reload --port 8000
```

Run the dashboard:

```powershell
.\venv\Scripts\streamlit.exe run dashboard\app.py
```

The dashboard provides AQI forecasts, categories/alerts, city comparison, trend chart, per-horizon analytics, SHAP global/local explanations with a permutation fallback, and a grounded Copilot tab. The repository contains scheduled hourly ingestion and daily retraining GitHub Actions workflows with concurrency guards and manual triggers.

## Repository guide

| Location | Purpose |
|---|---|
| `src/pearls_aqi/data/` | Open-Meteo providers, cleaning, validation |
| `src/pearls_aqi/features/` | Feature/target engineering and Hopsworks access |
| `src/pearls_aqi/models/` | Training, evaluation, registry, SHAP explanations |
| `pipelines/` | Backfill, feature ingestion, training and ablation CLIs |
| `api/` and `dashboard/` | FastAPI service and Streamlit UI |
| `artifacts/` | Data, models, experiments, EDA outputs |

Default tests skip live integrations. Run `pytest --run-integration` when credentials/network access are available.
