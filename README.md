# Pearls AQI Predictor

A multi-city, serverless machine learning system that forecasts the US Air Quality Index (AQI) for major Pakistani cities at **24, 48, and 72 hours** ahead — with a live API, an interactive dashboard, and a grounded, tool-calling conversational assistant (the **AQI Copilot**).

**Live API:** [aqi-predict-9a0f8674.fastapicloud.dev](https://aqi-predict-9a0f8674.fastapicloud.dev)
**Live Dashboard:** [pearls-aqi-predictor-intern.streamlit.app](https://pearls-aqi-predictor-intern.streamlit.app)

---

## Table of Contents

- [Overview](#overview)
- [Cities Supported](#cities-supported)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Modeling Methodology](#modeling-methodology)
- [Results](#results)
- [Feature Store & Model Registry](#feature-store--model-registry)
- [Explainability](#explainability)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [AQI Copilot](#aqi-copilot)
- [Automation (CI/CD)](#automation-cicd)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Pearls AQI Predictor covers the full lifecycle of a production ML product:

1. **Ingest** raw weather and pollutant data hourly from Open-Meteo.
2. **Engineer** point-in-time-correct features (calendar, lag, rolling, weather, pollutant) with strict leakage protection.
3. **Train and evaluate** multiple candidate models (baselines, Ridge, Random Forest, PyTorch neural network) per city, per forecast horizon.
4. **Store** features and models in both local artifacts and the Hopsworks Feature Store / Model Registry.
5. **Serve** predictions through a FastAPI backend.
6. **Present** forecasts through a Streamlit dashboard and a grounded LLM-powered chat assistant.

The system is intentionally **serverless and batch-oriented**: no server runs continuously. GitHub Actions provisions temporary runners on a schedule (hourly for features, daily for training), and both the API and dashboard are hosted on managed platforms (FastAPI Cloud, Streamlit Community Cloud).

> **Data provenance note:** All air-quality and weather data is *modeled* atmospheric data (CAMS global model, via Open-Meteo), not physical ground-station sensor readings. This is disclosed throughout the product rather than presented as raw measurement.

---

## Cities Supported

| City | Backfill range | Rows | Status |
|---|---|---|---|
| Lahore | 2022-08-01 – 2025-08-01 | 26,328 | Registered (local + Hopsworks) |
| Karachi | 2022-08-01 – 2025-08-01 | 26,328 | Registered (local) |
| Islamabad | 2022-08-01 – 2025-08-01 | 26,328 | Registered (local) |
| Peshawar | 2022-08-01 – 2025-08-01 | 26,328 | Registered (local) |
| Quetta | 2022-08-01 – 2025-08-01 | 26,328 | Registered (local) |
| Sargodha | 2022-08-01 – 2025-08-01 | 26,328 | Registered (local) |

Additional cities can be added by extending `config/cities.yaml` and running the backfill + training pipelines for the new city.

---

## Architecture

```
Open-Meteo Air Quality API (CAMS)  ─┐
Open-Meteo Weather API (fcst/hist) ─┴─► Validation
                                          │
                                          ▼
                              Feature Engineering
                    (calendar · cyclical · lag · rolling · targets)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                     ▼
              Local CSV artifacts                    Hopsworks Feature Store
              (offline fallback)                      (feature group + view)
                        │                                     │
                        └─────────────────┬───────────────────┘
                                           ▼
                         Model Training & Evaluation
             (baselines · Ridge · Random Forest · PyTorch NN)
                    chronological 80/20 split per horizon
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                       ▼
              Local joblib registry                  Hopsworks Model Registry
                        │                                       │
                        └──────────────────┬────────────────────┘
                                           ▼
                                   FastAPI backend
                          /cities · /predict/{city} · /copilot/chat
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                       ▼
                Streamlit Dashboard                     AQI Copilot (Groq LLM)
      Overview · Comparison · Analytics · Copilot    tool-calling, grounded responses
```

Automation is handled by three GitHub Actions workflows: an hourly feature pipeline, a daily training pipeline, and a CI workflow (lint + tests) on every push/PR.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.11 |
| Data handling | pandas, NumPy |
| Classical ML | scikit-learn (Ridge Regression, Random Forest) |
| Deep learning | PyTorch (feed-forward neural network) |
| Explainability | SHAP 0.51.0 (primary), permutation importance (fallback) |
| Feature store / registry | Hopsworks 5.0.4 |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Visualization (EDA) | Matplotlib, Seaborn |
| LLM / Copilot | Groq API (`llama-3.3-70b-versatile`) |
| Automation | GitHub Actions |
| Deployment | FastAPI Cloud (API), Streamlit Community Cloud (dashboard) |

---

## Repository Structure

```
├── api/                        # Uvicorn entry point (compatibility shim for `api.main:app`)
├── config/
│   ├── cities.yaml              # City names, coordinates, enabled/trained flags
│   ├── features.yaml            # Feature/target definitions
│   └── model.yaml                # Model tuning grids, evaluation config
├── src/pearls_aqi/
│   ├── data/
│   │   ├── air_quality_provider.py   # Open-Meteo Air Quality client
│   │   ├── weather_provider.py       # Open-Meteo Forecast/Historical Weather client
│   │   ├── base_provider.py          # Shared HTTP request/retry logic
│   │   ├── cleaning.py               # Merge, sort, enrich, short-gap imputation
│   │   └── validation.py             # Schema/range/duplicate validation
│   ├── features/
│   │   ├── builder.py                # Calendar, lag, rolling feature construction
│   │   ├── targets.py                # 24h/48h/72h target construction
│   │   └── store.py                  # Hopsworks Feature Store + local/cloud loader
│   ├── models/
│   │   ├── baselines.py              # Persistence & seasonal-naive baselines
│   │   ├── sklearn_models.py         # Ridge & Random Forest wrappers
│   │   ├── torch_models.py           # PyTorch neural network models
│   │   ├── train.py                  # Training orchestration, tuning, champion selection
│   │   ├── split.py                  # Chronological train/test split
│   │   ├── evaluate.py               # MAE / RMSE / R² computation
│   │   ├── registry.py               # Local + Hopsworks model registry
│   │   └── explain.py                # SHAP + permutation importance
│   └── api/                    # FastAPI application (predict, copilot endpoints)
├── pipelines/
│   ├── backfill_pipeline.py     # Historical Open-Meteo backfill
│   ├── feature_pipeline.py      # Hourly ingestion + feature engineering
│   ├── training_pipeline.py     # Training, evaluation, registration
│   └── ablation_pipeline.py     # Pruning / regularization / blending experiments
├── dashboard/
│   └── app.py                    # Streamlit application
├── scripts/
│   └── run_eda.py                # Reproducible EDA artifact generation
├── artifacts/
│   ├── data/                     # Backfilled CSV datasets
│   ├── models/<city>/            # champion.joblib + champion.json per city
│   └── eda/                      # Generated EDA charts and summaries
├── tests/
│   ├── unit/                     # Fast, fully offline tests
│   ├── contract/                 # Open-Meteo API response shape tests
│   └── integration/               # Live Hopsworks/Open-Meteo tests (opt-in only)
├── .github/workflows/
│   ├── ci.yml                     # Lint + test on push/PR
│   ├── hourly_feature_pipeline.yml
│   └── daily_training_pipeline.yml
├── requirements.txt
└── pyproject.toml
```

---

## Data Sources

| Source | Endpoint | Provides |
|---|---|---|
| Open-Meteo Air Quality API | `air-quality-api.open-meteo.com` | US AQI, PM2.5, PM10, CO, NO₂, SO₂, ozone (CAMS global model) |
| Open-Meteo Weather Forecast API | `api.open-meteo.com` | Temperature, humidity, pressure, wind speed, precipitation (forecast) |
| Open-Meteo Historical Weather API | `archive-api.open-meteo.com` | Same weather variables, for backfill date ranges |

---

## Feature Engineering

Every hourly row, per city, is transformed into a feature vector before reaching any model:

| Category | Features |
|---|---|
| Calendar | hour, day, month, day of week, weekend flag |
| Cyclical encoding | sine/cosine of hour; sine/cosine of month |
| Current weather | temperature, relative humidity, surface pressure, wind speed, precipitation |
| Pollutants | PM2.5, PM10, CO, NO₂, SO₂, ozone, current US AQI |
| AQI lags | 1h, 3h, 6h, 12h, 24h, 48h, 72h |
| AQI change / rate | change rate and percent change at 1h, 3h, 24h |
| Rolling statistics | AQI mean and std at 3h, 6h, 12h, 24h windows |
| Forecast weather | Open-Meteo forecast fields (select 48h/72h model variants) |
| Targets | AQI at +24h, +48h, +72h (three independent supervised targets) |

**Leakage prevention:**
- Data sorted by city and UTC timestamp before any lag/target computation.
- Lags use strictly positive (backward) shifts; targets use negative (forward) shifts — both computed per city.
- Rolling windows apply `shift(1)` before rolling, excluding the current observation from its own statistic.
- A temporal leakage assertion rejects any row later than a supplied cutoff.
- Temporal features are recomputed across each city's **full continuous history** at training time (not per-batch), preventing lag columns from resetting at backfill batch boundaries.

---

## Modeling Methodology

- **Chronological 80/20 split** — earliest ~80% of each city's data trains the model, most recent ~20% is held out for testing. Never a random split, to avoid the model being evaluated on data chronologically "before" some of its training data.
- **Per-horizon, per-city champion selection** — each of the three forecast horizons is served by its own independently validated best model, rather than one model for everything, since different horizons behave as genuinely different problems.
- **Candidates compared:** persistence baseline, seasonal-naive baseline, Ridge Regression, Random Forest, PyTorch feed-forward neural network — plus refined variants (pruned features, stronger regularization, validation-weighted Ridge/RF blends).
- **Metrics:** MAE, RMSE, and R² are reported for every model and horizon.

### Key experiments

| Experiment | Outcome | Disposition |
|---|---|---|
| Forecasted-weather features (historical approximation) | Degraded Ridge/RF at 48h/72h | Reverted |
| Feature pruning (remove redundant lag/rolling columns) | Meaningfully improved neural network at 48h/72h | Kept |
| Stronger Ridge regularization | Meaningfully improved Ridge at 48h/72h | Kept |
| Validation-weighted Ridge/RF blend | Best 24h result; correctly weighted 0% Ridge at 48h/72h | Kept for 24h |
| Per-horizon champion selection | Worst-case R² improved from −9.47 to −0.06 | Adopted as standard |
| 1-year vs. 3-year backfill (controlled, identical holdout) | R² improved at every horizon, largest gain at 72h (+0.181) | Standardized on 3 years |

---

## Results

Current champion model and held-out metrics per city, per horizon:

| City | Horizon | Champion model | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Lahore | 24h | Ridge/RF blend (5%/95%) | 22.96 | 37.16 | 0.521 |
| Lahore | 48h | Pruned neural network | 32.43 | 49.10 | 0.149 |
| Lahore | 72h | Pruned neural network | 32.17 | 47.67 | 0.195 |
| Karachi | 24h | Ridge/RF blend (15%/85%) | 10.51 | 14.62 | 0.545 |
| Karachi | 48h | Ridge (α=1000) | 13.97 | 18.62 | 0.243 |
| Karachi | 72h | Ridge (α=1000) | 15.25 | 20.16 | 0.113 |
| Islamabad | 24h | Pruned neural network | 13.93 | 18.26 | 0.698 |
| Islamabad | 48h | Ridge (α=300) | 19.28 | 24.67 | 0.445 |
| Islamabad | 72h | Ridge (α=100) | 20.19 | 25.99 | 0.378 |
| Peshawar | 24h | Pruned neural network | 14.46 | 18.86 | 0.647 |
| Peshawar | 48h | Ridge/RF blend (55%/45%) | 18.18 | 23.51 | 0.450 |
| Peshawar | 72h | Ridge (α=100) | 19.19 | 24.67 | 0.391 |
| Quetta | 24h | Ridge (α=30) | 15.96 | 26.14 | 0.434 |
| Quetta | 48h | Pruned Ridge (α=30) | 20.77 | 32.60 | 0.122 |
| Quetta | 72h | Pruned Ridge (α=30) | 22.44 | 34.79 | 0.002 |
| Sargodha | 24h | Ridge/RF blend (90%/10%) | 19.65 | 26.10 | 0.687 |
| Sargodha | 48h | Ridge/RF blend (50%/50%) | 26.95 | 34.61 | 0.432 |
| Sargodha | 72h | Ridge (α=1000) | 28.58 | 36.37 | 0.352 |

*R² above 0 means the model beats a naive "predict the average" baseline; higher is better. No single algorithm wins everywhere — the champion type varies by city and horizon, supporting the per-horizon selection design.*

---

## Feature Store & Model Registry

| Resource | Name |
|---|---|
| Raw observation feature group | `aqi_observations_v1` |
| Engineered feature group | `aqi_features_hourly_v2` |
| Feature view | `aqi_features_fv_v2` |
| Model naming pattern | `pearls_aqi_predictor_<city>` |

Local development uses CSV artifacts and `artifacts/models/<city>/champion.{joblib,json}`. Hopsworks is used as the cloud feature store and model registry; Lahore is fully registered in both (currently model version 7), with the remaining cities registered locally and being rolled out to Hopsworks using the same methodology.

To force training against the live Hopsworks Feature View instead of local CSVs:

```bash
python pipelines/training_pipeline.py --city lahore --feature-store
```

---

## Explainability

The project uses **SHAP** as its primary explainability method, satisfying the brief's requirement, with **permutation importance** (built into scikit-learn) as an automatic fallback if SHAP is unavailable or incompatible with a given model.

- `shap_feature_importance()` — TreeExplainer for tree models, LinearExplainer for linear models; global feature ranking.
- `shap_local_explanation()` — explains one specific prediction in terms of each feature's contribution.
- `global_feature_importance()` — permutation-importance fallback.

Both global and local explanations are surfaced in the dashboard's Model Analytics tab, and `explain_prediction` is also exposed as an AQI Copilot tool.

---

## API Reference

Base URL (production): `https://aqi-predict-9a0f8674.fastapicloud.dev`

### `GET /cities`
Returns the list of enabled cities and their configuration.

### `GET /predict/{city}`
Returns current AQI plus 24h/48h/72h forecasts.

```json
{
  "city": "lahore",
  "observation_timestamp": "2026-08-30T14:00:00Z",
  "current_aqi": 160.0,
  "aqi_standard": "us_aqi",
  "data_label": "modeled",
  "model_name": "ridge_rf_blend",
  "model_version": 7,
  "forecasts": [
    { "horizon_hours": 24, "aqi": 149.3, "category": "Unhealthy for Sensitive Groups", "hazardous": false },
    { "horizon_hours": 48, "aqi": 164.5, "category": "Unhealthy", "hazardous": true },
    { "horizon_hours": 72, "aqi": 154.4, "category": "Unhealthy", "hazardous": true }
  ],
  "stale": false
}
```


### `POST /api/v1/copilot/chat`
Grounded, tool-calling conversational endpoint — see [AQI Copilot](#aqi-copilot).

---

## Dashboard

| Tab | Contents |
|---|---|
| **Overview** | City selector, current/24h/48h/72h AQI cards (color-coded by US AQI category), trend chart, pollutant snapshot, hazardous/stale-data alerts |
| **City Comparison** | Current + 24h/48h/72h forecasts across all trained cities, as a table and chart |
| **Model Analytics** | Saved metrics, current champion selection per horizon, SHAP/permutation feature importance, local explanations |
| **AQI Copilot** | Conversational interface with grounded, tool-backed answers |

Run locally:

```bash
streamlit run dashboard/app.py
```

---

## AQI Copilot

A grounded, tool-calling conversational assistant — every number it states comes from a real tool call against the live pipeline, **never** from the LLM's own general knowledge.

| Property | Value |
|---|---|
| Provider | Groq |
| Model | `llama-3.3-70b-versatile` |
| Default state | `COPILOT_ENABLED=false` (opt-in) |
| History window | Last 6 messages |
| Logging | Tool name, outcome, latency, correlation ID per call |

**Tools available:** `get_current_aqi`, `get_aqi_forecast`, `get_weather`, `get_pollutants`, `compare_cities`, `get_aqi_history`, `explain_prediction`.

**Safety behavior:**
- Refuses to invent data for unsupported cities; states plainly no model exists.
- Distinguishes current / historical / forecasted values and flags stale data explicitly.
- Resists prompt-injection attempts to override instructions or fabricate values.
- Runs crisis/self-harm risk detection as a priority check ahead of normal AQI routing; on detection, responds with real Pakistan crisis resources (e.g., Rescue 1122, Umang, National Youth Helpline) instead of proceeding with a forecast.

Example questions:
- *"What is Lahore's AQI forecast for the next three days?"*
- *"Why is Lahore's AQI predicted to be high tomorrow?"*
- *"Compare Lahore, Karachi, and Islamabad tomorrow."*
- *"Which city is forecast to improve the most over three days?"*

---

## Automation (CI/CD)

| Workflow | Schedule | Purpose |
|---|---|---|
| `hourly_feature_pipeline.yml` | `17 * * * *` | Runs the feature pipeline for all configured cities |
| `daily_training_pipeline.yml` | `35 1 * * *` | Retrains and re-registers champions for all supported cities |
| `ci.yml` | On push / PR | Installs dependencies, runs Ruff linting and pytest with coverage |

All scheduled workflows include `workflow_dispatch` (manual trigger) and concurrency guards. Required repository secrets: `HOPSWORKS_API_KEY`, `HOPSWORKS_HOST`, `HOPSWORKS_PROJECT`, and (once the Copilot is enabled in CI) `GROQ_API_KEY`.

**Why GitHub Actions over Jenkins/Airflow:** both alternatives require a continuously running server or hosted scheduler, which conflicts with this project's serverless requirement. GitHub Actions provisions temporary, on-demand runners with no infrastructure to manage.

---

## Local Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/Haseeb-py/pearls-aqi-predictor.git
cd pearls-aqi-predictor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

# 3. Install in editable mode
pip install -r requirements.txt
pip install -e .

# 4. Configure environment variables (see below)
cp .env.example .env             # then fill in real values

# 5. Backfill historical data for a city
python pipelines/backfill_pipeline.py --start-date 2022-08-01 --end-date 2025-08-01 --cities lahore

# 6. Train models
python pipelines/training_pipeline.py --city lahore

# 7. Run the API
uvicorn api.main:app --reload --port 8000

# 8. Run the dashboard (in a second terminal)
streamlit run dashboard/app.py
```

---

## Environment Variables

| Variable | Required for | Notes |
|---|---|---|
| `HOPSWORKS_API_KEY` | Feature Store / Model Registry | Never commit — set via `.env` locally, repo secrets in CI |
| `HOPSWORKS_HOST` | Feature Store / Model Registry | |
| `HOPSWORKS_PROJECT` | Feature Store / Model Registry | |
| `GROQ_API_KEY` | AQI Copilot | Only required when `COPILOT_ENABLED=true` |
| `COPILOT_ENABLED` | AQI Copilot | Defaults to `false` |

---

## Testing

```bash
# Fast, fully offline unit + contract tests
pytest -m "not integration" -q

# Include live Hopsworks/Open-Meteo integration tests (requires credentials)
pytest --run-integration -q
```

Coverage includes: feature/target leakage assertions, chronological split correctness, API error handling (404/503), hazardous-AQI alert thresholds, and Copilot tool-selection, off-topic handling, and crisis-detection regression tests.

---

## Design Decisions

- **Batch processing, not streaming** — AQI changes on the scale of hours, not seconds, and the data source itself is hourly; streaming infrastructure would add cost without benefit.
- **Per-horizon champion models** — different forecast horizons are different problems; forcing one model to serve all three measurably underperformed independent selection.
- **GitHub Actions over Jenkins/Airflow** — matches the serverless requirement without provisioning always-on infrastructure.
- **SHAP with a permutation-importance fallback** — satisfies the brief's explicit requirement while degrading gracefully if SHAP becomes incompatible with a future model.
- **A grounded Copilot, not a general-purpose chatbot** — restricted to allow-listed tools, explicitly instructed never to state a number that didn't come from a real tool call, with LLM-based (not keyword-based) intent classification for routing and crisis detection.

---

## Roadmap

- Extend Quetta's longer-horizon modeling with city-specific feature engineering.
- Extend Hopsworks Feature Store / Model Registry integration to all six cities.
- Integrate a genuine real-time forecast-weather feature into live serving.
- Add production monitoring and drift detection.
- Expand SHAP coverage across every city/model/horizon combination.

---

## License

Internal internship project. License terms to be confirmed with the project owner before external reuse.
