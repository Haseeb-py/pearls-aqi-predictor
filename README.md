# Pearls AQI Predictor

End-to-end three-day AQI forecasting for six Pakistani cities: Karachi, Lahore,
Islamabad, Peshawar, Quetta, and Sargodha.

The system uses Open-Meteo/CAMS weather and air-quality data, Hopsworks Feature
Store and Model Registry, Ridge/Random Forest/PyTorch forecasting models,
FastAPI, Streamlit, SHAP, and GitHub Actions automation.

## Run locally

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\src"
.\venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
.\venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

See [deployment instructions](docs/deployment.md) and the
[final report](docs/final_report.md) for architecture, evaluation, and CI/CD.
