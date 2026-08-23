# Deployment

The production architecture has two serverless services:

- **FastAPI on Google Cloud Run**: API, feature retrieval, model-registry download, and Copilot.
- **Streamlit Community Cloud**: dashboard UI that calls the Cloud Run API.

Streamlit Community Cloud can host the dashboard, but it is not a replacement
for the FastAPI backend. Keeping them separate preserves the required FastAPI
service and allows the API to scale independently.

## 1. Deploy the API to Cloud Run

From the repository root, after installing and authenticating the Google Cloud
CLI:

```powershell
gcloud run deploy pearls-aqi-api --source . --region asia-south1 --allow-unauthenticated --set-env-vars "HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai,HOPSWORKS_PROJECT=YOUR_PROJECT,ALLOWED_ORIGINS=https://YOUR_APP.streamlit.app" --set-secrets "HOPSWORKS_API_KEY=HOPSWORKS_API_KEY:latest"
```

Create `HOPSWORKS_API_KEY` in Google Secret Manager before this command. Never
place the key in Git, Dockerfiles, or Streamlit source files. Copy the returned
Cloud Run URL and verify:

```powershell
Invoke-RestMethod "https://YOUR_CLOUD_RUN_URL/cities"
Invoke-RestMethod "https://YOUR_CLOUD_RUN_URL/predict/lahore"
```

## 2. Deploy the dashboard to Streamlit Community Cloud

1. Create an app from this GitHub repository.
2. Set the main file to `streamlit_app.py` and select Python 3.11.
3. In **Advanced settings → Secrets**, add:

```toml
API_BASE_URL = "https://YOUR_CLOUD_RUN_URL"
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "YOUR_PROJECT"
HOPSWORKS_API_KEY = "YOUR_HOPSWORKS_API_KEY"
```

The Hopsworks values let the dashboard obtain analytics/history and lazily
download a registered model when local artifacts are unavailable. The API URL
must not end in a slash.

## 3. Verify automation and serving

- Run **Hourly feature pipeline** from GitHub Actions; it writes the latest
  features to Hopsworks.
- Run **Daily model training** once manually; it retrains each city and uploads
  new model-registry versions.
- Open the deployed dashboard, check all six cities, the 24/48/72-hour cards,
  comparison view, SHAP analytics, alerts, and Copilot.

## Local production-like check

```powershell
$env:PYTHONPATH = "$PWD\src"
.\venv\Scripts\python.exe scripts\verify_hopsworks.py
```
