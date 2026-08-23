# Deployment

The production architecture has two serverless services:

- **FastAPI on Google Cloud Run**: API, feature retrieval, model-registry download, and Copilot.
- **Streamlit Community Cloud**: dashboard UI that calls the Cloud Run API.

Streamlit Community Cloud can host the dashboard, but it is not a replacement
for the FastAPI backend. Keeping them separate preserves the required FastAPI
service and allows the API to scale independently.

## Option A: Deploy the API to Cloud Run

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

## Option B: Deploy the API to Render (no Google Cloud required)

Create a **Web Service** at [Render](https://render.com) from this GitHub
repository. Use these exact values:

| Render field | Value |
|---|---|
| Language | Python 3 |
| Branch | `main` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `PYTHONPATH=src uvicorn pearls_aqi.api.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/cities` |
| Instance Type | Free (for demo) |

Add these environment variables in Render's **Environment** page:

```text
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=YOUR_PROJECT
HOPSWORKS_API_KEY=YOUR_HOPSWORKS_API_KEY
ALLOWED_ORIGINS=https://YOUR_APP.streamlit.app
```

Do not commit these values. Render uses the `.python-version` file to select
Python 3.11. After deployment, copy the public `https://...onrender.com` URL
and use it as `API_BASE_URL` in the Streamlit Community Cloud secrets.

Render's free web service has 512 MB RAM and sleeps after 15 inactive minutes,
so the first request can take around a minute. It is appropriate for a project
demo, not high-availability production traffic.
