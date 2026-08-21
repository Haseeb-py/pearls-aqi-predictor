# ADR 0001: Selection and Audit of Open-Meteo & CAMS Air Quality Provider

* **Status:** Accepted
* **Date:** 2026-08-13
* **Decision:** Use Open-Meteo Weather Forecast, Historical Weather, and Air Quality (CAMS domain) APIs with US AQI standard.

## Context
The Pearls AQI Predictor requires weather and air quality data across Pakistani cities to forecast AQI for 24h, 48h, and 72h horizons.

## Decision
We select Open-Meteo APIs:
1. `https://api.open-meteo.com/v1/forecast` (weather forecast)
2. `https://archive-api.open-meteo.com/v1/archive` (historical weather)
3. `https://air-quality-api.open-meteo.com/v1/air-quality` (air quality, CAMS domain)

All pollutant concentrations and AQI values are calculated on the **US AQI** standard and labeled as **modeled air-quality data**.

## Validation Evidence
- Audit performed on Lahore for January 2025 (744 complete hours).
- Spot checks passed for Karachi, Islamabad, Peshawar, and Quetta.
- Zero credential requirements for API access.
