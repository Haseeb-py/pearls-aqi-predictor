# Open-Meteo Data Source Capability Report

**Date of Audit:** 2026-08-13  
**Target Region:** Pakistan (Lahore, Karachi, Islamabad, Peshawar, Quetta)  
**Standard Enforced:** US AQI (`us_aqi`)  
**Data Attribution & Labeling:** ECMWF CAMS global air quality reanalysis/forecast modeled data via Open-Meteo API.

## 1. Lahore 1-Month Audit (2025-01-01 to 2025-01-31)
- **Total Hourly Observations:** 744 hours (31 full days)
- **Completeness:** 100% (Null AQI: 0, Null PM2.5: 0)
- **US AQI Metrics:** Min: 134.0, Max: 303.0, Mean: 209.1
- **Variables Retained:**
  - Weather: `temperature_2m_c`, `relative_humidity_2m_pct`, `surface_pressure_hpa`, `wind_speed_10m_kph`, `precipitation_mm`
  - Air Quality: `pm2_5_ug_m3`, `pm10_ug_m3`, `carbon_monoxide_ug_m3`, `nitrogen_dioxide_ug_m3`, `sulphur_dioxide_ug_m3`, `ozone_ug_m3`, `us_aqi`

## 2. Multi-City Spot Checks
| City | Weather Rows | AQ Rows | Merged Rows | Sample US AQI | Status |
|---|---|---|---|---|---|
| Lahore | 744 | 744 | 744 | 206.0 | PASSED |
| Karachi | 48 | 72 | 72 | 70.0 | PASSED |
| Islamabad | 48 | 72 | 72 | 158.0 | PASSED |
| Peshawar | 48 | 72 | 72 | 151.0 | PASSED |
| Quetta | 48 | 72 | 72 | 84.0 | PASSED |

## 3. Findings & Limitations
1. **API Rate Limits & Authentication:** Open-Meteo APIs are free, require no API key for non-commercial evaluation, and provide high rate-limit capacity.
2. **Data Labeling:** All air quality observations originate from CAMS global gridded numerical atmospheric models and are explicitly labeled as **modeled air-quality data**.
3. **Data Contract Compliance:** All requested weather and pollutant features align with the requirements.
