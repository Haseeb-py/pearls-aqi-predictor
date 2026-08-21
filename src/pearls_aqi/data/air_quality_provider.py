"""Open-Meteo Air Quality API provider adapter using global CAMS domain."""

from typing import Any, Dict

import pandas as pd

from pearls_aqi.data.base_provider import BaseProvider


class OpenMeteoAirQualityProvider(BaseProvider):
    """Adapter for Open-Meteo Air Quality API (global CAMS domain, US AQI)."""

    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    AQ_VARIABLES = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi",
    ]

    def fetch_historical_air_quality(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical air quality data (CAMS domain) for a given date range (YYYY-MM-DD)."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.AQ_VARIABLES),
            "domains": "cams_global",
            "timezone": "UTC",
        }
        data = self.fetch_json(self.AIR_QUALITY_URL, params)
        return self._parse_hourly_aq_response(data)

    def fetch_current_air_quality(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 7,
        forecast_days: int = 7,
    ) -> pd.DataFrame:
        """Fetch current, past, and forecast air quality data."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(self.AQ_VARIABLES),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "domains": "cams_global",
            "timezone": "UTC",
        }
        data = self.fetch_json(self.AIR_QUALITY_URL, params)
        return self._parse_hourly_aq_response(data)

    def _parse_hourly_aq_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df.rename(
            columns={
                "time": "event_time_utc",
                "pm2_5": "pm2_5_ug_m3",
                "pm10": "pm10_ug_m3",
                "carbon_monoxide": "carbon_monoxide_ug_m3",
                "nitrogen_dioxide": "nitrogen_dioxide_ug_m3",
                "sulphur_dioxide": "sulphur_dioxide_ug_m3",
                "ozone": "ozone_ug_m3",
                "us_aqi": "aqi",
            },
            inplace=True,
        )
        df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
        df["aqi_standard"] = "us_aqi"
        df["data_label"] = "modeled air-quality data"
        return df
