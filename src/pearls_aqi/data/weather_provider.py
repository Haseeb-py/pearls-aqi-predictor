"""Open-Meteo Weather Forecast and Historical API provider adapters."""

from typing import Any, Dict

import pandas as pd

from pearls_aqi.data.base_provider import BaseProvider


class OpenMeteoWeatherProvider(BaseProvider):
    """Adapter for Open-Meteo Weather Forecast and Historical APIs."""

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

    WEATHER_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "precipitation",
    ]

    def fetch_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical weather data for a given lat/lon and date range (YYYY-MM-DD)."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.WEATHER_VARIABLES),
            "timezone": "UTC",
        }
        data = self.fetch_json(self.HISTORICAL_URL, params)
        return self._parse_hourly_weather_response(data)

    def fetch_forecast_weather(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 7,
    ) -> pd.DataFrame:
        """Fetch current and forecast weather data."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(self.WEATHER_VARIABLES),
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        data = self.fetch_json(self.FORECAST_URL, params)
        return self._parse_hourly_weather_response(data)

    def fetch_previous_run_weather(
        self, latitude: float, longitude: float, start_date: str, end_date: str, lead_days: int
    ) -> pd.DataFrame:
        """Fetch weather predicted ``lead_days`` before each valid timestamp.

        Open-Meteo's Previous Runs API provides archived forecast values, rather
        than realised weather shifted into the future.  This makes the columns
        suitable for an issuance-time forecasting backtest.
        """
        if lead_days not in (2, 3):
            raise ValueError("Only 2- and 3-day forecast weather is supported.")
        variables = [f"{name}_previous_day{lead_days}" for name in self.WEATHER_VARIABLES]
        data = self.fetch_json(
            self.PREVIOUS_RUNS_URL,
            {
                "latitude": latitude, "longitude": longitude,
                "start_date": start_date, "end_date": end_date,
                "hourly": ",".join(variables), "timezone": "UTC",
            },
        )
        df = self._parse_hourly_weather_response(data)
        rename = {
            f"temperature_2m_previous_day{lead_days}": f"forecast_temperature_2m_c_{lead_days * 24}h",
            f"relative_humidity_2m_previous_day{lead_days}": f"forecast_relative_humidity_2m_pct_{lead_days * 24}h",
            f"surface_pressure_previous_day{lead_days}": f"forecast_surface_pressure_hpa_{lead_days * 24}h",
            f"wind_speed_10m_previous_day{lead_days}": f"forecast_wind_speed_10m_kph_{lead_days * 24}h",
            f"precipitation_previous_day{lead_days}": f"forecast_precipitation_mm_{lead_days * 24}h",
        }
        return df.rename(columns=rename)

    def _parse_hourly_weather_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df.rename(
            columns={
                "time": "event_time_utc",
                "temperature_2m": "temperature_2m_c",
                "relative_humidity_2m": "relative_humidity_2m_pct",
                "surface_pressure": "surface_pressure_hpa",
                "wind_speed_10m": "wind_speed_10m_kph",
                "precipitation": "precipitation_mm",
            },
            inplace=True,
        )
        df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
        return df
