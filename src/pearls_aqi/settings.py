"""Typed settings management using Pydantic Settings."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    DEFAULT_TIMEZONE: str = Field(default="Asia/Karachi")

    HOPSWORKS_HOST: str = Field(default="eu-west.cloud.hopsworks.ai")
    HOPSWORKS_PROJECT: Optional[str] = Field(default=None)
    HOPSWORKS_API_KEY: Optional[str] = Field(default=None)

    WEATHER_PROVIDER: str = Field(default="open_meteo")
    AIR_QUALITY_PROVIDER: str = Field(default="open_meteo")
    AQI_STANDARD: str = Field(default="us_aqi")

    MODEL_NAME: str = Field(default="pearls_aqi_predictor")
    MODEL_VERSION: int = Field(default=1)

    API_BASE_URL: str = Field(default="http://localhost:8000")
    ALLOWED_ORIGINS: str = Field(default="http://localhost:8501")

    LLM_PROVIDER: str = Field(default="groq")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GROQ_API_KEY: Optional[str] = Field(default=None)
    COPILOT_ENABLED: bool = Field(default=True)

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CONFIG_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "config")

    def load_cities_config(self) -> dict:
        cities_path = self.CONFIG_DIR / "cities.yaml"
        if not cities_path.exists():
            raise FileNotFoundError(f"Cities config file not found at {cities_path}")
        with open(cities_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_features_config(self) -> dict:
        features_path = self.CONFIG_DIR / "features.yaml"
        if not features_path.exists():
            raise FileNotFoundError(f"Features config file not found at {features_path}")
        with open(features_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_model_config(self) -> dict:
        with open(self.CONFIG_DIR / "model.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


settings = Settings()
