"""Pydantic schemas for data validation and API response contracts."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, field_validator


class SourceRecordType(str, Enum):
    OBSERVATION = "observation"
    REANALYSIS = "reanalysis"
    FORECAST = "forecast"


class QualityFlag(str, Enum):
    VALID = "valid"
    IMPUTED = "imputed"
    SUSPECT = "suspect"


class CityConfig(BaseModel):
    name: str
    slug: str
    province: Optional[str] = None
    territory: Optional[str] = None
    latitude: float
    longitude: float
    timezone: str = "Asia/Karachi"
    enabled: bool = True


class RawObservation(BaseModel):
    city_slug: str
    event_time_utc: datetime
    ingested_at_utc: datetime
    source_name: str
    source_record_type: SourceRecordType
    latitude: float
    longitude: float
    temperature_2m_c: Optional[float] = None
    relative_humidity_2m_pct: Optional[float] = None
    surface_pressure_hpa: Optional[float] = None
    wind_speed_10m_kph: Optional[float] = None
    precipitation_mm: Optional[float] = None
    pm2_5_ug_m3: Optional[float] = None
    pm10_ug_m3: Optional[float] = None
    carbon_monoxide_ug_m3: Optional[float] = None
    nitrogen_dioxide_ug_m3: Optional[float] = None
    sulphur_dioxide_ug_m3: Optional[float] = None
    ozone_ug_m3: Optional[float] = None
    aqi: float
    aqi_standard: str = "us_aqi"
    quality_flag: QualityFlag = QualityFlag.VALID
    data_label: str = "modeled air-quality data"

    @field_validator("aqi")
    @classmethod
    def validate_aqi_range(cls, v: float) -> float:
        if v < 0 or v > 1000:
            raise ValueError(f"AQI value {v} outside plausible range [0, 1000]")
        return v


class ForecastHorizonOutput(BaseModel):
    horizon_hours: int
    valid_at_utc: datetime
    aqi: float
    category: str
    is_hazardous: bool


class ForecastResponse(BaseModel):
    city: str
    issued_at_utc: datetime
    latest_observation_at_utc: datetime
    current_aqi: float
    aqi_standard: str = "us_aqi"
    data_label: str = "modeled air-quality data"
    model_name: str
    model_version: int
    forecasts: List[ForecastHorizonOutput]
    is_stale: bool = False


class CopilotChatRequest(BaseModel):
    message: str
    cities: List[str] = []
    history: List[str] = []


class CopilotChatResponse(BaseModel):
    answer: str
    tools_used: List[str]
    evidence: dict
    provider: str
    correlation_id: str = ""
    generated_at_utc: Optional[datetime] = None
    tool_events: List[dict] = []
