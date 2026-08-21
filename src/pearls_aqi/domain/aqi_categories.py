"""US AQI standard categorization and health alert boundaries."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AQICategoryInfo:
    category: str
    min_aqi: int
    max_aqi: int
    color_hex: str
    health_concern: str
    guidance: str
    is_hazardous: bool


US_AQI_CATEGORIES: Dict[str, AQICategoryInfo] = {
    "Good": AQICategoryInfo(
        category="Good",
        min_aqi=0,
        max_aqi=50,
        color_hex="#00E400",
        health_concern="Good",
        guidance="Air quality is satisfactory, and air pollution poses little or no risk.",
        is_hazardous=False,
    ),
    "Moderate": AQICategoryInfo(
        category="Moderate",
        min_aqi=51,
        max_aqi=100,
        color_hex="#FFFF00",
        health_concern="Moderate",
        guidance="Air quality is acceptable; however, sensitive individuals may experience minor symptoms.",
        is_hazardous=False,
    ),
    "Unhealthy for Sensitive Groups": AQICategoryInfo(
        category="Unhealthy for Sensitive Groups",
        min_aqi=101,
        max_aqi=150,
        color_hex="#FF7E00",
        health_concern="Unhealthy for Sensitive Groups",
        guidance="Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        is_hazardous=False,
    ),
    "Unhealthy": AQICategoryInfo(
        category="Unhealthy",
        min_aqi=151,
        max_aqi=200,
        color_hex="#FF0000",
        health_concern="Unhealthy",
        guidance="Some members of the general public may experience health effects; sensitive groups may experience more serious effects.",
        is_hazardous=False,
    ),
    "Very Unhealthy": AQICategoryInfo(
        category="Very Unhealthy",
        min_aqi=201,
        max_aqi=300,
        color_hex="#8F3F97",
        health_concern="Very Unhealthy",
        guidance="Health alert: The risk of health effects is increased for everyone.",
        is_hazardous=False,
    ),
    "Hazardous": AQICategoryInfo(
        category="Hazardous",
        min_aqi=301,
        max_aqi=500,
        color_hex="#7E0023",
        health_concern="Hazardous",
        guidance="Health warning of emergency conditions: Everyone is more likely to be affected.",
        is_hazardous=True,
    ),
}


def get_us_aqi_category(aqi_value: float) -> AQICategoryInfo:
    """Return the US AQI category for a given numerical AQI value."""
    if aqi_value < 0:
        aqi_value = 0.0
    val = round(aqi_value)

    if val <= 50:
        return US_AQI_CATEGORIES["Good"]
    elif val <= 100:
        return US_AQI_CATEGORIES["Moderate"]
    elif val <= 150:
        return US_AQI_CATEGORIES["Unhealthy for Sensitive Groups"]
    elif val <= 200:
        return US_AQI_CATEGORIES["Unhealthy"]
    elif val <= 300:
        return US_AQI_CATEGORIES["Very Unhealthy"]
    else:
        return US_AQI_CATEGORIES["Hazardous"]


def requires_health_alert(aqi_value: float) -> bool:
    """Return whether a forecast needs the dashboard health warning."""
    return get_us_aqi_category(aqi_value).min_aqi >= 151
