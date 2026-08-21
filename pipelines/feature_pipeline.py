"""Hourly Feature Ingestion Pipeline CLI."""

import argparse

import pandas as pd

from pearls_aqi.data.air_quality_provider import OpenMeteoAirQualityProvider
from pearls_aqi.data.cleaning import merge_and_clean_city_data
from pearls_aqi.data.validation import validate_observation_df
from pearls_aqi.data.weather_provider import OpenMeteoWeatherProvider
from pearls_aqi.features.builder import build_features
from pearls_aqi.features.store import upsert_features_df
from pearls_aqi.features.targets import build_targets
from pearls_aqi.settings import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Pearls AQI Hourly Feature Pipeline")
    parser.add_argument(
        "--cities",
        type=str,
        default="all",
        help="Comma-separated city slugs or 'all'",
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=3,
        help="Number of past days to fetch for rolling/lag calculations",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=3,
        help="Number of forecast days to fetch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and print summary without writing to Hopsworks",
    )
    return parser.parse_args()


def run_feature_pipeline(
    cities_str: str = "all",
    past_days: int = 3,
    forecast_days: int = 3,
    dry_run: bool = False,
):
    print("=== Running Hourly Feature Pipeline ===")

    cities_config = settings.load_cities_config().get("cities", [])
    if cities_str == "all":
        target_cities = [c for c in cities_config if c.get("enabled", True)]
    else:
        requested_slugs = [s.strip().lower() for s in cities_str.split(",")]
        target_cities = [c for c in cities_config if c.get("slug") in requested_slugs]

    if not target_cities:
        print("No matching enabled cities found.")
        return pd.DataFrame()

    weather_provider = OpenMeteoWeatherProvider()
    aq_provider = OpenMeteoAirQualityProvider()

    all_city_dfs = []
    for city in target_cities:
        slug = city["slug"]
        lat = city["latitude"]
        lon = city["longitude"]
        print(f"Ingesting latest features for {city['name']} ({slug})...")

        w_df = weather_provider.fetch_forecast_weather(lat, lon, forecast_days=forecast_days)
        a_df = aq_provider.fetch_current_air_quality(lat, lon, past_days=past_days, forecast_days=forecast_days)

        if w_df.empty and a_df.empty:
            print(f"  Warning: No data returned for {slug}")
            continue

        merged_df = merge_and_clean_city_data(w_df, a_df, slug, lat, lon)
        validated_df = validate_observation_df(merged_df)

        feat_df = build_features(validated_df)
        full_df = build_targets(feat_df)
        all_city_dfs.append(full_df)
        print(f"  -> Processed {len(full_df)} feature rows for {slug}")

    if not all_city_dfs:
        print("No feature data generated.")
        return pd.DataFrame()

    combined_df = pd.concat(all_city_dfs, ignore_index=True)
    print(f"\nTotal hourly feature records prepared: {len(combined_df)}")

    if not dry_run:
        print("Upserting features to Hopsworks Feature Store...")
        count, msg = upsert_features_df(combined_df)
        print(f"Hopsworks Result: {msg} ({count} records)")
    else:
        print("[DRY RUN COMPLETE] No remote writes executed.")

    return combined_df


def main():
    args = parse_args()
    run_feature_pipeline(
        cities_str=args.cities,
        past_days=args.past_days,
        forecast_days=args.forecast_days,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
