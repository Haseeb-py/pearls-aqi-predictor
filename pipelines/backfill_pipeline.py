"""Idempotent historical backfill pipeline CLI."""

import argparse
from datetime import date, timedelta

import pandas as pd

from pearls_aqi.data.air_quality_provider import OpenMeteoAirQualityProvider
from pearls_aqi.data.cleaning import merge_and_clean_city_data
from pearls_aqi.data.validation import validate_observation_df
from pearls_aqi.domain.exceptions import DataValidationError
from pearls_aqi.data.weather_provider import OpenMeteoWeatherProvider
from pearls_aqi.features.builder import build_features
from pearls_aqi.features.store import upsert_features_df
from pearls_aqi.features.targets import build_targets
from pearls_aqi.settings import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Pearls AQI Historical Backfill Pipeline")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default="all",
        help="Comma-separated city slugs or 'all' (e.g. karachi,lahore)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and print summary without writing to Feature Store",
    )
    parser.add_argument(
        "--no-save-local",
        action="store_false",
        dest="save_local",
        default=True,
        help="Do not save engineered features to artifacts/data/",
    )
    parser.add_argument(
        "--upsert-hopsworks",
        action="store_true",
        help="Upsert engineered features to Hopsworks Feature Store",
    )
    return parser.parse_args()


def run_backfill(
    start_date: str,
    end_date: str,
    cities_str: str = "all",
    dry_run: bool = False,
    save_local: bool = True,
    upsert_hopsworks: bool = False,
):
    print(f"=== Running Backfill Pipeline: {start_date} to {end_date} ===")

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
        print(f"Fetching data for {city['name']} ({slug})...")

        w_df = weather_provider.fetch_historical_weather(lat, lon, start_date, end_date)
        a_df = aq_provider.fetch_historical_air_quality(lat, lon, start_date, end_date)

        if w_df.empty and a_df.empty:
            print(f"  Warning: No data returned for {slug}")
            continue

        merged_df = merge_and_clean_city_data(w_df, a_df, slug, lat, lon)
        # Request archived forecasts at their valid time, then align them back
        # to the issuance time.  No realised target-time weather is shifted.
        for lead_days in (2, 3):
            offset = timedelta(days=lead_days)
            forecast_df = weather_provider.fetch_previous_run_weather(
                lat,
                lon,
                (date.fromisoformat(start_date) + offset).isoformat(),
                (date.fromisoformat(end_date) + offset).isoformat(),
                lead_days,
            )
            forecast_df["event_time_utc"] = forecast_df["event_time_utc"] - pd.to_timedelta(lead_days * 24, unit="h")
            merged_df = merged_df.merge(forecast_df, on="event_time_utc", how="left")
        try:
            validated_df = validate_observation_df(merged_df)
        except DataValidationError as exc:
            ozone_values = merged_df.get("ozone_ug_m3", pd.Series(dtype=float))
            ozone = merged_df.loc[(ozone_values < 0) | (ozone_values > 5000), ["event_time_utc", "ozone_ug_m3"]]
            if not ozone.empty:
                raise DataValidationError(
                    f"{exc} Offending ozone rows: {ozone.head(5).to_dict(orient='records')}"
                ) from exc
            raise

        feat_df = build_features(validated_df)
        full_df = build_targets(feat_df)
        all_city_dfs.append(full_df)
        print(f"  -> Processed {len(full_df)} feature rows for {slug}")

    if not all_city_dfs:
        print("No feature data generated.")
        return pd.DataFrame()

    combined_df = pd.concat(all_city_dfs, ignore_index=True)
    print(f"\nTotal combined feature rows across {len(target_cities)} cities: {len(combined_df)}")

    if save_local or dry_run:
        output_dir = settings.BASE_DIR / "artifacts" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"backfill_{start_date}_to_{end_date}.csv"
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path, parse_dates=["event_time_utc"])
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True).drop_duplicates(
                ["city_slug", "event_time_utc"], keep="last"
            )
        combined_df.to_csv(csv_path, index=False)
        print(f"Saved local artifact to: {csv_path}")

    if upsert_hopsworks and not dry_run:
        print("\nUpserting to Hopsworks Feature Store...")
        count, msg = upsert_features_df(combined_df)
        print(f"Hopsworks Status: {msg} ({count} rows)")

    if dry_run:
        print("\n[DRY RUN COMPLETE] No remote writes executed.")

    return combined_df


def main():
    args = parse_args()
    run_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        cities_str=args.cities,
        dry_run=args.dry_run,
        save_local=args.save_local,
        upsert_hopsworks=args.upsert_hopsworks,
    )


if __name__ == "__main__":
    main()
