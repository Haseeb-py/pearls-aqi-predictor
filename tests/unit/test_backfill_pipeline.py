"""Unit tests for Backfill Pipeline CLI."""

import pytest

from pipelines.backfill_pipeline import run_backfill


@pytest.mark.integration
def test_run_backfill_dry_run():
    df = run_backfill(
        start_date="2025-01-01",
        end_date="2025-01-02",
        cities_str="lahore",
        dry_run=True,
        save_local=False,
        upsert_hopsworks=False,
    )
    assert not df.empty
    assert "city_slug" in df.columns
    assert "target_aqi_24h" in df.columns
    assert (df["city_slug"] == "lahore").all()
