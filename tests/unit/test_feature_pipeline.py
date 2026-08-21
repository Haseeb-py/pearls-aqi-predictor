"""Unit tests for Hourly Feature Pipeline CLI."""

import pytest

from pipelines.feature_pipeline import run_feature_pipeline


@pytest.mark.integration
def test_run_feature_pipeline_dry_run():
    df = run_feature_pipeline(
        cities_str="karachi",
        past_days=1,
        forecast_days=1,
        dry_run=True,
    )
    assert not df.empty
    assert "city_slug" in df.columns
    assert (df["city_slug"] == "karachi").all()
