"""Unit tests for leakage prevention controls."""

import pandas as pd
import pytest

from pearls_aqi.domain.exceptions import DataLeakageError
from pearls_aqi.features.targets import assert_no_future_leakage


def test_assert_no_future_leakage_valid():
    times = pd.date_range("2025-01-01 00:00:00", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"event_time_utc": times})
    cutoff = pd.Timestamp("2025-01-01 23:00:00", tz="UTC")

    assert assert_no_future_leakage(df, cutoff) is True


def test_assert_no_future_leakage_detected():
    times = pd.date_range("2025-01-01 00:00:00", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"event_time_utc": times})
    cutoff = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")

    with pytest.raises(DataLeakageError, match="Detected 11 rows"):
        assert_no_future_leakage(df, cutoff)
