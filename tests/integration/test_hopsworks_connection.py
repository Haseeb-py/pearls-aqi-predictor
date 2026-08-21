"""Integration test verifying Hopsworks connection from environment variables."""

import pytest

from pearls_aqi.features.store import verify_hopsworks_connection
from pearls_aqi.settings import settings


@pytest.mark.integration
def test_hopsworks_connection_from_env():
    """Verify Hopsworks authentication from environment variables without logging API key."""
    if not settings.HOPSWORKS_API_KEY or not settings.HOPSWORKS_PROJECT:
        pytest.skip("Hopsworks environment variables HOPSWORKS_API_KEY or HOPSWORKS_PROJECT not present.")

    success = verify_hopsworks_connection()
    assert success is True, "Hopsworks connection check failed."
