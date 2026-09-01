"""Test selection defaults that keep live services opt-in."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that call external services",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="integration test; pass --run-integration to enable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

@pytest.fixture(autouse=True)
def disable_live_groq_by_default(monkeypatch):
    """Unit tests must not call the developer's configured Groq account."""
    from pearls_aqi import copilot

    monkeypatch.setattr(copilot.settings, "APP_ENV", "development")
    monkeypatch.setattr(copilot.settings, "GROQ_API_KEY", None)
