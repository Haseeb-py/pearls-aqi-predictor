import pytest


fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from pearls_aqi.api.main import app


def test_cities_returns_enabled_configured_cities():
    response = TestClient(app).get("/cities")

    assert response.status_code == 200
    assert any(city["slug"] == "lahore" for city in response.json())


def test_predict_unknown_city_returns_clear_404():
    response = TestClient(app).get("/predict/not-a-supported-city")

    assert response.status_code == 404
    assert "Unknown city" in response.json()["detail"]
