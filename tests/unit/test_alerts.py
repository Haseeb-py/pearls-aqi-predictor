from pearls_aqi.domain.aqi_categories import requires_health_alert


def test_unhealthy_forecast_requires_alert():
    assert not requires_health_alert(150)
    assert requires_health_alert(151)
    assert requires_health_alert(350)
