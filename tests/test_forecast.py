from datetime import date, datetime, timezone

import pytest

from src.forecast import (
    DailyForecast,
    WeatherForecast,
)


def test_daily_forecast_accepts_valid_values():
    forecast = DailyForecast(
        forecast_date=date(2026, 9, 28),
        rainfall_mm=12.5,
        precipitation_probability=0.7,
    )

    assert forecast.forecast_date == date(2026, 9, 28)
    assert forecast.rainfall_mm == 12.5
    assert forecast.precipitation_probability == 0.7


@pytest.mark.parametrize(
    "rainfall_mm",
    [-1.0, float("nan"), float("inf")],
)
def test_daily_forecast_rejects_invalid_rainfall(
    rainfall_mm,
):
    with pytest.raises(ValueError):
        DailyForecast(
            forecast_date=date(2026, 9, 28),
            rainfall_mm=rainfall_mm,
            precipitation_probability=0.5,
        )


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01, float("nan"), float("inf")],
)
def test_daily_forecast_rejects_invalid_probability(
    probability,
):
    with pytest.raises(ValueError):
        DailyForecast(
            forecast_date=date(2026, 9, 28),
            rainfall_mm=10.0,
            precipitation_probability=probability,
        )


def test_weather_forecast_accepts_chronological_entries():
    forecast = WeatherForecast(
        source="test-provider",
        issued_at=datetime(
            2026,
            9,
            28,
            6,
            0,
            tzinfo=timezone.utc,
        ),
        daily=(
            DailyForecast(
                forecast_date=date(2026, 9, 28),
                rainfall_mm=10.0,
                precipitation_probability=0.6,
            ),
            DailyForecast(
                forecast_date=date(2026, 9, 29),
                rainfall_mm=5.0,
                precipitation_probability=0.4,
            ),
        ),
    )

    assert len(forecast.daily) == 2


def test_weather_forecast_rejects_empty_entries():
    with pytest.raises(ValueError):
        WeatherForecast(
            source="test-provider",
            issued_at=datetime(
                2026,
                9,
                28,
                tzinfo=timezone.utc,
            ),
            daily=(),
        )


def test_weather_forecast_rejects_duplicate_dates():
    entry = DailyForecast(
        forecast_date=date(2026, 9, 28),
        rainfall_mm=10.0,
        precipitation_probability=0.6,
    )

    with pytest.raises(ValueError):
        WeatherForecast(
            source="test-provider",
            issued_at=datetime(
                2026,
                9,
                28,
                tzinfo=timezone.utc,
            ),
            daily=(entry, entry),
        )


def test_weather_forecast_rejects_out_of_order_dates():
    first = DailyForecast(
        forecast_date=date(2026, 9, 29),
        rainfall_mm=10.0,
        precipitation_probability=0.6,
    )

    second = DailyForecast(
        forecast_date=date(2026, 9, 28),
        rainfall_mm=5.0,
        precipitation_probability=0.4,
    )

    with pytest.raises(ValueError):
        WeatherForecast(
            source="test-provider",
            issued_at=datetime(
                2026,
                9,
                28,
                tzinfo=timezone.utc,
            ),
            daily=(first, second),
        )