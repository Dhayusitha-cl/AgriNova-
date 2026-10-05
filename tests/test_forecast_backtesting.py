from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.forecast import (
    DailyForecast,
    WeatherForecast,
)
from src.forecast_backtesting import (
    evaluate_forecast_wet_day_probability,
    validate_forecast_replay_timing,
)


def make_forecast(
    issued_at,
    start=date(2024, 6, 15),
    probabilities=(0.2, 0.8, 0.4),
):
    entries = tuple(
        DailyForecast(
            forecast_date=date(
                start.year,
                start.month,
                start.day + offset,
            ),
            rainfall_mm=5.0,
            precipitation_probability=probability,
        )
        for offset, probability in enumerate(
            probabilities
        )
    )

    return WeatherForecast(
        source="test",
        issued_at=issued_at,
        daily=entries,
    )


def test_forecast_replay_accepts_forecast_available_at_decision():
    forecast = make_forecast(
        datetime(2024, 6, 14, 12, 0)
    )

    assert validate_forecast_replay_timing(
        forecast=forecast,
        decision_timestamp=datetime(
            2024, 6, 14, 18, 0
        ),
        horizon_days=3,
    )


def test_forecast_replay_rejects_forecast_issued_after_decision():
    forecast = make_forecast(
        datetime(2024, 6, 15, 12, 0)
    )

    with pytest.raises(ValueError, match="issued after"):
        validate_forecast_replay_timing(
            forecast=forecast,
            decision_timestamp=datetime(
                2024, 6, 14, 18, 0
            ),
            horizon_days=3,
        )


def test_forecast_replay_rejects_missing_horizon_date():
    forecast = make_forecast(
        datetime(2024, 6, 14, 12, 0),
        probabilities=(0.2,),
    )

    with pytest.raises(ValueError, match="missing dates"):
        validate_forecast_replay_timing(
            forecast=forecast,
            decision_timestamp=datetime(
                2024, 6, 14, 18, 0
            ),
            horizon_days=3,
        )


def test_forecast_replay_rejects_mixed_timezone_awareness():
    forecast = make_forecast(
        datetime(
            2024,
            6,
            14,
            12,
            0,
            tzinfo=timezone.utc,
        )
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        validate_forecast_replay_timing(
            forecast=forecast,
            decision_timestamp=datetime(
                2024,
                6,
                14,
                18,
                0,
            ),
            horizon_days=3,
        )


def test_forecast_probability_evaluation_uses_one_mm_wet_threshold():
    forecast = make_forecast(
        datetime(2024, 6, 14, 12, 0)
    )

    actual_future = pd.DataFrame(
        {
            "date": [
                "2024-06-15",
                "2024-06-16",
                "2024-06-17",
            ],
            "rainfall_mm": [
                0.0,
                2.0,
                0.5,
            ],
        }
    )

    result = evaluate_forecast_wet_day_probability(
        forecast=forecast,
        actual_future=actual_future,
    )

    assert result["num_forecast_days"] == 3
    assert result["observed_wet_day_rate"] == pytest.approx(
        1 / 3
    )

    expected_brier = (
        (0.2 - 0.0) ** 2
        + (0.8 - 1.0) ** 2
        + (0.4 - 0.0) ** 2
    ) / 3

    assert result["brier_score"] == pytest.approx(
        expected_brier
    )


def test_forecast_probability_evaluation_ignores_non_overlapping_dates():
    forecast = make_forecast(
        datetime(2024, 6, 14, 12, 0)
    )

    actual_future = pd.DataFrame(
        {
            "date": [
                "2024-06-15",
                "2024-06-16",
                "2024-06-17",
                "2024-06-18",
            ],
            "rainfall_mm": [
                0.0,
                2.0,
                0.5,
                20.0,
            ],
        }
    )

    result = evaluate_forecast_wet_day_probability(
        forecast=forecast,
        actual_future=actual_future,
    )

    assert result["num_forecast_days"] == 3
