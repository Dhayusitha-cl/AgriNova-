"""
Historical forecast replay and forecast-conditioned backtesting.

This module evaluates forecast evidence using only information that
would have been available at the historical decision timestamp.

It intentionally does not calibrate forecast_weight automatically.
Historical forecast archives are required before such calibration is
meaningful.
"""

from datetime import datetime

import math
import pandas as pd

from src.backtesting import (
    calculate_actual_future_total,
    calculate_monte_carlo_total_distribution,
    create_backtest_split,
    evaluate_actual_against_distribution,
    get_initial_state,
)
from src.forecast import WeatherForecast
from src.monte_carlo_weather import (
    generate_calibrated_monte_carlo_scenarios,
)


WET_RAINFALL_THRESHOLD_MM = 1.0


def _validate_decision_timestamp(decision_timestamp):
    if not isinstance(decision_timestamp, datetime):
        raise TypeError(
            "decision_timestamp must be a datetime."
        )

    return decision_timestamp


def _validate_forecast_issue_time(
    forecast,
    decision_timestamp,
):
    """
    Ensure the forecast existed at the historical decision moment.

    Naive and timezone-aware datetimes must not be mixed because doing
    so would make temporal ordering ambiguous.
    """

    issued_at = forecast.issued_at

    decision_is_aware = (
        decision_timestamp.tzinfo is not None
        and decision_timestamp.utcoffset() is not None
    )

    issued_is_aware = (
        issued_at.tzinfo is not None
        and issued_at.utcoffset() is not None
    )

    if decision_is_aware != issued_is_aware:
        raise ValueError(
            "decision_timestamp and forecast.issued_at must both be "
            "timezone-aware or both be timezone-naive."
        )

    if issued_at > decision_timestamp:
        raise ValueError(
            "Forecast was issued after the historical decision "
            "timestamp and cannot be used in replay."
        )


def validate_forecast_replay_timing(
    forecast,
    decision_timestamp,
    horizon_days,
):
    """
    Validate that a historical forecast was available at the
    decision time and covers the complete simulation horizon.
    """

    if not isinstance(
        forecast,
        WeatherForecast,
    ):
        raise TypeError(
            "forecast must be a WeatherForecast instance."
        )

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    decision_timestamp = _validate_decision_timestamp(
        decision_timestamp
    )

    _validate_forecast_issue_time(
        forecast,
        decision_timestamp,
    )

    decision_date = decision_timestamp.date()

    required_dates = {
        timestamp.date()
        for timestamp in pd.date_range(
            decision_date,
            periods=horizon_days,
            freq="D",
        )
    }

    # The initial state is fixed on the decision date, so the
    # forecast is consumed only from the following simulation day.
    required_dates.discard(decision_date)

    forecast_dates = {
        entry.forecast_date
        for entry in forecast.daily
    }

    missing_dates = required_dates - forecast_dates

    if missing_dates:
        raise ValueError(
            "Forecast is missing dates required by the replay "
            f"horizon: {sorted(missing_dates)}"
        )

    return True


def evaluate_forecast_wet_day_probability(
    forecast,
    actual_future,
):
    """
    Evaluate forecast precipitation probabilities against
    subsequently observed wet/dry outcomes.

    A wet day is defined as rainfall >= 1 mm.

    Returns descriptive probability-forecast metrics.
    """

    if not isinstance(
        forecast,
        WeatherForecast,
    ):
        raise TypeError(
            "forecast must be a WeatherForecast instance."
        )

    if not isinstance(
        actual_future,
        pd.DataFrame,
    ):
        raise TypeError(
            "actual_future must be a pandas DataFrame."
        )

    required_columns = {
        "date",
        "rainfall_mm",
    }

    missing = (
        required_columns
        - set(actual_future.columns)
    )

    if missing:
        raise ValueError(
            "actual_future is missing required columns: "
            f"{sorted(missing)}"
        )

    actual = actual_future.copy()
    actual["date"] = pd.to_datetime(
        actual["date"]
    )

    forecast_by_date = {
        entry.forecast_date: entry
        for entry in forecast.daily
    }

    probabilities = []
    observations = []

    for _, row in actual.iterrows():
        observation_date = row["date"].date()

        forecast_entry = forecast_by_date.get(
            observation_date
        )

        if forecast_entry is None:
            continue

        rainfall_mm = float(
            row["rainfall_mm"]
        )

        if not math.isfinite(rainfall_mm):
            raise ValueError(
                "Actual rainfall must be finite."
            )

        if rainfall_mm < 0:
            raise ValueError(
                "Actual rainfall cannot be negative."
            )

        probabilities.append(
            float(
                forecast_entry
                .precipitation_probability
            )
        )

        observations.append(
            float(
                rainfall_mm
                >= WET_RAINFALL_THRESHOLD_MM
            )
        )

    if not probabilities:
        raise ValueError(
            "No overlapping forecast and actual observation "
            "dates were found."
        )

    probability_array = pd.Series(
        probabilities,
        dtype=float,
    )

    observation_array = pd.Series(
        observations,
        dtype=float,
    )

    brier_score = float(
        (
            probability_array
            - observation_array
        ).pow(2).mean()
    )

    return {
        "num_forecast_days": int(
            len(probabilities)
        ),
        "mean_forecast_probability": float(
            probability_array.mean()
        ),
        "observed_wet_day_rate": float(
            observation_array.mean()
        ),
        "brier_score": brier_score,
    }


def run_forecast_conditioned_backtest(
    dataframe,
    forecast,
    decision_timestamp,
    calibration_artifact,
    horizon_days=14,
    num_simulations=1000,
    forecast_weight=0.0,
    random_seed=42,
):
    """
    Run a leakage-safe forecast-conditioned historical replay.

    Historical rainfall before the decision date is used for the
    climate state/model. Future rainfall is held out. The forecast
    is accepted only if it existed at the historical decision
    timestamp.

    forecast_weight is supplied explicitly. It is not calibrated
    automatically by this function.
    """

    if not isinstance(
        forecast,
        WeatherForecast,
    ):
        raise TypeError(
            "forecast must be a WeatherForecast instance."
        )

    decision_timestamp = _validate_decision_timestamp(
        decision_timestamp
    )

    validate_forecast_replay_timing(
        forecast=forecast,
        decision_timestamp=decision_timestamp,
        horizon_days=horizon_days,
    )

    decision_date = decision_timestamp.date()

    backtest = create_backtest_split(
        dataframe=dataframe,
        decision_date=decision_date,
        horizon_days=horizon_days,
    )

    training_data = backtest["training_data"]
    actual_future = backtest["actual_future"]

    initial_state = get_initial_state(
        dataframe,
        decision_date,
    )

    scenarios = (
        generate_calibrated_monte_carlo_scenarios(
            start_date=decision_date,
            num_days=horizon_days,
            num_simulations=num_simulations,
            initial_state=initial_state,
            random_seed=random_seed,
            calibration_artifact=calibration_artifact,
            forecast=forecast,
            forecast_weight=forecast_weight,
        )
    )

    actual_total = calculate_actual_future_total(
        actual_future
    )

    simulated_totals = (
        calculate_monte_carlo_total_distribution(
            scenarios
        )
    )

    rainfall_evaluation = (
        evaluate_actual_against_distribution(
            actual_total_mm=actual_total,
            simulated_totals=simulated_totals,
        )
    )

    forecast_evaluation = (
        evaluate_forecast_wet_day_probability(
            forecast=forecast,
            actual_future=actual_future,
        )
    )

    return {
        "decision_timestamp": decision_timestamp,
        "decision_date": pd.Timestamp(
            decision_date
        ),
        "initial_state": initial_state,
        "training_rows": len(training_data),
        "actual_future_rows": len(actual_future),
        "forecast_weight": float(
            forecast_weight
        ),
        **rainfall_evaluation,
        **forecast_evaluation,
    }




