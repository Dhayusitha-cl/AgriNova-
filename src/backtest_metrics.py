"""
Backtesting metrics for rainfall scenario calibration.

These functions evaluate whether observed future rainfall
is reasonably represented by the simulated rainfall distribution.

Important:
- These are evaluation metrics only.
- They do not represent forecast guarantees.
- Simulation probabilities are not guarantees of future rainfall.
"""

import numpy as np
import pandas as pd


def calculate_backtest_metrics(results):
    """
    Calculate aggregate rainfall backtesting metrics.

    Parameters
    ----------
    results : pandas.DataFrame
        Output from the backtesting functions.

    Required columns
    ----------------
    actual_total_mm
        Observed rainfall over the backtest horizon.

    simulated_mean_mm
        Mean rainfall from Monte Carlo simulations.

    actual_percentile
        Percentile position of the observed rainfall
        within the simulated distribution.

    Returns
    -------
    dict
        Aggregate backtesting metrics.
    """

    if results.empty:
        raise ValueError("Backtest results cannot be empty.")

    required = {
        "actual_total_mm",
        "simulated_mean_mm",
        "actual_percentile",
    }

    missing = required - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    actual = results["actual_total_mm"].astype(float)
    simulated = results["simulated_mean_mm"].astype(float)

    errors = simulated - actual
    absolute_errors = np.abs(errors)

    # Percentage error is undefined when actual rainfall is zero.
    percentage_errors = (
        absolute_errors / actual.replace(0, np.nan)
    ) * 100

    actual_percentiles = (
        results["actual_percentile"].astype(float)
    )

    return {
        "num_backtests": int(len(results)),

        "mean_actual_mm": float(
            actual.mean()
        ),

        "mean_simulated_mm": float(
            simulated.mean()
        ),

        "mean_error_mm": float(
            errors.mean()
        ),

        "mean_absolute_error_mm": float(
            absolute_errors.mean()
        ),

        "mean_absolute_percentage_error": float(
            percentage_errors.mean()
        ),

        "median_actual_percentile": float(
            actual_percentiles.median()
        ),

        "mean_actual_percentile": float(
            actual_percentiles.mean()
        ),

        "pct_actual_below_p90": float(
            (actual_percentiles <= 90).mean() * 100
        ),

        "pct_actual_below_p95": float(
            (actual_percentiles <= 95).mean() * 100
        ),

        "pct_actual_below_p50": float(
            (actual_percentiles <= 50).mean() * 100
        ),
    }


def calculate_interval_coverage(
    results,
    lower_percentile=10,
    upper_percentile=90,
):
    """
    Calculate how often observed rainfall falls inside
    the simulated percentile interval.

    Parameters
    ----------
    results : pandas.DataFrame
        Backtesting results.

    lower_percentile : int or float
        Lower percentile used for the interval.

    upper_percentile : int or float
        Upper percentile used for the interval.

    Required columns
    ----------------
    actual_total_mm
    simulated_p10_mm
    simulated_p90_mm

    Returns
    -------
    dict
        Interval coverage statistics.

    Important
    ---------
    This is a descriptive backtesting metric.
    It does not represent a forecast guarantee.
    """

    if results.empty:
        raise ValueError(
            "Backtest results cannot be empty."
        )

    if lower_percentile >= upper_percentile:
        raise ValueError(
            "lower_percentile must be less than upper_percentile."
        )

    required = {
        "actual_total_mm",
        "simulated_p10_mm",
        "simulated_p90_mm",
    }

    missing = required - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    inside = (
        (results["actual_total_mm"]
         >= results["simulated_p10_mm"])
        &
        (results["actual_total_mm"]
         <= results["simulated_p90_mm"])
    )

    return {
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "coverage_percent": float(
            inside.mean() * 100
        ),
        "num_inside": int(
            inside.sum()
        ),
        "num_total": int(
            len(inside)
        ),
    }


def compare_monthly_transition_matrices(
    training_data,
):
    """
    Compare rainfall-state transition probabilities
    separately for each month.

    This is a diagnostic tool for checking whether
    rainfall persistence differs across months.

    Parameters
    ----------
    training_data : pandas.DataFrame
        Must contain:
        - date
        - rainfall_state

    Returns
    -------
    dict
        Dictionary mapping month number to a
        rainfall-state transition probability matrix.
    """

    required = {
        "date",
        "rainfall_state",
    }

    missing = required - set(training_data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = training_data.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    states = [
        "dry",
        "drizzle",
        "rain",
    ]

    output = {}

    for month in sorted(
        data["date"].dt.month.unique()
    ):

        month_data = data[
            data["date"].dt.month == month
        ].copy()

        counts = pd.DataFrame(
            0,
            index=states,
            columns=states,
            dtype=int,
        )

        current_states = (
            month_data["rainfall_state"]
            .iloc[:-1]
        )

        next_states = (
            month_data["rainfall_state"]
            .iloc[1:]
        )

        for current_state, next_state in zip(
            current_states,
            next_states,
        ):
            if (
                current_state in states
                and next_state in states
            ):
                counts.loc[
                    current_state,
                    next_state
                ] += 1

        row_totals = counts.sum(axis=1)

        matrix = counts.div(
            row_totals.replace(0, np.nan),
            axis=0,
        ).fillna(0.0)

        output[int(month)] = matrix

    return output