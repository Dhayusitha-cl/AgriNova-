"""
Comparison of simple historical baseline and
CropLogic-Saathi Monte Carlo rainfall simulation.
"""

import pandas as pd

from src.backtest_baseline import (
    calculate_historical_mean_baseline,
)
from src.backtesting import (
    create_backtest_split,
    run_single_month_aware_backtest,
)


def run_baseline_vs_monte_carlo(
    dataframe,
    decision_dates,
    horizon_days=14,
    num_simulations=1000,
    random_seed=42,
):
    """
    Compare a simple historical rainfall baseline
    against the month-aware Monte Carlo model.

    Future observations are used only for evaluation.
    """

    results = []

    for i, decision_date in enumerate(decision_dates):

        split = create_backtest_split(
            dataframe,
            decision_date,
            horizon_days,
        )

        baseline = calculate_historical_mean_baseline(
            split["training_data"],
            decision_date,
            horizon_days,
        )

        monte_carlo = run_single_month_aware_backtest(
            dataframe,
            decision_date,
            horizon_days,
            num_simulations,
            random_seed + i,
        )

        actual = float(
            split["actual_future"]["rainfall_mm"].sum()
        )

        baseline_error = baseline - actual

        monte_carlo_error = (
            monte_carlo["simulated_mean_mm"] - actual
        )

        results.append(
            {
                "decision_date": pd.Timestamp(
                    decision_date
                ),
                "actual_total_mm": actual,
                "baseline_mean_mm": float(
                    baseline
                ),
                "baseline_error_mm": float(
                    baseline_error
                ),
                "baseline_absolute_error_mm": float(
                    abs(baseline_error)
                ),
                "monte_carlo_mean_mm": float(
                    monte_carlo["simulated_mean_mm"]
                ),
                "monte_carlo_error_mm": float(
                    monte_carlo_error
                ),
                "monte_carlo_absolute_error_mm": float(
                    abs(monte_carlo_error)
                ),
                "monte_carlo_actual_percentile": float(
                    monte_carlo["actual_percentile"]
                ),
            }
        )

    return pd.DataFrame(results)


def summarize_baseline_vs_monte_carlo(results):
    """
    Calculate aggregate comparison metrics.
    """

    if results.empty:
        raise ValueError(
            "Comparison results cannot be empty."
        )

    return {
        "num_backtests": int(len(results)),

        "baseline_mae_mm": float(
            results["baseline_absolute_error_mm"].mean()
        ),

        "monte_carlo_mae_mm": float(
            results[
                "monte_carlo_absolute_error_mm"
            ].mean()
        ),

        "baseline_mean_error_mm": float(
            results["baseline_error_mm"].mean()
        ),

        "monte_carlo_mean_error_mm": float(
            results["monte_carlo_error_mm"].mean()
        ),

        "baseline_better_count": int(
            (
                results["baseline_absolute_error_mm"]
                <
                results["monte_carlo_absolute_error_mm"]
            ).sum()
        ),

        "monte_carlo_better_count": int(
            (
                results["monte_carlo_absolute_error_mm"]
                <
                results["baseline_absolute_error_mm"]
            ).sum()
        ),
    }