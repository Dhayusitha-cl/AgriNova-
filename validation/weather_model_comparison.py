"""
Leakage-safe rainfall-model validation for Yavatmal.

Uses the project's existing backtesting implementation to evaluate
the month-aware Monte Carlo rainfall model on held-out historical
decision dates.

This is rainfall-simulation validation only. It does not establish
crop-establishment or SOW / WAIT / SWITCH decision quality.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting import (
    run_multi_date_month_aware_backtest,
)


DATA_YEARS = range(2019, 2025)

DECISION_DATES = [
    "2020-06-15",
    "2020-07-01",
    "2021-06-15",
    "2021-07-01",
    "2022-06-15",
    "2022-07-01",
    "2023-06-15",
    "2023-07-01",
    "2024-06-15",
    "2024-07-01",
]

HORIZON_DAYS = 14
NUM_SIMULATIONS = 1000
RANDOM_SEED = 42


def load_data():
    """Load processed Yavatmal rainfall data."""

    frames = []

    for year in DATA_YEARS:
        path = Path(
            f"data/processed/rainfall_yavatmal_{year}.csv"
        )

        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def print_summary(results):
    """Print a concise validation summary."""

    print()
    print("=" * 72)
    print("YAVATMAL RAINFALL MODEL VALIDATION")
    print("=" * 72)
    print(f"Decision dates : {len(DECISION_DATES)}")
    print(f"Horizon        : {HORIZON_DAYS} days")
    print(f"Simulations    : {NUM_SIMULATIONS}")
    print()

    columns = [
        "decision_date",
        "actual_total_mm",
        "simulated_mean_mm",
        "absolute_error_mm",
        "p10_mm",
        "p90_mm",
        "inside_p10_p90",
    ]

    available = [
        column
        for column in columns
        if column in results.columns
    ]

    print(results[available].to_string(index=False))

    print()
    print("-" * 72)

    if "absolute_error_mm" in results.columns:
        print(
            "Mean absolute rainfall error : "
            f"{results['absolute_error_mm'].mean():.2f} mm"
        )

    if "inside_p10_p90" in results.columns:
        coverage = (
            results["inside_p10_p90"].mean() * 100
        )
        print(
            f"P10-P90 coverage             : "
            f"{coverage:.1f}%"
        )

    print()
    print("INTERPRETATION")
    print("-" * 72)
    print(
        "These results are historical backtesting evidence."
    )
    print(
        "Future observations are held out from model training."
    )
    print(
        "This validates rainfall simulation behaviour only;"
    )
    print(
        "it does not establish agronomic decision quality."
    )
    print(
        "Simulation probabilities are estimates, not guarantees."
    )


def main():
    """Run the rainfall validation."""

    data = load_data()

    results = run_multi_date_month_aware_backtest(
        dataframe=data,
        decision_dates=DECISION_DATES,
        horizon_days=HORIZON_DAYS,
        num_simulations=NUM_SIMULATIONS,
        random_seed=RANDOM_SEED,
    )

    print_summary(results)


if __name__ == "__main__":
    main()