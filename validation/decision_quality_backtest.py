"""
CropLogic-Saathi rainfall-simulation validation.

Purpose
-------
Evaluate whether the existing Monte Carlo rainfall simulator
produces plausible 14-day rainfall distributions when compared
with held-out historical observations.

IMPORTANT
---------

This is rainfall-model validation only.

It is NOT yet the final agronomic decision-quality validation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import src.backtesting as b

from src.soil_water import simulate_soil_water
from src.crop_establishment import soils


# ============================================================
# CONFIGURATION
# ============================================================

DATA_YEARS = range(2019, 2025)

BACKTEST_DATES = [
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load processed Yavatmal rainfall observations."""

    frames = []

    for year in DATA_YEARS:

        path = (
            f"data/processed/"
            f"rainfall_yavatmal_{year}.csv"
        )

        dataframe = pd.read_csv(path)

        dataframe["date"] = pd.to_datetime(
            dataframe["date"]
        )

        frames.append(dataframe)

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        data
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )


# ============================================================
# RUN VALIDATION
# ============================================================

def run_validation():

    data = load_data()

    results = []

    print("=" * 100)
    print("CROPLOGIC-SAATHI RAINFALL-SIMULATION VALIDATION")
    print("=" * 100)

    print(
        f"Horizon        : {HORIZON_DAYS} days"
    )

    print(
        f"Simulations    : {NUM_SIMULATIONS}"
    )

    print(
        f"Backtest dates : {len(BACKTEST_DATES)}"
    )

    print()

    for date_string in BACKTEST_DATES:

        decision_date = pd.Timestamp(
            date_string
        )

        print(
            f"Running validation: {date_string}"
        )

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        training = b.get_training_data(
            data,
            decision_date,
        )

        if training.empty:
            print(
                "  SKIPPED: no training data."
            )
            continue

        # ----------------------------------------------------
        # HELD-OUT FUTURE
        # ----------------------------------------------------

        actual = b.get_actual_future_data(
            data,
            decision_date,
            HORIZON_DAYS,
        )

        if actual.empty:
            print(
                "  SKIPPED: insufficient future data."
            )
            continue

        # ----------------------------------------------------
        # INITIAL STATE
        # ----------------------------------------------------

        initial_state = b.get_initial_state(
            training,
            decision_date,
        )

        # ----------------------------------------------------
        # RAINFALL SIMULATION EVALUATION
        # ----------------------------------------------------

        evaluation = b.evaluate_rainfall_simulation(
            training_data=training,
            actual_future=actual,
            start_date=decision_date,
            initial_state=initial_state,
            horizon_days=HORIZON_DAYS,
            num_simulations=NUM_SIMULATIONS,
            random_seed=RANDOM_SEED,
        )

        result = {
            "date": decision_date,
            "initial_state": initial_state,
            **evaluation,
        }

        results.append(result)

        print(
            f"  Initial state : {initial_state}"
        )

        print(
            f"  Actual total  : "
            f"{evaluation['actual_total_mm']:.2f} mm"
        )

        print(
            f"  Simulated mean : "
            f"{evaluation['simulated_mean_mm']:.2f} mm"
        )

        print(
            f"  P10-P90       : "
            f"{evaluation['p10_mm']:.2f} - "
            f"{evaluation['p90_mm']:.2f} mm"
        )

        print(
            f"  Absolute error: "
            f"{evaluation['absolute_error_mm']:.2f} mm"
        )

        print(
            f"  Interval cover : "
            f"{evaluation['covered_by_p10_p90']}"
        )

        print()

    return pd.DataFrame(results)

def estimate_current_soil_water(
    training_data,
    decision_date,
    soil_type,
    history_days=7,
):
    """
    Reconstruct soil-water storage immediately before a
    historical decision date.

    Only rainfall observations strictly before decision_date
    are used.

    The first available day in the reconstruction window is
    initialized at 50% of field capacity, matching the default
    initialization used by simulate_soil_water().

    This is a backtesting assumption, not a measured field value.
    """

    decision_date = pd.Timestamp(decision_date)

    history = training_data[
        training_data["date"] < decision_date
    ].tail(history_days)

    if len(history) < history_days:
        raise ValueError(
            f"Not enough pre-decision rainfall history for "
            f"{history_days}-day soil-water reconstruction."
        )

    rainfall_series = (
        history["rainfall_mm"]
        .astype(float)
        .tolist()
    )

    et_series = [5.0] * len(rainfall_series)

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    initial_water_mm = field_capacity * 0.5

    results = simulate_soil_water(
        rainfall_series=rainfall_series,
        et_series=et_series,
        soil_type=soil_type,
        initial_water_mm=initial_water_mm,
    )

    return float(
        results[-1]["final_water_mm"]
    )

# ============================================================
# REPORT
# ============================================================

def print_report(result_df):

    if result_df.empty:
        print("No validation results generated.")
        return

    print()
    print("=" * 100)
    print("RAINFALL VALIDATION SUMMARY")
    print("=" * 100)

    mean_actual = (
        result_df["actual_total_mm"].mean()
    )

    mean_simulated = (
        result_df["simulated_mean_mm"].mean()
    )

    mean_bias = (
        result_df["simulated_mean_mm"]
        - result_df["actual_total_mm"]
    ).mean()

    mean_absolute_error = (
        result_df["absolute_error_mm"].mean()
    )

    coverage = (
        result_df["covered_by_p10_p90"].mean()
        * 100.0
    )

    print(
        f"Mean actual rainfall       : "
        f"{mean_actual:.2f} mm"
    )

    print(
        f"Mean simulated rainfall    : "
        f"{mean_simulated:.2f} mm"
    )

    print(
        f"Mean rainfall bias         : "
        f"{mean_bias:.2f} mm"
    )

    print(
        f"Mean absolute error        : "
        f"{mean_absolute_error:.2f} mm"
    )

    print(
        f"P10-P90 coverage           : "
        f"{coverage:.1f}%"
    )

    print()
    print("=" * 100)
    print("DECISION-DATE RESULTS")
    print("=" * 100)

    columns = [
        "date",
        "initial_state",
        "actual_total_mm",
        "simulated_mean_mm",
        "p10_mm",
        "p90_mm",
        "absolute_error_mm",
        "covered_by_p10_p90",
    ]

    print(
        result_df[columns].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print(
        "This evaluates the rainfall simulation layer "
        "against held-out historical observations."
    )

    print(
        "It does not establish agronomic decision quality."
    )

    print(
        "A good rainfall fit alone does not prove that "
        "SOW / WAIT / SWITCH decisions are correct."
    )

    print(
        "The next validation stage must evaluate "
        "crop establishment risk and decision quality."
    )

    print(
        "Simulation probabilities are estimates, "
        "not guarantees."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    results = run_validation()

    print_report(
        results
    )