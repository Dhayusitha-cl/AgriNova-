"""
Decision-quality backtest for CropLogic-Saathi.

This evaluates the existing decision engine on historical
decision dates using leakage-safe training data.

Important:
- Only observations before the decision date are supplied
  to the decision engine.
- Future observations are used only for evaluation.
- Historical field moisture is unavailable, so soil water
  is initialized at 50% of soil field capacity.
- This is decision-quality backtesting evidence, not field
  validation and not proof of future performance.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.backtesting as b
from src.crop_data import crops
from src.decision_engine import make_decision
from src.soil_data import soils


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

CROP = "cotton"
HORIZON = 14
NUM_SIMULATIONS = 1000
INITIAL_MOISTURE_FRACTION = 0.50


def load_data():
    """Load processed Yavatmal rainfall data."""

    frames = []

    for year in DATA_YEARS:
        frame = pd.read_csv(
            f"data/processed/rainfall_yavatmal_{year}.csv"
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def estimate_initial_moisture(soil_type):
    """
    Use the documented model default because historical
    field moisture observations are unavailable.
    """

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    return field_capacity * INITIAL_MOISTURE_FRACTION


def run_validation(data, soil_type="medium_black"):
    """Run leakage-safe decision-quality backtesting."""

    rows = []

    initial_moisture = estimate_initial_moisture(
        soil_type
    )

    for decision_date in BACKTEST_DATES:

        decision_timestamp = pd.Timestamp(
            decision_date
        )

        print(
            f"Running validation: {decision_date}"
        )

        training = b.get_training_data(
            data,
            decision_timestamp,
        )

        actual_future = b.get_actual_future_data(
            data,
            decision_timestamp,
            HORIZON,
        )

        initial_state = b.get_initial_state(
            training,
            decision_timestamp,
        )

        rainfall_yesterday = float(
            training.iloc[-1]["rainfall_mm"]
        )

        result = make_decision(
            crop_name=CROP,
            soil_type=soil_type,
            current_moisture_mm=initial_moisture,
            rainfall_yesterday_mm=rainfall_yesterday,
            transition_matrix=None,
            num_simulations=NUM_SIMULATIONS,
            days_to_simulate=HORIZON,
            random_seed=42,
            start_date=decision_timestamp,
            rainfall_data=training,
            initial_state=initial_state,
        )

        economic = result[
            "economic_comparison"
        ]

        actual_total = float(
            actual_future["rainfall_mm"].sum()
        )

        rows.append(
            {
                "decision_date": decision_date,
                "initial_state": initial_state,
                "initial_moisture_mm": initial_moisture,
                "actual_14d_rainfall_mm": actual_total,
                "decision": result["decision"],
                "cotton_germ_prob": result[
                    "germ_prob_today"
                ],
                "wait_germ_prob": result[
                    "germ_prob_wait"
                ],
                "soybean_germ_prob": result[
                    "germ_prob_soybean"
                ],
                "sow_profit": economic[
                    "sow_today"
                ]["expected_profit"],
                "wait_profit": economic[
                    "wait"
                ]["expected_profit"],
                "switch_profit": economic[
                    "switch"
                ]["expected_profit"],
                "best_profit": economic[
                    "best_profit"
                ],
            }
        )

        print(
            f"  Initial state      : {initial_state}"
        )
        print(
            f"  Initial moisture   : "
            f"{initial_moisture:.2f} mm"
        )
        print(
            f"  Actual 14d rainfall: "
            f"{actual_total:.2f} mm"
        )
        print(
            f"  Decision            : "
            f"{result['decision']}"
        )
        print(
            f"  Cotton probability  : "
            f"{result['germ_prob_today']:.3f}"
        )
        print(
            f"  Wait probability    : "
            f"{result['germ_prob_wait']:.3f}"
        )
        print(
            f"  Soybean probability : "
            f"{result['germ_prob_soybean']:.3f}"
        )

    return pd.DataFrame(rows)


def print_report(results):
    """Print decision-quality validation summary."""

    print()
    print("=" * 110)
    print("CROPLOGIC-SAATHI DECISION-QUALITY BACKTEST")
    print("=" * 110)

    print(
        f"Crop                 : "
        f"{crops[CROP]['name']}"
    )
    print(
        f"Horizon              : "
        f"{HORIZON} days"
    )
    print(
        f"Simulations          : "
        f"{NUM_SIMULATIONS}"
    )
    print(
        f"Initial soil water   : "
        f"{INITIAL_MOISTURE_FRACTION:.0%} "
        f"of field capacity"
    )
    print(
        f"Backtest dates       : "
        f"{len(results)}"
    )

    print()
    print("-" * 110)

    display_columns = [
        "decision_date",
        "initial_state",
        "actual_14d_rainfall_mm",
        "decision",
        "cotton_germ_prob",
        "wait_germ_prob",
        "soybean_germ_prob",
        "sow_profit",
        "wait_profit",
        "switch_profit",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=" * 110)
    print("DECISION SUMMARY")
    print("=" * 110)

    decision_counts = (
        results["decision"]
        .value_counts()
    )

    for decision in [
        "SOW TODAY",
        "WAIT 5 DAYS",
        "SWITCH TO SOYBEAN",
    ]:
        print(
            f"{decision:<20}: "
            f"{int(decision_counts.get(decision, 0))}"
        )

    print()
    print("=" * 110)
    print("PROBABILITY SUMMARY")
    print("=" * 110)

    print(
        f"Mean cotton establishment probability : "
        f"{results['cotton_germ_prob'].mean():.3f}"
    )
    print(
        f"Mean wait establishment probability   : "
        f"{results['wait_germ_prob'].mean():.3f}"
    )
    print(
        f"Mean soybean establishment probability: "
        f"{results['soybean_germ_prob'].mean():.3f}"
    )

    print()
    print("=" * 110)
    print("ECONOMIC SUMMARY")
    print("=" * 110)

    print(
        f"Mean SOW expected profit    : "
        f"{results['sow_profit'].mean():.2f}"
    )
    print(
        f"Mean WAIT expected profit   : "
        f"{results['wait_profit'].mean():.2f}"
    )
    print(
        f"Mean SWITCH expected profit : "
        f"{results['switch_profit'].mean():.2f}"
    )

    print()
    print("=" * 110)
    print("INTERPRETATION")
    print("=" * 110)

    print(
        "This is historical decision-quality backtesting "
        "evidence."
    )
    print(
        "Future rainfall is excluded from decision inputs "
        "and used only for evaluation."
    )
    print(
        "Historical field moisture observations were "
        "unavailable, so soil water was initialized at "
        "50% of field capacity."
    )
    print(
        "Economic outcomes depend on the configured crop "
        "cost, yield and price assumptions."
    )
    print(
        "This does not establish field validation, causal "
        "impact, or guaranteed future decisions."
    )


if __name__ == "__main__":

    data = load_data()

    results = run_validation(
        data,
        soil_type="medium_black",
    )

    print_report(results)