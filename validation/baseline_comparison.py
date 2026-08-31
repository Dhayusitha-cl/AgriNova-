
"""
Baseline comparison for CropLogic-Saathi.

Compares three pre-sowing decision approaches on the same
historical decision dates:

1. Weather-only baseline
2. Simple rule-based baseline
3. CropLogic-Saathi probabilistic decision engine

Leakage protection:
- Only observations strictly before the decision date are
  supplied to the decision-making methods.
- The following 14 days are held out and used only for
  evaluation.
- Actual future rainfall is never used to make the decision.

Important:
- This is historical backtesting evidence.
- It is not field validation.
- The outcome evaluator is a simplified proxy based on
  held-out rainfall and documented crop parameters.
- It does not prove causal impact or guarantee future
  decisions.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.backtesting as b
from src.crop_data import crops
from src.decision_engine import (
    make_decision,
    simulate_rainfall_soil_water,
    evaluate_establishment,
)
from src.soil_data import soils


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

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
SWITCH_CROP = "soybean"

SOIL_TYPE = "medium_black"

HORIZON = 14
NUM_SIMULATIONS = 1000

INITIAL_MOISTURE_FRACTION = 0.50

# Simple baseline thresholds.
#
# These are intentionally fixed before evaluation and are not
# tuned using the held-out future observations.
RECENT_RAINFALL_DAYS = 3
SOW_RAINFALL_THRESHOLD_MM = 15.0
SWITCH_RAINFALL_THRESHOLD_MM = 5.0


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

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
    field-moisture observations are unavailable.
    """

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    return field_capacity * INITIAL_MOISTURE_FRACTION


# ---------------------------------------------------------------------
# BASELINE 1: WEATHER ONLY
# ---------------------------------------------------------------------

def weather_only_baseline(training_data):
    """
    Simple weather-only baseline.

    Uses only rainfall observations available before the
    historical decision date.

    Decision rule:
        recent 3-day rainfall >= 15 mm -> SOW TODAY
        otherwise -> WAIT 5 DAYS

    This baseline intentionally does not use:
        - soil type
        - soil moisture
        - crop establishment simulation
        - economic calculations
        - future rainfall
    """

    recent = training_data.tail(
        RECENT_RAINFALL_DAYS
    )

    recent_rainfall = float(
        recent["rainfall_mm"].sum()
    )

    if recent_rainfall >= SOW_RAINFALL_THRESHOLD_MM:
        return "SOW TODAY"

    return "WAIT 5 DAYS"


# ---------------------------------------------------------------------
# BASELINE 2: SIMPLE RULE
# ---------------------------------------------------------------------

def rule_based_baseline(
    training_data,
    current_moisture_mm,
    soil_type,
):
    """
    Simple rainfall + soil-moisture rule baseline.

    Rules:

        1. If recent rainfall and current soil moisture
           indicate reasonable establishment conditions:
               SOW TODAY

        2. If rainfall is very low and moisture is low:
               SWITCH TO SOYBEAN

        3. Otherwise:
               WAIT 5 DAYS

    The thresholds are fixed before evaluating the historical
    decision dates.

    This is intentionally much simpler than CropLogic-Saathi.
    """

    recent = training_data.tail(
        RECENT_RAINFALL_DAYS
    )

    recent_rainfall = float(
        recent["rainfall_mm"].sum()
    )

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    moisture_fraction = (
        current_moisture_mm / field_capacity
    )

    if (
        recent_rainfall >= SOW_RAINFALL_THRESHOLD_MM
        and moisture_fraction >= 0.50
    ):
        return "SOW TODAY"

    if (
        recent_rainfall < SWITCH_RAINFALL_THRESHOLD_MM
        and moisture_fraction < 0.50
    ):
        return "SWITCH TO SOYBEAN"

    return "WAIT 5 DAYS"


# ---------------------------------------------------------------------
# CROPLOGIC-SAATHI
# ---------------------------------------------------------------------

def croplogic_decision(
    training_data,
    decision_date,
    initial_state,
    current_moisture_mm,
):
    """
    Run the existing CropLogic-Saathi decision engine.

    Only historical training data is supplied.
    """

    rainfall_yesterday = float(
        training_data.iloc[-1]["rainfall_mm"]
    )

    result = make_decision(
        crop_name=CROP,
        soil_type=SOIL_TYPE,
        current_moisture_mm=current_moisture_mm,
        rainfall_yesterday_mm=rainfall_yesterday,
        transition_matrix=None,
        num_simulations=NUM_SIMULATIONS,
        days_to_simulate=HORIZON,
        random_seed=42,
        start_date=decision_date,
        rainfall_data=training_data,
        initial_state=initial_state,
    )

    return result


# ---------------------------------------------------------------------
# HELD-OUT OUTCOME PROXY
# ---------------------------------------------------------------------

def calculate_realized_establishment(
    actual_future,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Evaluate actual held-out rainfall using the same
    soil-water and establishment model used by CropLogic-Saathi.

    Future observations are supplied only after the decision
    has been generated.

    Returns:
        1.0 if the crop establishes successfully.
        0.0 otherwise.

    This is a historical realized-outcome evaluation,
    not a field validation result.
    """

    crop = crops[crop_name]

    germination_days = int(
        crop["germination_days"]
    )

    rainfall = (
        actual_future["rainfall_mm"]
        .astype(float)
        .head(germination_days)
        .tolist()
    )

    if len(rainfall) < germination_days:
        return 0.0

    soil_water_results = simulate_rainfall_soil_water(
        rainfall_scenario=[
            {"rainfall_mm": value}
            for value in rainfall
        ],
        soil_type=soil_type,
        initial_moisture_mm=initial_moisture_mm,
    )

    establishment = evaluate_establishment(
        soil_water_results=soil_water_results,
        crop=crop_name,
        soil_type=soil_type,
    )

    return float(
        establishment["establishment_success"]
    )


def calculate_realized_outcome(
    decision,
    actual_future,
    soil_type,
    initial_moisture_mm,
):
    """
    Evaluate the realized outcome of a historical decision
    using held-out rainfall and the soil-water model.

    Future rainfall is used only after the decision has
    already been generated.

    For WAIT 5 DAYS:
        - First simulate the five waiting days.
        - The resulting soil water becomes the sowing-day
          initial moisture.
        - Then evaluate crop establishment using the
          remaining held-out rainfall.

    Returns:
        1.0 if establishment succeeds.
        0.0 otherwise.
    """

    if decision == "SWITCH TO SOYBEAN":

        crop_name = SWITCH_CROP

        relevant_future = actual_future

        sowing_moisture = initial_moisture_mm

    elif decision == "WAIT 5 DAYS":

        crop_name = CROP

        wait_days = 5

        if len(actual_future) <= wait_days:
            return 0.0

        wait_future = actual_future.iloc[
            :wait_days
        ]

        wait_results = simulate_rainfall_soil_water(
            rainfall_scenario=[
                {
                    "rainfall_mm": float(
                        rainfall
                    )
                }
                for rainfall in wait_future[
                    "rainfall_mm"
                ]
            ],
            soil_type=soil_type,
            initial_moisture_mm=initial_moisture_mm,
        )

        sowing_moisture = float(
            wait_results[-1]["final_water_mm"]
        )

        relevant_future = actual_future.iloc[
            wait_days:
        ]

    else:

        crop_name = CROP

        relevant_future = actual_future

        sowing_moisture = initial_moisture_mm

    return calculate_realized_establishment(
        actual_future=relevant_future,
        crop_name=crop_name,
        soil_type=soil_type,
        initial_moisture_mm=sowing_moisture,
    )


# ---------------------------------------------------------------------
# SINGLE DATE
# ---------------------------------------------------------------------

def run_single_date(
    data,
    decision_date,
):
    """
    Evaluate all three approaches on one historical date.
    """

    decision_timestamp = pd.Timestamp(
        decision_date
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

    initial_moisture = estimate_initial_moisture(
        SOIL_TYPE
    )

    # -------------------------------------------------------------
    # WEATHER ONLY
    # -------------------------------------------------------------

    weather_decision = weather_only_baseline(
        training
    )

    # -------------------------------------------------------------
    # SIMPLE RULE
    # -------------------------------------------------------------

    rule_decision = rule_based_baseline(
        training_data=training,
        current_moisture_mm=initial_moisture,
        soil_type=SOIL_TYPE,
    )

    # -------------------------------------------------------------
    # CROPLOGIC-SAATHI
    # -------------------------------------------------------------

    croplogic_result = croplogic_decision(
        training_data=training,
        decision_date=decision_timestamp,
        initial_state=initial_state,
        current_moisture_mm=initial_moisture,
    )

    croplogic_decision_name = (
        croplogic_result["decision"]
    )

    # -------------------------------------------------------------
    # HELD-OUT OUTCOME EVALUATION
    # -------------------------------------------------------------

    weather_outcome = calculate_realized_outcome(
        decision=weather_decision,
        actual_future=actual_future,
        soil_type=SOIL_TYPE,
        initial_moisture_mm=initial_moisture,
    )

    rule_outcome = calculate_realized_outcome(
        decision=rule_decision,
        actual_future=actual_future,
        soil_type=SOIL_TYPE,
        initial_moisture_mm=initial_moisture,
    )

    croplogic_outcome = calculate_realized_outcome(
        decision=croplogic_decision_name,
        actual_future=actual_future,
        soil_type=SOIL_TYPE,
        initial_moisture_mm=initial_moisture,
    )

    actual_total = float(
        actual_future["rainfall_mm"].sum()
    )   
  
    return {
        "decision_date": decision_date,
        "initial_state": initial_state,
        "actual_14d_rainfall_mm": actual_total,

        "weather_only_decision": weather_decision,
        "weather_only_outcome": weather_outcome,

        "rule_based_decision": rule_decision,
        "rule_based_outcome": rule_outcome,

        "croplogic_decision": croplogic_decision_name,
        "croplogic_outcome": croplogic_outcome,

        "cotton_germ_prob": croplogic_result[
            "germ_prob_today"
        ],
        "wait_germ_prob": croplogic_result[
            "germ_prob_wait"
        ],
        "soybean_germ_prob": croplogic_result[
            "germ_prob_soybean"
        ],
    }


# ---------------------------------------------------------------------
# MULTI-DATE VALIDATION
# ---------------------------------------------------------------------

def run_validation(data):
    """Run the baseline comparison across all dates."""

    rows = []

    for decision_date in BACKTEST_DATES:

        print(
            f"Running validation: {decision_date}"
        )

        row = run_single_date(
            data,
            decision_date,
        )

        rows.append(row)

        print(
            f"  Weather-only : "
            f"{row['weather_only_decision']}"
        )
        print(
            f"  Rule-based   : "
            f"{row['rule_based_decision']}"
        )
        print(
            f"  CropLogic     : "
            f"{row['croplogic_decision']}"
        )
        print(
            f"  Actual 14d   : "
            f"{row['actual_14d_rainfall_mm']:.2f} mm"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def print_report(results):
    """Print baseline comparison results."""

    print()
    print("=" * 110)
    print("CROPLOGIC-SAATHI BASELINE COMPARISON")
    print("=" * 110)

    print(
        f"Crop                 : "
        f"{crops[CROP]['name']}"
    )
    print(
        f"Soil                 : "
        f"{soils[SOIL_TYPE]['name']}"
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
        f"Backtest dates       : "
        f"{len(results)}"
    )

    print()
    print("-" * 110)

    display_columns = [
        "decision_date",
        "actual_14d_rainfall_mm",
        "weather_only_decision",
        "rule_based_decision",
        "croplogic_decision",
        "weather_only_outcome",
        "rule_based_outcome",
        "croplogic_outcome",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------
    # DECISION DISTRIBUTION
    # -------------------------------------------------------------

    print()
    print("=" * 110)
    print("DECISION DISTRIBUTION")
    print("=" * 110)

    for column, label in [
        (
            "weather_only_decision",
            "WEATHER-ONLY",
        ),
        (
            "rule_based_decision",
            "RULE-BASED",
        ),
        (
            "croplogic_decision",
            "CROPLOGIC-SAATHI",
        ),
    ]:

        print()
        print(label)

        counts = (
            results[column]
            .value_counts()
        )

        for decision in [
            "SOW TODAY",
            "WAIT 5 DAYS",
            "SWITCH TO SOYBEAN",
        ]:
            print(
                f"  {decision:<20}: "
                f"{int(counts.get(decision, 0))}"
            )

    # -------------------------------------------------------------
    # OUTCOME PROXY
    # -------------------------------------------------------------

    print()
    print("=" * 110)
    print("HELD-OUT OUTCOME PROXY")
    print("=" * 110)

    weather_score = (
        results["weather_only_outcome"]
        .mean()
    )

    rule_score = (
        results["rule_based_outcome"]
        .mean()
    )

    croplogic_score = (
        results["croplogic_outcome"]
        .mean()
    )

    print(
        f"Weather-only success proxy : "
        f"{weather_score:.3f}"
    )

    print(
        f"Rule-based success proxy   : "
        f"{rule_score:.3f}"
    )

    print(
        f"CropLogic success proxy    : "
        f"{croplogic_score:.3f}"
    )

    # -------------------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------------------

    print()
    print("=" * 110)
    print("INTERPRETATION")
    print("=" * 110)

    print(
        "All three approaches receive information available "
        "before each historical decision date."
    )

    print(
        "The subsequent 14-day rainfall is held out and "
        "used only for evaluation."
    )

    print(
        "The weather-only and rule-based methods are "
        "intentionally simple baselines."
    )

    print(
        "The outcome metric is a simplified rainfall-based "
        "proxy, not field-measured crop establishment."
    )

    print(
        "These realized outcomes are a historical model-based "
        "evaluation and do not establish field performance, "
        "causal impact, or agronomic superiority."
    )

    print(
        "This comparison is intended to test whether the "
        "probabilistic decision approach provides useful "
        "additional decision support beyond simple rules."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    data = load_data()

    results = run_validation(
        data
    )

    print_report(
        results
    )

