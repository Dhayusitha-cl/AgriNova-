"""
Baseline comparison and decision-quality validation for CropLogic-Saathi.

Compares three pre-sowing decision approaches on the same
historical decision dates:

1. Weather-only baseline
2. Simple rule-based baseline
3. CropLogic-Saathi probabilistic decision engine

Decision-quality evaluation additionally evaluates all three
possible actions on the held-out future:

    - SOW TODAY
    - WAIT 5 DAYS
    - SWITCH TO SOYBEAN

Leakage protection
------------------
- Only observations strictly before the decision date are supplied
  to the decision-making methods.
- The held-out evaluation period is the 14 calendar days AFTER the
  decision date.
- Future rainfall is never used to make a decision.
- Future rainfall is inspected only after the decision has already
  been generated.
- Backtest-date eligibility does not depend on whether the future
  rainfall produced a successful outcome.

Sowing-window handling
----------------------
The primary benchmark uses the documented optimal sowing window for
the target crop in src/crop_data.py.

For cotton this is:

    June 15 - July 15

The optimal window is treated as a validation-date eligibility
window, not as a hard biological failure cutoff.

This script does NOT invent a "minimum remaining days" constraint.

Outcome limitations
-------------------
The realized outcome evaluator uses the project's existing simplified
soil-water and crop-establishment models.

It is therefore:

    historical model-based evaluation

and NOT:

    field-measured crop establishment,
    causal impact evidence,
    economic ground truth,
    or a guarantee of future performance.

The current realized outcome is binary (0/1), so realized regret is
also binary. Ties between actions are common and can make the
best-action-rate metric appear stronger than the underlying
decision discrimination actually is.
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
from src.crop_establishment import evaluate_establishment
from src.soil_data import soils
from src.soil_water import simulate_soil_water


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATA_YEARS = range(2019, 2025)

CROP = "cotton"
SWITCH_CROP = "soybean"

SOIL_TYPE = "medium_black"

HORIZON = 14
NUM_SIMULATIONS = 1000
RANDOM_SEED = 42

# Historical decision dates are sampled every N days inside the
# target crop's documented optimal sowing window.
BACKTEST_INTERVAL_DAYS = 7

# Soil reconstruction assumptions.
#
# These are modelling assumptions, not observed historical field
# measurements.
INITIAL_MOISTURE_FRACTION = 0.50
SOIL_RECONSTRUCTION_DAYS = 30
DAILY_ET_MM = 5.0

# Minimum amount of pre-decision history required before attempting
# a backtest.
MIN_TRAINING_DAYS = 30

# Simple baseline thresholds.
#
# These are fixed before evaluation and are not tuned using held-out
# future rainfall.
RECENT_RAINFALL_DAYS = 3
SOW_RAINFALL_THRESHOLD_MM = 15.0
SWITCH_RAINFALL_THRESHOLD_MM = 5.0

ACTIONS = [
    "SOW TODAY",
    "WAIT 5 DAYS",
    "SWITCH TO SOYBEAN",
]


# ---------------------------------------------------------------------
# SOWING-WINDOW HELPERS
# ---------------------------------------------------------------------

def parse_sowing_window(window_text):
    """
    Parse an optimal sowing window from crop_data.py.

    Expected format:

        "June 15 - July 15"

    Returns
    -------
    tuple
        (start_month, start_day, end_month, end_day)

    Raises
    ------
    ValueError
        If the configured window cannot be parsed.
    """

    if not isinstance(window_text, str):
        raise ValueError(
            "Crop optimal_sowing_window must be a string."
        )

    parts = [
        part.strip()
        for part in window_text.split("-")
    ]

    if len(parts) != 2:
        raise ValueError(
            f"Invalid sowing window format: {window_text!r}"
        )

    start_text, end_text = parts

    try:
        start = pd.to_datetime(
            start_text,
            format="%B %d",
        )

        end = pd.to_datetime(
            end_text,
            format="%B %d",
        )
    except ValueError as exc:
        raise ValueError(
            f"Unable to parse sowing window: "
            f"{window_text!r}"
        ) from exc

    return (
        int(start.month),
        int(start.day),
        int(end.month),
        int(end.day),
    )


def get_crop_sowing_window(crop_name):
    """
    Return the documented optimal sowing window for a crop.

    The window comes directly from src/crop_data.py.

    This function does not interpret the window as a hard agronomic
    cutoff.
    """

    if crop_name not in crops:
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

    return parse_sowing_window(
        crops[crop_name]["optimal_sowing_window"]
    )


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def load_data():
    """Load and combine processed Yavatmal rainfall data."""

    frames = []

    for year in DATA_YEARS:
        path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"rainfall_yavatmal_{year}.csv"
        )

        frame = pd.read_csv(path)

        if "date" not in frame.columns:
            raise ValueError(
                f"Missing 'date' column in {path}."
            )

        if "rainfall_mm" not in frame.columns:
            raise ValueError(
                f"Missing 'rainfall_mm' column in {path}."
            )

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        ).dt.normalize()

        frames.append(frame)

    if not frames:
        raise ValueError(
            "No rainfall datasets were loaded."
        )

    data = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return data


# ---------------------------------------------------------------------
# BACKTEST-DATE GENERATION
# ---------------------------------------------------------------------

def generate_backtest_dates(data):
    """
    Generate deterministic historical decision dates.

    Eligibility requires:

        1. date is inside the target crop's documented optimal
           sowing window,
        2. decision date exists in the dataset,
        3. sufficient pre-decision observations exist,
        4. Markov calibration succeeds using only pre-decision data,
        5. the complete 14-day post-decision evaluation window exists.

    Critically, future rainfall amounts are NOT inspected to decide
    whether a date is eligible.

    The future window is:

        decision_date + 1 day
        through
        decision_date + HORIZON days

    The optimal sowing window is used only to define the primary
    validation population. It is not treated as a hard biological
    cutoff.
    """

    (
        start_month,
        start_day,
        end_month,
        end_day,
    ) = get_crop_sowing_window(CROP)

    normalized_dates = (
        pd.to_datetime(data["date"])
        .dt.normalize()
    )

    available_dates = set(normalized_dates)

    dates = []

    for year in sorted(DATA_YEARS):

        start = pd.Timestamp(
            year=year,
            month=start_month,
            day=start_day,
        )

        end = pd.Timestamp(
            year=year,
            month=end_month,
            day=end_day,
        )

        candidate = start

        while candidate <= end:

            # ---------------------------------------------------------
            # DECISION DATE MUST EXIST
            # ---------------------------------------------------------

            if candidate not in available_dates:
                candidate += pd.Timedelta(
                    days=BACKTEST_INTERVAL_DAYS
                )
                continue

            # ---------------------------------------------------------
            # TRAINING DATA MUST EXIST
            #
            # Only observations strictly before the decision date
            # are allowed.
            # ---------------------------------------------------------

            training = data.loc[
                data["date"] < candidate
            ].copy()

            if len(training) < MIN_TRAINING_DAYS:
                candidate += pd.Timedelta(
                    days=BACKTEST_INTERVAL_DAYS
                )
                continue

            # ---------------------------------------------------------
            # MARKOV CALIBRATION MUST BE POSSIBLE
            #
            # Calibration sees training data only.
            # ---------------------------------------------------------

            try:
                b.calibrate_backtest_transition_matrix(
                    training
                )
            except ValueError:
                candidate += pd.Timedelta(
                    days=BACKTEST_INTERVAL_DAYS
                )
                continue

            # ---------------------------------------------------------
            # REQUIRE COMPLETE POST-DECISION EVALUATION WINDOW
            #
            # IMPORTANT:
            # The decision date itself is NOT part of the future
            # evaluation window.
            # ---------------------------------------------------------

            future_dates = pd.date_range(
                start=candidate + pd.Timedelta(days=1),
                periods=HORIZON,
                freq="D",
            )

            if not all(
                date in available_dates
                for date in future_dates
            ):
                candidate += pd.Timedelta(
                    days=BACKTEST_INTERVAL_DAYS
                )
                continue

            dates.append(
                candidate.strftime("%Y-%m-%d")
            )

            candidate += pd.Timedelta(
                days=BACKTEST_INTERVAL_DAYS
            )

    return dates


# ---------------------------------------------------------------------
# SOIL MOISTURE
# ---------------------------------------------------------------------

def estimate_initial_moisture(
    data,
    decision_date,
    soil_type,
):
    """
    Reconstruct pre-decision soil moisture.

    Only rainfall observations strictly before the decision date
    are used.

    Reconstruction:

        - preceding 30 calendar days
        - initial water = 50% of field capacity
        - fixed 5 mm/day ET assumption
        - canonical soil-water balance model

    This is model-reconstructed soil moisture, not a measured
    historical field observation.
    """

    decision_timestamp = pd.Timestamp(
        decision_date
    ).normalize()

    historical = (
        data.loc[
            data["date"] < decision_timestamp
        ]
        .sort_values("date")
        .copy()
    )

    if len(historical) < SOIL_RECONSTRUCTION_DAYS:
        raise ValueError(
            "Insufficient historical rainfall observations "
            "for soil-moisture reconstruction."
        )

    reconstruction = historical.tail(
        SOIL_RECONSTRUCTION_DAYS
    )

    expected_dates = pd.date_range(
        end=decision_timestamp - pd.Timedelta(days=1),
        periods=SOIL_RECONSTRUCTION_DAYS,
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(
        reconstruction["date"]
    )

    if not actual_dates.equals(expected_dates):
        raise ValueError(
            "Historical rainfall observations are not "
            "consecutive for soil-moisture reconstruction."
        )

    if soil_type not in soils:
        raise ValueError(
            f"Unknown soil type: {soil_type}"
        )

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    initial_water = (
        field_capacity
        * INITIAL_MOISTURE_FRACTION
    )

    rainfall_series = (
        reconstruction["rainfall_mm"]
        .astype(float)
        .tolist()
    )

    et_series = [
        DAILY_ET_MM
        for _ in rainfall_series
    ]

    results = simulate_soil_water(
        rainfall_series=rainfall_series,
        et_series=et_series,
        soil_type=soil_type,
        initial_water_mm=initial_water,
    )

    if not results:
        raise ValueError(
            "Soil-water reconstruction returned no results."
        )

    return float(
        results[-1]["final_water_mm"]
    )


# ---------------------------------------------------------------------
# BASELINE 1: WEATHER ONLY
# ---------------------------------------------------------------------

def weather_only_baseline(training_data):
    """
    Simple weather-only baseline.

    Uses only rainfall observed before the decision date.

    Rule:

        recent 3-day rainfall >= 15 mm
            -> SOW TODAY

        otherwise
            -> WAIT 5 DAYS

    No soil, crop-establishment, economic, or future information
    is supplied to this baseline.
    """

    if len(training_data) < RECENT_RAINFALL_DAYS:
        raise ValueError(
            "Insufficient training data for weather-only baseline."
        )

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

        recent rainfall >= 15 mm AND
        moisture fraction >= 0.50
            -> SOW TODAY

        recent rainfall < 5 mm AND
        moisture fraction < 0.50
            -> SWITCH TO SOYBEAN

        otherwise
            -> WAIT 5 DAYS

    Thresholds are fixed before evaluation.
    """

    if len(training_data) < RECENT_RAINFALL_DAYS:
        raise ValueError(
            "Insufficient training data for rule baseline."
        )

    if soil_type not in soils:
        raise ValueError(
            f"Unknown soil type: {soil_type}"
        )

    recent = training_data.tail(
        RECENT_RAINFALL_DAYS
    )

    recent_rainfall = float(
        recent["rainfall_mm"].sum()
    )

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    if field_capacity <= 0:
        raise ValueError(
            "Soil field capacity must be positive."
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

    Only information available before the historical decision date
    is supplied.
    """

    if training_data.empty:
        raise ValueError(
            "CropLogic requires non-empty training data."
        )

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
        random_seed=RANDOM_SEED,
        start_date=decision_date,
        rainfall_data=training_data,
        initial_state=initial_state,
    )

    return result


# ---------------------------------------------------------------------
# HELD-OUT OUTCOME EVALUATION
# ---------------------------------------------------------------------

def _evaluate_realized_establishment(
    actual_future,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Evaluate held-out rainfall using the existing soil-water and
    crop-establishment models.

    Returns
    -------
    dict | None
        Full crop-establishment result, or None when the held-out
        trajectory is shorter than the crop germination period.

    This is a model-based historical outcome proxy, not field
    validation.
    """

    if crop_name not in crops:
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

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
        return None

    et_series = [
        DAILY_ET_MM
        for _ in rainfall
    ]

    soil_water_results = simulate_soil_water(
        rainfall_series=rainfall,
        et_series=et_series,
        soil_type=soil_type,
        initial_water_mm=initial_moisture_mm,
    )

    return evaluate_establishment(
        soil_water_results=soil_water_results,
        crop=crop_name,
        soil_type=soil_type,
    )

def _evaluate_realized_establishment(
    actual_future,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Evaluate held-out rainfall using the existing soil-water and
    crop-establishment models.

    Returns the full establishment result so validation can use
    both the existing binary outcome and continuous diagnostics.
    """

    if crop_name not in crops:
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

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
        return None

    et_series = [
        DAILY_ET_MM
        for _ in rainfall
    ]

    soil_water_results = simulate_soil_water(
        rainfall_series=rainfall,
        et_series=et_series,
        soil_type=soil_type,
        initial_water_mm=initial_moisture_mm,
    )

    return evaluate_establishment(
        soil_water_results=soil_water_results,
        crop=crop_name,
        soil_type=soil_type,
    )

def calculate_realized_establishment(
    actual_future,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Return the existing binary model-based establishment outcome.

    This is a model-based historical outcome proxy, not field
    validation.
    """

    establishment = _evaluate_realized_establishment(
        actual_future=actual_future,
        crop_name=crop_name,
        soil_type=soil_type,
        initial_moisture_mm=initial_moisture_mm,
    )

    if establishment is None:
        return 0.0

    return float(
        float(establishment["establishment_success"])
    )

def calculate_establishment_diagnostics(
    establishment,
):
    """
    Calculate continuous model-based establishment diagnostics.

    These are validation diagnostics derived from the existing
    crop-establishment model, not biological guarantees.
    """

    if establishment is None:
        return None

    daily_results = establishment["daily_results"]

    minimum_required_pct = (
        establishment["minimum_moisture_pct"]
    )

    moisture_margins = [
        item["moisture_pct"] - minimum_required_pct
        for item in daily_results
    ]

    cumulative_moisture_deficit = sum(
        max(
            0.0,
            minimum_required_pct - item["moisture_pct"],
        )
        for item in daily_results
    )

    return {
        "establishment_success": float(
            establishment["establishment_success"]
        ),
        "minimum_moisture_margin_pct": min(
            moisture_margins
        ),
        "cumulative_moisture_deficit_pct_days": (
            cumulative_moisture_deficit
        ),
        "successful_days": establishment[
            "successful_days"
        ],
        "germination_days": establishment[
            "germination_days"
        ],
    }

def calculate_realized_diagnostics(
    decision,
    actual_future,
    soil_type,
    initial_moisture_mm,
):
    """
    Evaluate one held-out action and return continuous
    establishment diagnostics.

    The action semantics are identical to
    calculate_realized_outcome().
    """

    prepared = _prepare_realized_action(
        decision=decision,
        actual_future=actual_future,
        soil_type=soil_type,
        initial_moisture_mm=initial_moisture_mm,
    )

    if prepared is None:
        return None

    (
        crop_name,
        relevant_future,
        sowing_moisture,
    ) = prepared

    establishment = _evaluate_realized_establishment(
        actual_future=relevant_future,
        crop_name=crop_name,
        soil_type=soil_type,
        initial_moisture_mm=sowing_moisture,
    )

    return calculate_establishment_diagnostics(
        establishment
    )

def _prepare_realized_action(
    decision,
    actual_future,
    soil_type,
    initial_moisture_mm,
):
    """
    Prepare the crop, held-out trajectory, and sowing moisture
    corresponding to one decision.

    Returns
    -------
    tuple
        (crop_name, relevant_future, sowing_moisture)

    Future rainfall is used only after the decision has already
    been generated.
    """

    if decision == "SOW TODAY":

        crop_name = CROP

        relevant_future = actual_future

        sowing_moisture = initial_moisture_mm

    elif decision == "WAIT 5 DAYS":

        crop_name = CROP

        wait_days = 5

        if len(actual_future) <= wait_days:
            return None

        wait_future = actual_future.iloc[
            :wait_days
        ]

        wait_rainfall = (
            wait_future["rainfall_mm"]
            .astype(float)
            .tolist()
        )

        wait_et = [
            DAILY_ET_MM
            for _ in wait_rainfall
        ]

        wait_results = simulate_soil_water(
            rainfall_series=wait_rainfall,
            et_series=wait_et,
            soil_type=soil_type,
            initial_water_mm=initial_moisture_mm,
        )

        if not wait_results:
            return None

        sowing_moisture = float(
            wait_results[-1]["final_water_mm"]
        )

        relevant_future = actual_future.iloc[
            wait_days:
        ]

    elif decision == "SWITCH TO SOYBEAN":

        crop_name = SWITCH_CROP

        relevant_future = actual_future

        sowing_moisture = initial_moisture_mm

    else:
        raise ValueError(
            f"Unknown decision: {decision}"
        )

    return (
        crop_name,
        relevant_future,
        sowing_moisture,
    )

def calculate_realized_outcome(
    decision,
    actual_future,
    soil_type,
    initial_moisture_mm,
):
    """
    Evaluate one selected action against the held-out future.

    SOW TODAY
        Evaluate cotton from the decision-date soil state.

    WAIT 5 DAYS
        Simulate the first five held-out days.
        Then evaluate cotton using the remaining held-out days.

    SWITCH TO SOYBEAN
        Evaluate soybean from the decision-date soil state.

    Future rainfall is used only after the decision has already
    been generated.
    """

    prepared = _prepare_realized_action(
        decision=decision,
        actual_future=actual_future,
        soil_type=soil_type,
        initial_moisture_mm=initial_moisture_mm,
    )

    if prepared is None:
        return 0.0

    (
        crop_name,
        relevant_future,
        sowing_moisture,
    ) = prepared

    return calculate_realized_establishment(
        actual_future=relevant_future,
        crop_name=crop_name,
        soil_type=soil_type,
        initial_moisture_mm=sowing_moisture,
    )

# ---------------------------------------------------------------------
# ALL-ACTION EVALUATION
# ---------------------------------------------------------------------

def evaluate_all_actions(
    actual_future,
    initial_moisture_mm,
    soil_type,
):
    """
    Evaluate all possible actions against the same held-out future.

    This function must only be called after all decision-making
    methods have already produced their decisions.
    """

    outcomes = {}

    for action in ACTIONS:
        outcomes[action] = calculate_realized_outcome(
            decision=action,
            actual_future=actual_future,
            soil_type=soil_type,
            initial_moisture_mm=initial_moisture_mm,
        )

    return outcomes


def evaluate_all_action_diagnostics(
    actual_future,
    initial_moisture_mm,
    soil_type,
):
    """
    Evaluate continuous establishment diagnostics for all
    candidate actions on the same held-out future.
    """

    diagnostics = {}

    for action in ACTIONS:
        diagnostics[action] = calculate_realized_diagnostics(
            decision=action,
            actual_future=actual_future,
            soil_type=soil_type,
            initial_moisture_mm=initial_moisture_mm,
        )

    return diagnostics

def determine_best_actions(action_outcomes):
    """
    Determine the best realized action(s).

    Multiple actions may tie because the current outcome proxy is
    binary.
    """

    if not action_outcomes:
        raise ValueError(
            "No action outcomes were supplied."
        )

    best_outcome = max(
        action_outcomes.values()
    )

    best_actions = [
        action
        for action, outcome in action_outcomes.items()
        if outcome == best_outcome
    ]

    return best_actions, best_outcome


def calculate_decision_regret(
    selected_action,
    action_outcomes,
):
    """
    Calculate realized decision regret.

        regret =
            best realized outcome
            -
            selected action outcome

    With the current binary outcome proxy, regret is 0 or 1.
    """

    if selected_action not in action_outcomes:
        raise ValueError(
            f"Selected action {selected_action!r} "
            "is missing from action outcomes."
        )

    best_outcome = max(
        action_outcomes.values()
    )

    selected_outcome = action_outcomes[
        selected_action
    ]

    return float(
        best_outcome - selected_outcome
    )


def selected_action_is_best(
    selected_action,
    best_actions,
):
    """Return whether the selected action is among the best actions."""

    return selected_action in best_actions


# ---------------------------------------------------------------------
# SINGLE-DATE BACKTEST
# ---------------------------------------------------------------------

def run_single_date(
    data,
    decision_date,
):
    """
    Evaluate all three approaches on one historical decision date.

    The information boundary is:

        training:
            dates < decision_date

        decision:
            generated using training only

        evaluation:
            decision_date + 1 through +14 days
    """

    decision_timestamp = pd.Timestamp(
        decision_date
    ).normalize()

    # -------------------------------------------------------------
    # STRICT PRE-DECISION TRAINING
    # -------------------------------------------------------------

    training = b.get_training_data(
        data,
        decision_timestamp,
    )

    # -------------------------------------------------------------
    # STRICT POST-DECISION HOLDOUT
    # -------------------------------------------------------------
    #
    # We deliberately do not use the generic helper here if its
    # current semantics include the decision date. The validation
    # contract for this script is explicitly:
    #
    #     decision + 1 ... decision + HORIZON
    #
    # -------------------------------------------------------------

    future_start = (
        decision_timestamp
        + pd.Timedelta(days=1)
    )

    future_end_exclusive = (
        decision_timestamp
        + pd.Timedelta(days=HORIZON + 1)
    )

    actual_future = data.loc[
        (data["date"] >= future_start)
        & (data["date"] < future_end_exclusive)
    ].copy()

    actual_future = (
        actual_future
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(actual_future) != HORIZON:
        raise ValueError(
            f"Incomplete post-decision future window for "
            f"{decision_timestamp.date()}: expected "
            f"{HORIZON} days, found {len(actual_future)}."
        )

    expected_future_dates = pd.date_range(
        start=future_start,
        periods=HORIZON,
        freq="D",
    )

    actual_future_dates = pd.DatetimeIndex(
        actual_future["date"]
    )

    if not actual_future_dates.equals(
        expected_future_dates
    ):
        raise ValueError(
            "Post-decision future observations are not "
            "consecutive calendar days."
        )

    # -------------------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------------------

    initial_state = b.get_initial_state(
        training,
        decision_timestamp,
    )

    # -------------------------------------------------------------
    # RECONSTRUCT INITIAL SOIL MOISTURE
    # -------------------------------------------------------------

    initial_moisture = estimate_initial_moisture(
        data=data,
        decision_date=decision_timestamp,
        soil_type=SOIL_TYPE,
    )

    # -------------------------------------------------------------
    # WEATHER ONLY
    # -------------------------------------------------------------

    weather_decision = weather_only_baseline(
        training
    )

    # -------------------------------------------------------------
    # RULE BASED
    # -------------------------------------------------------------

    rule_decision = rule_based_baseline(
        training_data=training,
        current_moisture_mm=initial_moisture,
        soil_type=SOIL_TYPE,
    )

    # -------------------------------------------------------------
    # CROPLOGIC
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

    if croplogic_decision_name not in ACTIONS:
        raise ValueError(
            f"CropLogic returned unsupported decision: "
            f"{croplogic_decision_name!r}"
        )

    # -------------------------------------------------------------
    # HELD-OUT ACTION EVALUATION
    # -------------------------------------------------------------
    #
    # This happens only AFTER all decisions have been generated.
    # -------------------------------------------------------------

    action_outcomes = evaluate_all_actions(
        actual_future=actual_future,
        initial_moisture_mm=initial_moisture,
        soil_type=SOIL_TYPE,
    )

    action_diagnostics = evaluate_all_action_diagnostics(
        actual_future=actual_future,
        initial_moisture_mm=initial_moisture,
        soil_type=SOIL_TYPE,
    )

    sow_diagnostics = action_diagnostics[
        "SOW TODAY"
    ]

    wait_diagnostics = action_diagnostics[
        "WAIT 5 DAYS"
    ]

    switch_diagnostics = action_diagnostics[
        "SWITCH TO SOYBEAN"
    ]

    best_actions, best_outcome = (
        determine_best_actions(
            action_outcomes
        )
    )

    # -------------------------------------------------------------
    # SELECTED-ACTION OUTCOMES
    # -------------------------------------------------------------

    weather_outcome = action_outcomes[
        weather_decision
    ]

    rule_outcome = action_outcomes[
        rule_decision
    ]

    croplogic_outcome = action_outcomes[
        croplogic_decision_name
    ]

    # -------------------------------------------------------------
    # REGRET
    # -------------------------------------------------------------

    weather_regret = calculate_decision_regret(
        selected_action=weather_decision,
        action_outcomes=action_outcomes,
    )

    rule_regret = calculate_decision_regret(
        selected_action=rule_decision,
        action_outcomes=action_outcomes,
    )

    croplogic_regret = calculate_decision_regret(
        selected_action=croplogic_decision_name,
        action_outcomes=action_outcomes,
    )

    # -------------------------------------------------------------
    # BEST-ACTION FLAGS
    # -------------------------------------------------------------

    weather_best_action = selected_action_is_best(
        selected_action=weather_decision,
        best_actions=best_actions,
    )

    rule_best_action = selected_action_is_best(
        selected_action=rule_decision,
        best_actions=best_actions,
    )

    croplogic_best_action = selected_action_is_best(
        selected_action=croplogic_decision_name,
        best_actions=best_actions,
    )

    # -------------------------------------------------------------
    # ACTUAL HELD-OUT RAINFALL
    # -------------------------------------------------------------

    actual_total = float(
        actual_future["rainfall_mm"].sum()
    )

    return {
        "decision_date": decision_timestamp.strftime(
            "%Y-%m-%d"
        ),
        "evaluation_start": future_start.strftime(
            "%Y-%m-%d"
        ),
        "evaluation_end": (
            future_end_exclusive
            - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        "initial_state": initial_state,
        "initial_moisture_mm": initial_moisture,
        "actual_14d_rainfall_mm": actual_total,

        # Decisions
        "weather_only_decision": weather_decision,
        "rule_based_decision": rule_decision,
        "croplogic_decision": croplogic_decision_name,

        # Selected-action outcomes
        "weather_only_outcome": weather_outcome,
        "rule_based_outcome": rule_outcome,
        "croplogic_outcome": croplogic_outcome,

        # Every-action outcomes
        "sow_today_outcome": action_outcomes[
            "SOW TODAY"
        ],
        "wait_5_days_outcome": action_outcomes[
            "WAIT 5 DAYS"
        ],
        "switch_to_soybean_outcome": action_outcomes[
            "SWITCH TO SOYBEAN"
        ],

        "sow_min_moisture_margin_pct": (
            sow_diagnostics["minimum_moisture_margin_pct"]
        ),

        "sow_cumulative_moisture_deficit_pct_days": (
            sow_diagnostics[
                "cumulative_moisture_deficit_pct_days"
            ]
        ),

        "sow_successful_days": (
            sow_diagnostics["successful_days"]
        ),

        "sow_germination_days": (
            sow_diagnostics["germination_days"]
        ),

        "wait_min_moisture_margin_pct": (
            wait_diagnostics["minimum_moisture_margin_pct"]
        ),

        "wait_cumulative_moisture_deficit_pct_days": (
            wait_diagnostics[
                "cumulative_moisture_deficit_pct_days"
            ]
        ),

        "wait_successful_days": (
            wait_diagnostics["successful_days"]
        ),

        "wait_germination_days": (
            wait_diagnostics["germination_days"]
        ),

        "switch_min_moisture_margin_pct": (
            switch_diagnostics[
                "minimum_moisture_margin_pct"
            ]
        ),

        "switch_cumulative_moisture_deficit_pct_days": (
            switch_diagnostics[
                "cumulative_moisture_deficit_pct_days"
            ]
        ),

        "switch_successful_days": (
            switch_diagnostics["successful_days"]
        ),

        "switch_germination_days": (
            switch_diagnostics["germination_days"]
        ),

        # Best realized action
        "best_actions": ", ".join(
            best_actions
        ),
        "best_action_count": len(best_actions),
        "best_outcome": best_outcome,

        # Best-action flags
        "weather_only_best_action": weather_best_action,
        "rule_based_best_action": rule_best_action,
        "croplogic_best_action": croplogic_best_action,

        # Regret
        "weather_only_regret": weather_regret,
        "rule_based_regret": rule_regret,
        "croplogic_regret": croplogic_regret,

        # CropLogic probability outputs
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
    """Run the baseline comparison across all eligible dates."""

    rows = []

    backtest_dates = generate_backtest_dates(
        data
    )

    (
        start_month,
        start_day,
        end_month,
        end_day,
    ) = get_crop_sowing_window(CROP)


    # Windows formatting differs across platforms. Build the display
    # string directly instead of relying on %-d.

    crop_window_text = (
        f"{pd.Timestamp(year=2000, month=start_month, day=start_day).strftime('%B')} "
        f"{start_day} - "
        f"{pd.Timestamp(year=2000, month=end_month, day=end_day).strftime('%B')} "
        f"{end_day}"
    )

    print(
        f"Generated {len(backtest_dates)} eligible "
        f"historical decision dates."
    )

    print(
        f"Primary crop         : "
        f"{crops[CROP]['name']}"
    )

    print(
        f"Optimal sowing window: "
        f"{crop_window_text}"
    )

    print(
        f"Sampling interval    : "
        f"every {BACKTEST_INTERVAL_DAYS} days"
    )

    print(
        f"Held-out horizon     : "
        f"{HORIZON} days AFTER decision date"
    )

    print()

    for decision_date in backtest_dates:

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
            f"  CropLogic    : "
            f"{row['croplogic_decision']}"
        )

        print(
            f"  Best action  : "
            f"{row['best_actions']}"
        )

        print(
            f"  Held-out     : "
            f"{row['evaluation_start']} "
            f"to {row['evaluation_end']}"
        )

        print(
            f"  Actual 14d   : "
            f"{row['actual_14d_rainfall_mm']:.2f} mm"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PROBABILITY CALIBRATION
# ---------------------------------------------------------------------

CALIBRATION_BINS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]


def calculate_brier_score(predicted, observed):
    """
    Calculate the binary Brier score.

    Lower is better.

    Brier score:
        mean((predicted_probability - observed_outcome)^2)

    Parameters
    ----------
    predicted : iterable
        Predicted probabilities in [0, 1].

    observed : iterable
        Binary observed outcomes in {0, 1}.
    """

    predicted = pd.Series(
        predicted,
        dtype=float,
    )

    observed = pd.Series(
        observed,
        dtype=float,
    )

    if len(predicted) != len(observed):
        raise ValueError(
            "Predicted and observed values must have "
            "the same length."
        )

    if predicted.empty:
        raise ValueError(
            "Cannot calculate Brier score on empty data."
        )

    if not predicted.between(0.0, 1.0).all():
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    if not observed.isin([0.0, 1.0]).all():
        raise ValueError(
            "Observed outcomes must be binary 0/1."
        )

    return float(
        ((predicted - observed) ** 2).mean()
    )


def calculate_calibration_bins(
    predicted,
    observed,
    bins=None,
):
    """
    Calculate reliability/calibration statistics.

    Each row contains:

        probability bin
        number of observations
        mean predicted probability
        observed success rate

    The final bin includes probability == 1.0.
    """

    if bins is None:
        bins = CALIBRATION_BINS

    predicted = pd.Series(
        predicted,
        dtype=float,
    )

    observed = pd.Series(
        observed,
        dtype=float,
    )

    if len(predicted) != len(observed):
        raise ValueError(
            "Predicted and observed values must have "
            "the same length."
        )

    rows = []

    for index, (lower, upper) in enumerate(bins):

        if index == len(bins) - 1:
            mask = (
                (predicted >= lower)
                & (predicted <= upper)
            )
        else:
            mask = (
                (predicted >= lower)
                & (predicted < upper)
            )

        bin_predicted = predicted[mask]
        bin_observed = observed[mask]

        if len(bin_predicted) == 0:
            rows.append(
                {
                    "bin": f"{lower:.1f}-{upper:.1f}",
                    "count": 0,
                    "mean_predicted": None,
                    "observed_rate": None,
                }
            )
            continue

        rows.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": int(len(bin_predicted)),
                "mean_predicted": float(
                    bin_predicted.mean()
                ),
                "observed_rate": float(
                    bin_observed.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def print_probability_calibration(results):
    """
    Print Brier scores and reliability-bin statistics.

    Calibration is evaluated against the same held-out outcomes
    already used by the baseline comparison.

    No future rainfall is used to generate the probabilities.
    """

    probability_specs = [
        (
            "Cotton SOW",
            "cotton_germ_prob",
            "sow_today_outcome",
        ),
        (
            "Cotton WAIT",
            "wait_germ_prob",
            "wait_5_days_outcome",
        ),
        (
            "Soybean SWITCH",
            "soybean_germ_prob",
            "switch_to_soybean_outcome",
        ),
    ]

    print()
    print("=" * 120)
    print("CROPLOGIC-SAATHI PROBABILITY CALIBRATION")
    print("=" * 120)

    print(
        "Brier score: lower is better."
    )

    print(
        "Observed outcome is the held-out model-based "
        "establishment outcome for the corresponding action."
    )

    print()

    summary_rows = []

    for label, probability_column, outcome_column in probability_specs:

        predicted = results[
            probability_column
        ]

        observed = results[
            outcome_column
        ]

        brier = calculate_brier_score(
            predicted,
            observed,
        )

        summary_rows.append(
            {
                "action": label,
                "brier_score": brier,
                "mean_predicted": float(
                    predicted.mean()
                ),
                "observed_rate": float(
                    observed.mean()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        f"{'Action':<22}"
        f"{'Brier score':>15}"
        f"{'Mean predicted':>18}"
        f"{'Observed':>15}"
    )

    print("-" * 70)

    for row in summary.itertuples(index=False):

        print(
            f"{row.action:<22}"
            f"{row.brier_score:>15.3f}"
            f"{row.mean_predicted:>18.3f}"
            f"{row.observed_rate:>15.3f}"
        )

    # -------------------------------------------------------------
    # RELIABILITY BINS
    # -------------------------------------------------------------

    for label, probability_column, outcome_column in probability_specs:

        print()
        print("-" * 120)
        print(
            f"CALIBRATION BINS: {label}"
        )
        print("-" * 120)

        calibration = calculate_calibration_bins(
            predicted=results[
                probability_column
            ],
            observed=results[
                outcome_column
            ],
        )

        print(
            f"{'Bin':<12}"
            f"{'N':>8}"
            f"{'Mean predicted':>20}"
            f"{'Observed rate':>18}"
        )

        print("-" * 65)

        for row in calibration.itertuples(index=False):

            if row.count == 0:
                print(
                    f"{row.bin:<12}"
                    f"{0:>8}"
                    f"{'--':>20}"
                    f"{'--':>18}"
                )
            else:
                print(
                    f"{row.bin:<12}"
                    f"{row.count:>8}"
                    f"{row.mean_predicted:>20.3f}"
                    f"{row.observed_rate:>18.3f}"
                )

    print()
    print(
        "Calibration interpretation: within a well-populated "
        "probability bin, mean predicted probability should be "
        "reasonably close to the observed success rate."
    )

    print(
        "With only the current historical sample, sparse bins "
        "should not be treated as strong evidence of calibration."
    )

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def print_report(results):
    """Print baseline and decision-quality validation results."""

    if results.empty:
        print("No validation results generated.")
        return

    print()
    print("=" * 120)
    print("CROPLOGIC-SAATHI BASELINE COMPARISON")
    print("=" * 120)

    print(
        f"Crop                 : "
        f"{crops[CROP]['name']}"
    )

    print(
        f"Optimal window       : "
        f"{crops[CROP]['optimal_sowing_window']}"
    )

    print(
        f"Switch crop          : "
        f"{crops[SWITCH_CROP]['name']}"
    )

    print(
        f"Switch crop window   : "
        f"{crops[SWITCH_CROP]['optimal_sowing_window']}"
    )

    print(
        f"Soil                 : "
        f"{soils[SOIL_TYPE]['name']}"
    )

    print(
        f"Horizon              : "
        f"{HORIZON} days after decision"
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
    print("-" * 120)

    # ------------------------------------------------------------------
    # Decision distributions
    # ------------------------------------------------------------------

    print("\nDECISION DISTRIBUTIONS")
    print("----------------------")

    for column, label in [
        (
            "weather_only_decision",
            "Weather-only",
        ),
        (
            "rule_based_decision",
            "Rule-based",
        ),
        (
            "croplogic_decision",
            "CropLogic-Saathi",
        ),
    ]:
        print(f"\n{label}:")

        counts = (
            results[column]
            .value_counts()
            .reindex(
                ACTIONS,
                fill_value=0,
            )
        )

        for action, count in counts.items():
            print(
                f"  {action}: {count}"
            )

    # ------------------------------------------------------------------
    # Binary held-out outcomes
    # ------------------------------------------------------------------

    print("\nBINARY HELD-OUT ESTABLISHMENT OUTCOMES")
    print("---------------------------------------")

    print(
        f"Weather-only mean outcome : "
        f"{results['weather_only_outcome'].mean():.3f}"
    )

    print(
        f"Rule-based mean outcome   : "
        f"{results['rule_based_outcome'].mean():.3f}"
    )

    print(
        f"CropLogic mean outcome    : "
        f"{results['croplogic_outcome'].mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Best-action rates
    # ------------------------------------------------------------------

    print("\nBEST-ACTION RATES")
    print("-----------------")

    print(
        f"Weather-only : "
        f"{results['weather_only_best_action'].mean():.3f}"
    )

    print(
        f"Rule-based   : "
        f"{results['rule_based_best_action'].mean():.3f}"
    )

    print(
        f"CropLogic    : "
        f"{results['croplogic_best_action'].mean():.3f}"
    )

    print(
        f"\nBest-action tie rate: "
        f"{(results['best_action_count'] > 1).mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Binary regret
    # ------------------------------------------------------------------

    print("\nBINARY DECISION REGRET")
    print("----------------------")

    print(
        f"Weather-only mean regret : "
        f"{results['weather_only_regret'].mean():.3f}"
    )

    print(
        f"Rule-based mean regret   : "
        f"{results['rule_based_regret'].mean():.3f}"
    )

    print(
        f"CropLogic mean regret    : "
        f"{results['croplogic_regret'].mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Action outcome success rates
    # ------------------------------------------------------------------

    print("\nACTION OUTCOME SUCCESS RATES")
    print("-----------------------------")

    print(
        f"SOW TODAY: "
        f"{results['sow_today_outcome'].mean():.3f}"
    )

    print(
        f"WAIT 5 DAYS: "
        f"{results['wait_5_days_outcome'].mean():.3f}"
    )

    print(
        f"SWITCH TO SOYBEAN: "
        f"{results['switch_to_soybean_outcome'].mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Phase 2: continuous establishment diagnostics
    # ------------------------------------------------------------------

    print("\nCONTINUOUS ESTABLISHMENT DIAGNOSTICS")
    print("--------------------------------------")

    diagnostic_actions = {
        "SOW TODAY": "sow",
        "WAIT 5 DAYS": "wait",
        "SWITCH TO SOYBEAN": "switch",
    }

    for action, prefix in diagnostic_actions.items():

        margin_column = (
            f"{prefix}_min_moisture_margin_pct"
        )

        deficit_column = (
            f"{prefix}_cumulative_moisture_deficit_pct_days"
        )

        successful_days_column = (
            f"{prefix}_successful_days"
        )

        germination_days_column = (
            f"{prefix}_germination_days"
        )

        mean_margin = (
            results[margin_column].mean()
        )

        mean_deficit = (
            results[deficit_column].mean()
        )

        successful_day_ratio = (
            results[successful_days_column]
            / results[germination_days_column]
        ).mean()

        print(
            f"{action}:"
        )

        print(
            f"  Mean minimum moisture margin: "
            f"{mean_margin:.2f} percentage points"
        )

        print(
            f"  Mean cumulative moisture deficit: "
            f"{mean_deficit:.2f} %-days"
        )

        print(
            f"  Mean successful-day ratio: "
            f"{successful_day_ratio:.3f}"
        )

    # ------------------------------------------------------------------
    # Detailed per-date table
    # ------------------------------------------------------------------

    print()
    print("-" * 120)

    display_columns = [
        "decision_date",
        "evaluation_start",
        "evaluation_end",
        "actual_14d_rainfall_mm",
        "weather_only_decision",
        "rule_based_decision",
        "croplogic_decision",
        "sow_today_outcome",
        "wait_5_days_outcome",
        "switch_to_soybean_outcome",

        "sow_min_moisture_margin_pct",
        "sow_cumulative_moisture_deficit_pct_days",
        "sow_successful_days",
        "sow_germination_days",

        "wait_min_moisture_margin_pct",
        "wait_cumulative_moisture_deficit_pct_days",
        "wait_successful_days",
        "wait_germination_days",

        "switch_min_moisture_margin_pct",
        "switch_cumulative_moisture_deficit_pct_days",
        "switch_successful_days",
        "switch_germination_days",

        "best_actions",
        "best_action_count",
        "croplogic_best_action",
        "croplogic_regret",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # ------------------------------------------------------------------
    # Probability summary
    # ------------------------------------------------------------------

    print("\nPREDICTED ESTABLISHMENT PROBABILITIES")
    print("--------------------------------------")

    print(
        f"Cotton / SOW TODAY mean probability : "
        f"{results['cotton_germ_prob'].mean():.3f}"
    )

    print(
        f"WAIT mean probability               : "
        f"{results['wait_germ_prob'].mean():.3f}"
    )

    print(
        f"Soybean / SWITCH mean probability   : "
        f"{results['soybean_germ_prob'].mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Decision agreement
    # ------------------------------------------------------------------

    print("\nDECISION AGREEMENT")
    print("------------------")

    weather_agreement = (
        results["weather_only_decision"]
        == results["croplogic_decision"]
    )

    rule_agreement = (
        results["rule_based_decision"]
        == results["croplogic_decision"]
    )

    print(
        f"Weather-only vs CropLogic : "
        f"{weather_agreement.mean():.3f}"
    )

    print(
        f"Rule-based vs CropLogic   : "
        f"{rule_agreement.mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Probability calibration
    # ------------------------------------------------------------------

    print_probability_calibration(results)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    print("\nINTERPRETATION")
    print("--------------")

    print(
        "Validation uses decision dates inside the documented "
        "crop sowing window."
    )

    print(
        "Each decision is generated using information available "
        "strictly before the decision date."
    )

    print(
        "Held-out rainfall covers the 14 calendar days after "
        "the decision."
    )

    print(
        "The binary establishment outcome is a model-based "
        "historical proxy, not field validation."
    )

    print(
        "Continuous establishment diagnostics are derived from "
        "the existing soil-water and crop-establishment model."
    )

    print(
        "They are diagnostic measures and are not biological "
        "guarantees or a replacement for the binary outcome."
    )

    print(
        "Best-action ties are retained rather than forcing a "
        "continuous diagnostic winner."
    )

    print(
        "These results do not establish economic superiority "
        "or field-level impact."
    )



    # -------------------------------------------------------------
    # DECISION DISTRIBUTION
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("DECISION DISTRIBUTION")
    print("=" * 120)

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

        for decision in ACTIONS:
            print(
                f"  {decision:<20}: "
                f"{int(counts.get(decision, 0))}"
            )

    # -------------------------------------------------------------
    # OUTCOME PROXY
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("HELD-OUT OUTCOME PROXY")
    print("=" * 120)

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
    # BEST-ACTION RATE
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("BEST-ACTION RATE")
    print("=" * 120)

    weather_best_rate = (
        results["weather_only_best_action"]
        .mean()
    )

    rule_best_rate = (
        results["rule_based_best_action"]
        .mean()
    )

    croplogic_best_rate = (
        results["croplogic_best_action"]
        .mean()
    )

    print(
        f"Weather-only best-action rate : "
        f"{weather_best_rate:.3f}"
    )

    print(
        f"Rule-based best-action rate   : "
        f"{rule_best_rate:.3f}"
    )

    print(
        f"CropLogic best-action rate    : "
        f"{croplogic_best_rate:.3f}"
    )

    # -------------------------------------------------------------
    # BEST-ACTION TIE RATE
    # -------------------------------------------------------------

    tie_rate = (
        results["best_action_count"] > 1
    ).mean()

    print()
    print(
        f"Best-action tie rate          : "
        f"{tie_rate:.3f}"
    )

    print(
        "A high tie rate means the binary outcome proxy "
        "cannot strongly distinguish among actions."
    )

    # -------------------------------------------------------------
    # REGRET
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("REALIZED DECISION REGRET")
    print("=" * 120)

    weather_regret = (
        results["weather_only_regret"]
        .mean()
    )

    rule_regret = (
        results["rule_based_regret"]
        .mean()
    )

    croplogic_regret = (
        results["croplogic_regret"]
        .mean()
    )

    print(
        f"Weather-only mean regret : "
        f"{weather_regret:.3f}"
    )

    print(
        f"Rule-based mean regret   : "
        f"{rule_regret:.3f}"
    )

    print(
        f"CropLogic mean regret    : "
        f"{croplogic_regret:.3f}"
    )

    print()
    print(
        "Regret = best realized outcome "
        "minus selected action outcome."
    )

    print(
        "The current binary outcome proxy makes regret "
        "0 or 1 only."
    )

    # -------------------------------------------------------------
    # ACTION OUTCOME SUMMARY
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("REALIZED OUTCOME BY ACTION")
    print("=" * 120)

    print(
        f"SOW TODAY success rate       : "
        f"{results['sow_today_outcome'].mean():.3f}"
    )

    print(
        f"WAIT 5 DAYS success rate     : "
        f"{results['wait_5_days_outcome'].mean():.3f}"
    )

    print(
        f"SWITCH TO SOYBEAN rate       : "
        f"{results['switch_to_soybean_outcome'].mean():.3f}"
    )

    # -------------------------------------------------------------
    # PROBABILITY SUMMARY
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("CROPLOGIC PROBABILITY SUMMARY")
    print("=" * 120)

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

    # -------------------------------------------------------------
    # DECISION AGREEMENT
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("DECISION AGREEMENT")
    print("=" * 120)

    weather_crop_logic_agreement = (
        results["weather_only_decision"]
        == results["croplogic_decision"]
    ).mean()

    rule_crop_logic_agreement = (
        results["rule_based_decision"]
        == results["croplogic_decision"]
    ).mean()

    print(
        f"Weather-only vs CropLogic agreement : "
        f"{weather_crop_logic_agreement:.3f}"
    )

    print(
        f"Rule-based vs CropLogic agreement   : "
        f"{rule_crop_logic_agreement:.3f}"
    )

    # -------------------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------------------

    print()
    print("=" * 120)
    print("INTERPRETATION")
    print("=" * 120)

    print(
        "The primary validation population is restricted to the "
        "documented optimal sowing window for the target crop."
    )

    print(
        "The optimal sowing window is used as a validation-date "
        "eligibility rule, not as a hard biological failure cutoff."
    )

    print(
        "All decision methods receive only observations strictly "
        "before each historical decision date."
    )

    print(
        "The 14-day evaluation period begins on the day AFTER "
        "the decision date."
    )

    print(
        "Future rainfall is therefore held out from the decision "
        "and used only after the decision is generated."
    )

    print(
        "The weather-only and rule-based approaches are intentionally "
        "simple baselines."
    )

    print(
        "For each date, all three possible actions are evaluated "
        "against the same held-out rainfall trajectory."
    )

    print(
        "The realized outcome is a simplified model-based "
        "establishment proxy."
    )

    print(
        "It is not field-measured establishment and should not "
        "be interpreted as causal or agronomic impact evidence."
    )

    print(
        "Best-action rate must be interpreted together with the "
        "best-action tie rate because the current outcome is binary."
    )

    print(
        "A high best-action rate with many ties does not demonstrate "
        "that one decision method is substantially more accurate."
    )

    print(
        "The current validation does not establish economic superiority."
    )

    print(
        "The next validation stage should evaluate probability "
        "calibration and a more informative decision/economic "
        "outcome metric before making strong performance claims."
    )

    # -------------------------------------------------------------
    # PROBABILITY CALIBRATION
    # -------------------------------------------------------------

    print_probability_calibration(
        results
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