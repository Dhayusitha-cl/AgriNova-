"""
CropLogic-Saathi decision engine.

Flow:
    calibrated rainfall transitions
        -> Monte Carlo rainfall scenarios
        -> soil-water balance
        -> crop establishment probability
        -> economic comparison
        -> SOW / WAIT / SWITCH decision

Simulation probabilities are estimates, not guarantees.
"""

import math
import re
from datetime import date, datetime

import numpy as np

from .crop_data import crops
from .soil_data import soils
from .monte_carlo_weather import (
    generate_monte_carlo_scenarios,
    generate_calibrated_monte_carlo_scenarios,
)
from .soil_water import simulate_soil_water
from .crop_establishment import evaluate_establishment
from .economic_engine import compare_all_decisions
from .rainfall_states import classify_rainfall

from .forecast import WeatherForecast
from .calibration_artifact import CalibrationArtifact

def _validate_inputs(
    crop_name,
    soil_type,
    current_moisture_mm,
    rainfall_yesterday_mm,
    transition_matrix,
    num_simulations,
    days_to_simulate,
):
    """Validate decision-engine inputs."""

    if crop_name not in crops:
        raise ValueError(f"Unknown crop: {crop_name}")

    if soil_type not in soils:
        raise ValueError(f"Unknown soil type: {soil_type}")

    current_moisture_mm = float(
        current_moisture_mm
    )

    if not np.isfinite(current_moisture_mm):
        raise ValueError(
            "current_moisture_mm must be a finite number."
        )

    if current_moisture_mm < 0:
        raise ValueError(
            "current_moisture_mm cannot be negative."
        )

    field_capacity = soils[soil_type]["field_capacity_mm"]

    if current_moisture_mm > field_capacity:
        raise ValueError(
            "current_moisture_mm cannot exceed field capacity."
        )

    rainfall_yesterday_mm = float(
        rainfall_yesterday_mm
    )

    if not np.isfinite(rainfall_yesterday_mm):
        raise ValueError(
            "rainfall_yesterday_mm must be a finite number."
        )

    if rainfall_yesterday_mm < 0:
        raise ValueError(
            "rainfall_yesterday_mm cannot be negative."
        )

    if transition_matrix is not None:
        matrix = np.asarray(
            transition_matrix,
            dtype=float,
        )

        if matrix.shape != (3, 3):
            raise ValueError(
                "transition_matrix must have shape (3, 3)."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "transition probabilities must be finite numbers."
        )

        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError(
                "transition probabilities must be between 0 and 1."
            )

        if not np.allclose(
            matrix.sum(axis=1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                "Each transition-matrix row must sum to 1."
            )

    if isinstance(
        num_simulations,
        (bool, np.bool_),
    ) or not isinstance(
        num_simulations,
        (int, np.integer),
    ):
        raise ValueError(
            "num_simulations must be an integer."
         )

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    if isinstance(
        days_to_simulate,
        (bool, np.bool_),
    ) or not isinstance(
        days_to_simulate,
        (int, np.integer),
    ):
        raise ValueError(
            "days_to_simulate must be an integer."
        )

    if days_to_simulate <= 0:
        raise ValueError(
            "days_to_simulate must be greater than zero."
        )


def calculate_daily_et(rainfall_scenario):
    """
    Calculate daily evapotranspiration.

    Temperature is not currently part of the rainfall scenario,
    so a transparent fixed daily ET assumption is used.
    """

    return [5.0 for _ in rainfall_scenario]


def simulate_rainfall_soil_water(
    rainfall_scenario,
    soil_type,
    initial_moisture_mm,
):
    """Convert a rainfall scenario into a soil-water trajectory."""

    rainfall_series = [
        float(day["rainfall_mm"])
        for day in rainfall_scenario
    ]

    et_series = calculate_daily_et(
        rainfall_scenario
    )

    return simulate_soil_water(
        rainfall_series=rainfall_series,
        et_series=et_series,
        soil_type=soil_type,
        initial_water_mm=initial_moisture_mm,
    )


def _scenario_establishment_probability(
    scenarios,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Calculate establishment probability over Monte Carlo scenarios.

    Every scenario must contain at least as many days as the crop's
    germination period.
    """

    successful = 0
    trajectories = []

    crop_germination_days = int(
        crops[crop_name]["germination_days"]
    )

    for scenario in scenarios:

        scenario = list(scenario)

        if len(scenario) < crop_germination_days:
            raise ValueError(
                "Scenario is shorter than the crop germination period."
            )

        soil_water_results = simulate_rainfall_soil_water(
            rainfall_scenario=scenario,
            soil_type=soil_type,
            initial_moisture_mm=initial_moisture_mm,
        )

        establishment = evaluate_establishment(
            soil_water_results=soil_water_results,
            crop=crop_name,
            soil_type=soil_type,
        )

        if establishment["establishment_success"]:
            successful += 1

        trajectory = [float(initial_moisture_mm)]

        trajectory.extend(
            float(result["final_water_mm"])
            for result in soil_water_results
        )

        trajectories.append(trajectory)

    if not scenarios:
        return 0.0, np.empty((0, 0))

    probability = successful / len(scenarios)

    return (
        float(probability),
        np.asarray(
            trajectories,
            dtype=float,
        ),
    )

def _get_initial_state(rainfall_yesterday_mm):
    """
    Convert recent rainfall into the starting rainfall state
    using the project's centralized rainfall classification rule.
    """

    state = classify_rainfall(rainfall_yesterday_mm)

    if state == "missing":
        raise ValueError(
            "rainfall_yesterday_mm cannot be missing."
        )

    return state

def _calculate_confidence(
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
):
    """
    Calculate a simple separation-based confidence indicator.

    This is NOT a statistical confidence interval.
    """

    probabilities = [
        float(germ_prob_today),
        float(germ_prob_wait),
        float(germ_prob_soybean),
    ]

    ordered = sorted(
        probabilities,
        reverse=True,
    )

    if len(ordered) < 2:
        return 0.0

    separation = ordered[0] - ordered[1]

    return float(
        min(
            max(
                separation * 2.0,
                0.0,
            ),
            1.0,
        )
    )


def _safe_wait_trajectories(
    wait_scenarios,
    wait_days,
    crop_name,
    soil_type,
    current_moisture_mm,
):
    """
    Evaluate WAIT scenarios.

    The scenario contains:

        WAIT period
            +
        crop establishment period

    Therefore the full scenario is simulated, but establishment
    is evaluated only after the waiting period.

    Always returns one trajectory per Monte Carlo simulation.

    A WAIT scenario must contain enough days for both the
    waiting period and the crop establishment period. The
    function does not fabricate missing days with zero rainfall.
    """

    wait_trajectories = []
    successful = 0

    crop_germination_days = int(
        crops[crop_name]["germination_days"]
    )

    for scenario in wait_scenarios:
        scenario = list(scenario)

        required_days = (
            wait_days + crop_germination_days
        )

        if len(scenario) < required_days:
            raise ValueError(
                "WAIT scenario is shorter than the required "
                "wait plus crop germination period."
            )

        # -----------------------------------------------------
        # WAIT PERIOD
        # -----------------------------------------------------

        wait_weather = scenario[:wait_days]

        wait_results = simulate_rainfall_soil_water(
            rainfall_scenario=wait_weather,
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )

        if wait_results:
            wait_moisture = float(
                wait_results[-1]["final_water_mm"]
            )
        else:
            wait_moisture = float(
                current_moisture_mm
            )

        # -----------------------------------------------------
        # POST-WAIT ESTABLISHMENT PERIOD
        # -----------------------------------------------------

        future_scenario = scenario[wait_days:]

        future_results = simulate_rainfall_soil_water(
            rainfall_scenario=future_scenario,
            soil_type=soil_type,
            initial_moisture_mm=wait_moisture,
        )

        establishment = evaluate_establishment(
            soil_water_results=future_results,
            crop=crop_name,
            soil_type=soil_type,
        )

        if establishment["establishment_success"]:
            successful += 1

        # -----------------------------------------------------
        # FULL WAIT TRAJECTORY
        # -----------------------------------------------------

        trajectory = [
            float(current_moisture_mm)
        ]

        trajectory.extend(
            float(result["final_water_mm"])
            for result in wait_results
        )

        trajectory.extend(
            float(result["final_water_mm"])
            for result in future_results
        )

        wait_trajectories.append(
            trajectory
        )

    if not wait_scenarios:
        return 0.0, np.empty((0, 0))

    probability = successful / len(
        wait_scenarios
    )

    return (
        float(probability),
        np.asarray(
            wait_trajectories,
            dtype=float,
        ),
    )


def _calculate_decision_separation(all_decisions):
    """
    Calculate the absolute expected-profit gap between
    the best and second-best available decisions.

    This is an economic separation indicator, not a
    statistical confidence measure.
    """

    if not all_decisions:
        return 0.0

    profits = [
        float(item["expected_profit"])
        for item in all_decisions
    ]

    if not all(math.isfinite(profit) for profit in profits):
        raise ValueError(
            "Decision expected profits must be finite."
        )

    ordered = sorted(profits, reverse=True)

    if len(ordered) < 2:
        return 0.0

    return float(ordered[0] - ordered[1])



def _assess_crop_eligibility(crop_name, start_date=None):
    """
    Assess provisional alternative-crop eligibility.

    The configured sowing window is only a timing filter. It does not
    establish regional, soil, variety, or market suitability.
    """
    if crop_name not in crops:
        raise ValueError(f"Unknown crop: {crop_name}")

    if start_date is None:
        return {
            "eligible": True,
            "status": "timing_unassessed",
            "reason": (
                "No start_date supplied; sowing-window eligibility "
                "could not be assessed."
            ),
        }

    if isinstance(start_date, datetime):
        assessment_date = start_date.date()
    elif isinstance(start_date, date):
        assessment_date = start_date
    else:
        try:
            assessment_date = date.fromisoformat(str(start_date))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "start_date must be a date, datetime, or ISO date string."
            ) from exc

    window = crops[crop_name].get("optimal_sowing_window")
    if not isinstance(window, str):
        return {
            "eligible": False,
            "status": "missing_window",
            "reason": "No configured sowing window is available.",
        }

    match = re.fullmatch(
        r"\s*([A-Za-z]+)\s+(\d{1,2})\s*-\s*"
        r"([A-Za-z]+)\s+(\d{1,2})\s*",
        window,
    )
    if not match:
        return {
            "eligible": False,
            "status": "invalid_window",
            "reason": f"Cannot interpret configured window: {window!r}.",
        }

    start_month, start_day, end_month, end_day = match.groups()

    try:
        start_md = (
            datetime.strptime(start_month, "%B").month,
            int(start_day),
        )
        end_md = (
            datetime.strptime(end_month, "%B").month,
            int(end_day),
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid sowing window configured for {crop_name}: {window!r}"
        ) from exc

    try:
        date(2000, *start_md)
        date(2000, *end_md)
    except ValueError as exc:
        raise ValueError(
            f"Invalid sowing window configured for {crop_name}: {window!r}"
        ) from exc

    current_md = (assessment_date.month, assessment_date.day)

    if start_md <= end_md:
        within_window = start_md <= current_md <= end_md
    else:
        # Handles sowing windows crossing December–January.
        within_window = (
            current_md >= start_md
            or current_md <= end_md
        )

    if within_window:
        return {
            "eligible": True,
            "status": "within_configured_window",
            "reason": (
                "Assessment date falls within the configured sowing window; "
                "regional and soil suitability remain unverified."
            ),
        }

    return {
        "eligible": False,
        "status": "outside_configured_window",
        "reason": (
            f"Assessment date {assessment_date.isoformat()} is outside "
            f"the configured sowing window ({window})."
        ),
    }


def make_decision(
    crop_name,
    soil_type,
    current_moisture_mm,
    rainfall_yesterday_mm,
    transition_matrix,
    num_simulations=500,
    days_to_simulate=14,
    random_seed=42,
    start_date=None,
    rainfall_data=None,
    initial_state=None,
    calibration_artifact=None,
    forecast=None,
    forecast_weight=None,
):
    """
    Generate a probabilistic pre-sowing decision.

    Compares:

        1. SOW TODAY
        2. WAIT 5 DAYS
        3. SWITCH TO SOYBEAN

    Returns:
        dict containing recommendation, probabilities,
        trajectories, economic comparison and assumptions.
    """

    # =========================================================
    # VALIDATION
    # =========================================================

    _validate_inputs(
        crop_name=crop_name,
        soil_type=soil_type,
        current_moisture_mm=current_moisture_mm,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        transition_matrix=transition_matrix,
        num_simulations=num_simulations,
        days_to_simulate=days_to_simulate,
    )

    if initial_state is not None:
        valid_states = {
            "dry",
            "drizzle",
            "rain",
        }

        if initial_state not in valid_states:
            raise ValueError(
                "initial_state must be one of: "
                "'dry', 'drizzle', 'rain'."
            )

    if forecast is not None:
        if not isinstance(forecast, WeatherForecast):
            raise TypeError(
                "forecast must be a WeatherForecast instance."
            )

        if start_date is None:
            raise ValueError(
                "forecast requires start_date."
            )

        if transition_matrix is not None:
            raise ValueError(
                "forecast cannot be used with an explicit "
                "transition_matrix."
            )

        if forecast_weight is None:
            raise ValueError(
                "forecast_weight must be provided when "
                "forecast is supplied."
            )

    elif forecast_weight is not None:
        raise ValueError(
            "forecast_weight cannot be provided without forecast."
        )
    # =========================================================
    # INITIAL WEATHER STATE
    # =========================================================

    if initial_state is None:
        initial_state = _get_initial_state(
            rainfall_yesterday_mm
        )

    # =========================================================
    # DETERMINE REQUIRED SIMULATION HORIZON
    # =========================================================

    crop_germination_days = int(
        crops[crop_name]["germination_days"]
    )

    soybean_germination_days = int(
        crops["soybean"]["germination_days"]
    )

    simulation_days = max(
        days_to_simulate,
        crop_germination_days,
        soybean_germination_days,
        *(
            int(data["germination_days"])
            for data in crops.values()
        ),
    )


    # =========================================================
    # SOW TODAY
    # =========================================================

    if start_date is not None:
        sow_scenarios = generate_calibrated_monte_carlo_scenarios(
            start_date=start_date,
            num_days=simulation_days,
            num_simulations=num_simulations,
            initial_state=initial_state,
            random_seed=random_seed,
            rainfall_data=rainfall_data,
            calibration_artifact=calibration_artifact,
            forecast=forecast,
            forecast_weight=forecast_weight,
        )
    else:
        if transition_matrix is None:
            raise ValueError(
                "Either start_date or transition_matrix must be provided."
            )

        sow_scenarios = generate_monte_carlo_scenarios(
            transition_matrix=transition_matrix,
            num_days=simulation_days,
            num_simulations=num_simulations,
            initial_state=initial_state,
            rainfall_data=rainfall_data,
            random_seed=random_seed,
        )

    germ_prob_today, trajectories = (
        _scenario_establishment_probability(
            scenarios=sow_scenarios,
            crop_name=crop_name,
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )
    )

    # =========================================================
    # WAIT 5 DAYS
    # =========================================================

    wait_days = 5

    # We need enough days for:
    #
    #     WAIT period
    #          +
    #     crop establishment period
    #
    wait_simulation_days = (
        wait_days
        + max(
            days_to_simulate,
            crop_germination_days,
        )
    )

    if start_date is not None:
        wait_scenarios = generate_calibrated_monte_carlo_scenarios(
            start_date=start_date,
            num_days=wait_simulation_days,
            num_simulations=num_simulations,
            initial_state=initial_state,
            random_seed=random_seed + 1,
            rainfall_data=rainfall_data,
            calibration_artifact=calibration_artifact,
            forecast=forecast,
            forecast_weight=forecast_weight,
        )
    else:
        wait_scenarios = generate_monte_carlo_scenarios(
            transition_matrix=transition_matrix,
            num_days=wait_simulation_days,
            num_simulations=num_simulations,
            initial_state=initial_state,
            random_seed=random_seed + 1,
            rainfall_data=rainfall_data,
        )

    germ_prob_wait, wait_trajectories = (
        _safe_wait_trajectories(
            wait_scenarios=wait_scenarios,
            wait_days=wait_days,
            crop_name=crop_name,
            soil_type=soil_type,
            current_moisture_mm=current_moisture_mm,
        )
    )

    # =========================================================
    # LEGACY SOYBEAN PROBABILITY + ALTERNATIVE CROP EVALUATION
    # =========================================================

    # Preserve the existing soybean probability field for API/SDK
    # compatibility, even when soybean is not an eligible alternative.
    germ_prob_soybean, soybean_trajectories = (
        _scenario_establishment_probability(
            scenarios=sow_scenarios,
            crop_name="soybean",
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )
    )

    crop_eligibility = {}
    alternative_probabilities = {}
    alternative_trajectories = {}

    for candidate in crops:
        if candidate == crop_name:
            crop_eligibility[candidate] = {
                "eligible": False,
                "status": "primary_crop",
                "reason": "This is already the selected primary crop.",
            }
            continue

        eligibility = _assess_crop_eligibility(
            candidate,
            start_date=start_date,
        )
        crop_eligibility[candidate] = eligibility

        if not eligibility["eligible"]:
            continue

        probability, trajectories_for_crop = (
            _scenario_establishment_probability(
                scenarios=sow_scenarios,
                crop_name=candidate,
                soil_type=soil_type,
                initial_moisture_mm=current_moisture_mm,
            )
        )

        alternative_probabilities[candidate] = float(probability)
        alternative_trajectories[candidate] = trajectories_for_crop

    # =========================================================
    # ECONOMIC COMPARISON
    # =========================================================

    economic_comparison = compare_all_decisions(
        crop_name=crop_name,
        soil_type=soil_type,
        germ_prob_today=germ_prob_today,
        germ_prob_wait=germ_prob_wait,
        germ_prob_soybean=germ_prob_soybean,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        current_moisture_mm=current_moisture_mm,
        switch_probabilities=alternative_probabilities,
    )

    decision_separation = _calculate_decision_separation(
        economic_comparison["all_decisions"]
    )

    economic_decision = economic_comparison.get(
        "best_decision",
        "Wait 5 Days",
    )

    if economic_decision == "Sow Today":
        decision = "SOW TODAY"

    elif economic_decision == "Wait 5 Days":
        decision = "WAIT 5 DAYS"

    elif economic_decision.startswith("Switch to "):
        switched_crop = economic_decision[len("Switch to "):].upper()
        decision = f"SWITCH TO {switched_crop}"

    else:
        decision = "WAIT 5 DAYS"

    # =========================================================
    # CONFIDENCE
    # =========================================================

    confidence = _calculate_confidence(
        germ_prob_today=germ_prob_today,
        germ_prob_wait=germ_prob_wait,
        germ_prob_soybean=germ_prob_soybean,
    )

    # =========================================================
    # DISPLAY INDICATOR
    # =========================================================

    if decision == "SOW TODAY":
        color = "🟢"

    elif decision == "WAIT 5 DAYS":
        color = "🟡"

    else:
        color = "🔴"

    # =========================================================
    # CROP / SOIL REQUIREMENTS
    # =========================================================

    crop = crops[crop_name]
    soil = soils[soil_type]

    min_moisture_mm = (
        soil["field_capacity_mm"]
        * float(crop["min_moisture_pct"])
        / 100.0
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    return {
        "decision": decision,

        "economic_comparison": economic_comparison,

        "color": color,

        "germ_prob_today": float(
            germ_prob_today
        ),

        "germ_prob_wait": float(
            germ_prob_wait
        ),

        "germ_prob_soybean": float(
            germ_prob_soybean
        ),

        "confidence": float(
            confidence
        ),

        "trajectories": trajectories,

        "wait_simulations": wait_trajectories,

        "soybean_trajectories": soybean_trajectories,

        "alternative_probabilities": alternative_probabilities,

        "alternative_trajectories": alternative_trajectories,

        "crop_eligibility": crop_eligibility,

        "decision_separation": float(decision_separation),

        "current_moisture": float(
            current_moisture_mm
        ),

        "min_moisture_required": float(
            min_moisture_mm
        ),

        "initial_rainfall_state": initial_state,

        "num_simulations": int(
            num_simulations
        ),

        "days_to_simulate": int(
            days_to_simulate
        ),

        "assumptions": {
            "daily_et_mm": 5.0,

            "wait_days": wait_days,

            "economic_decision_policy": (
                "Final recommendation is selected using "
                "the economic comparison across SOW, WAIT "
                "and SWITCH."
            ),

            "confidence_definition": (
                "Probability separation between the "
                "best and second-best option; not a "
                "statistical confidence interval."
            ),

            "decision_separation_definition": (
                "Absolute expected-profit difference between the "
                "best and second-best evaluated decisions, per acre. "
                "Not a probability or statistical confidence measure."
            ),

            "simulation_note": (
                "Monte Carlo probabilities represent "
                "simulated scenarios and are not guarantees."
            ),
        },
    }
