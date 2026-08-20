"""
CropLogic-Saathi decision engine.

Core flow:

    calibrated rainfall transitions
        -> Monte Carlo rainfall scenarios
        -> soil-water balance
        -> crop establishment probability
        -> economic comparison
        -> SOW / WAIT / SWITCH decision

Important:
    Simulation probabilities are scenario estimates, not guarantees.
    Economic assumptions are kept separate from the physical simulation.
"""

import numpy as np

from .crop_data import crops
from .soil_data import soils
from .monte_carlo_weather import generate_monte_carlo_scenarios
from .soil_water import simulate_soil_water
from .crop_establishment import evaluate_establishment
from .economic_engine import compare_all_decisions


# ============================================================
# INPUT VALIDATION
# ============================================================

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
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

    if soil_type not in soils:
        raise ValueError(
            f"Unknown soil type: {soil_type}"
        )

    if current_moisture_mm < 0:
        raise ValueError(
            "current_moisture_mm cannot be negative."
        )

    field_capacity = soils[soil_type][
        "field_capacity_mm"
    ]

    if current_moisture_mm > field_capacity:
        raise ValueError(
            "current_moisture_mm cannot exceed "
            "field capacity."
        )

    if rainfall_yesterday_mm < 0:
        raise ValueError(
            "rainfall_yesterday_mm cannot be negative."
        )

    matrix = np.asarray(
        transition_matrix,
        dtype=float,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "transition_matrix must have shape (3, 3)."
        )

    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(
            "transition probabilities must be "
            "between 0 and 1."
        )

    if not np.allclose(
        matrix.sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Each transition-matrix row must sum to 1."
        )

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    if days_to_simulate <= 0:
        raise ValueError(
            "days_to_simulate must be greater than zero."
        )


# ============================================================
# DAILY EVAPOTRANSPIRATION
# ============================================================

def calculate_daily_et(
    rainfall_scenario,
):
    """
    Estimate daily evapotranspiration.

    The current rainfall scenarios do not contain temperature
    information, so a transparent constant ET assumption is used.

    This is a model assumption, not a measured value.
    """

    et_mm_per_day = 5.0

    return [
        et_mm_per_day
        for _ in rainfall_scenario
    ]


# ============================================================
# RAINFALL -> SOIL WATER
# ============================================================

def simulate_rainfall_soil_water(
    rainfall_scenario,
    soil_type,
    initial_moisture_mm,
):
    """
    Convert one rainfall scenario into a soil-water trajectory.
    """

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


# ============================================================
# ESTABLISHMENT PROBABILITY
# ============================================================

def _scenario_establishment_probability(
    scenarios,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Calculate establishment probability across
    Monte Carlo weather scenarios.

    Returns:
        probability:
            Fraction of scenarios where establishment succeeds.

        trajectories:
            Simulated soil-water trajectories.
    """

    if not scenarios:
        return 0.0, np.asarray([])

    successful = 0
    trajectories = []

    for scenario in scenarios:

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

        trajectory = [
            initial_moisture_mm
        ]

        trajectory.extend(
            result["final_water_mm"]
            for result in soil_water_results
        )

        trajectories.append(
            trajectory
        )

    probability = (
        successful / len(scenarios)
    )

    return (
        float(probability),
        np.asarray(
            trajectories,
            dtype=float,
        ),
    )


# ============================================================
# INITIAL RAINFALL STATE
# ============================================================

def _get_initial_state(
    rainfall_yesterday_mm,
):
    """
    Convert recent rainfall into the starting
    Markov rainfall state.

    Thresholds are current model assumptions and should
    eventually be calibrated against project rainfall data.
    """

    if rainfall_yesterday_mm <= 0:
        return "dry"

    if rainfall_yesterday_mm < 10:
        return "drizzle"

    return "rain"


# ============================================================
# DECISION CONFIDENCE
# ============================================================

def _calculate_confidence(
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
):
    """
    Calculate a simple decision-confidence indicator.

    This measures separation between the best and
    second-best establishment probabilities.

    It is NOT a statistical confidence interval.
    """

    probabilities = [
        germ_prob_today,
        germ_prob_wait,
        germ_prob_soybean,
    ]

    ordered = sorted(
        probabilities,
        reverse=True,
    )

    if len(ordered) < 2:
        return 0.0

    separation = (
        ordered[0] - ordered[1]
    )

    return float(
        min(
            max(
                separation * 2.0,
                0.0,
            ),
            1.0,
        )
    )


# ============================================================
# MAIN DECISION FUNCTION
# ============================================================

def make_decision(
    crop_name,
    soil_type,
    current_moisture_mm,
    rainfall_yesterday_mm,
    transition_matrix,
    num_simulations=500,
    days_to_simulate=14,
    random_seed=42,
):
    """
    Generate a probabilistic pre-sowing decision.

    The engine compares:

        1. SOW TODAY
        2. WAIT 5 DAYS
        3. SWITCH TO SOYBEAN

    Returns:
        dict containing:

        - decision
        - economic comparison
        - establishment probabilities
        - Monte Carlo trajectories
        - uncertainty information
        - assumptions

    Important:
        SOW / WAIT / SWITCH are decision-support outputs.
        They are not guarantees of crop success.
    """

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    _validate_inputs(
        crop_name=crop_name,
        soil_type=soil_type,
        current_moisture_mm=current_moisture_mm,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        transition_matrix=transition_matrix,
        num_simulations=num_simulations,
        days_to_simulate=days_to_simulate,
    )

    # ========================================================
    # INITIAL WEATHER STATE
    # ========================================================

    initial_state = _get_initial_state(
        rainfall_yesterday_mm
    )

    # ========================================================
    # SCENARIO A — SOW TODAY
    # ========================================================

    sow_scenarios = generate_monte_carlo_scenarios(
        transition_matrix=transition_matrix,
        num_days=days_to_simulate,
        num_simulations=num_simulations,
        initial_state=initial_state,
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

    # ========================================================
    # SCENARIO B — WAIT 5 DAYS
    # ========================================================

    wait_days = 5

    wait_scenarios = generate_monte_carlo_scenarios(
        transition_matrix=transition_matrix,
        num_days=wait_days + days_to_simulate,
        num_simulations=num_simulations,
        initial_state=initial_state,
        random_seed=random_seed + 1,
    )

    wait_successes = 0
    wait_trajectories = []

    crop_germination_days = crops[
        crop_name
    ]["germination_days"]

    for scenario in wait_scenarios:

        # ----------------------------------------------------
        # Simulate the 5-day waiting period
        # ----------------------------------------------------

        wait_weather = scenario[
            :wait_days
        ]

        wait_soil_results = (
            simulate_rainfall_soil_water(
                rainfall_scenario=wait_weather,
                soil_type=soil_type,
                initial_moisture_mm=current_moisture_mm,
            )
        )

        if wait_soil_results:

            wait_moisture = (
                wait_soil_results[-1][
                    "final_water_mm"
                ]
            )

        else:

            wait_moisture = (
                current_moisture_mm
            )

        # ----------------------------------------------------
        # Simulate crop establishment after waiting
        # ----------------------------------------------------

        future_scenario = scenario[
            wait_days:
        ]

        if len(future_scenario) < crop_germination_days:
            continue

        soil_water_results = (
            simulate_rainfall_soil_water(
                rainfall_scenario=future_scenario,
                soil_type=soil_type,
                initial_moisture_mm=wait_moisture,
            )
        )

        establishment = evaluate_establishment(
            soil_water_results=soil_water_results,
            crop=crop_name,
            soil_type=soil_type,
        )

        if establishment[
            "establishment_success"
        ]:
            wait_successes += 1

        wait_trajectory = [
            current_moisture_mm
        ]

        # Include the moisture evolution during
        # the waiting period.
        wait_trajectory.extend(
            result["final_water_mm"]
            for result in wait_soil_results
        )

        # Include the moisture evolution after sowing.
        wait_trajectory.extend(
            result["final_water_mm"]
            for result in soil_water_results
        )

        wait_trajectories.append(
            wait_trajectory
        )

    if wait_scenarios:

        germ_prob_wait = (
            wait_successes
            / len(wait_scenarios)
        )

    else:

        germ_prob_wait = 0.0

    # ========================================================
    # SCENARIO C — SWITCH TO SOYBEAN
    # ========================================================

    soybean_scenarios = sow_scenarios

    germ_prob_soybean, _ = (
        _scenario_establishment_probability(
            scenarios=soybean_scenarios,
            crop_name="soybean",
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )
    )

    # ========================================================
    # ECONOMIC COMPARISON
    # ========================================================
    #
    # The physical model produces establishment probabilities.
    #
    # The economic engine converts those probabilities into
    # expected monetary outcomes.
    #
    # Economic assumptions remain inside economic_engine.py.
    #

    economic_comparison = (
        compare_all_decisions(
            crop_name=crop_name,
            soil_type=soil_type,
            germ_prob_today=germ_prob_today,
            germ_prob_wait=germ_prob_wait,
            germ_prob_soybean=germ_prob_soybean,
            rainfall_yesterday_mm=rainfall_yesterday_mm,
            current_moisture_mm=current_moisture_mm,
        )
    )

    economic_decision = (
        economic_comparison[
            "best_decision"
        ]
    )

    # ========================================================
    # MAP ECONOMIC DECISION
    # ========================================================

    if economic_decision == "Sow Today":

        decision = "SOW TODAY"

    elif economic_decision == "Wait 5 Days":

        decision = "WAIT 5 DAYS"

    elif economic_decision == "Switch to Soybean":

        decision = "SWITCH TO SOYBEAN"

    else:

        decision = "WAIT 5 DAYS"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = _calculate_confidence(
        germ_prob_today,
        germ_prob_wait,
        germ_prob_soybean,
    )

    # ========================================================
    # DECISION STATUS ICON
    # ========================================================

    if decision == "SOW TODAY":

        color = "🟢"

    elif decision == "WAIT 5 DAYS":

        color = "🟡"

    else:

        color = "🔴"

    # ========================================================
    # CROP / SOIL INFORMATION
    # ========================================================

    crop = crops[crop_name]
    soil = soils[soil_type]

    min_moisture_mm = (
        soil["field_capacity_mm"]
        * crop["min_moisture_pct"]
        / 100.0
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "decision": decision,

        "economic_comparison": (
            economic_comparison
        ),

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

        "wait_simulations": np.asarray(
            wait_trajectories,
            dtype=float,
        ),

        "current_moisture": float(
            current_moisture_mm
        ),

        "min_moisture_required": float(
            min_moisture_mm
        ),

        "initial_rainfall_state": (
            initial_state
        ),

        "num_simulations": (
            num_simulations
        ),

        "days_to_simulate": (
            days_to_simulate
        ),

        "assumptions": {
            "daily_et_mm": 5.0,

            "wait_days": wait_days,

            "economic_decision_policy": (
                "Final recommendation is selected "
                "using expected economic outcome "
                "across SOW, WAIT and SWITCH."
            ),

            "confidence_definition": (
                "Probability separation between "
                "the best and second-best option; "
                "not a statistical confidence interval."
            ),

            "simulation_probability_note": (
                "Establishment probabilities are "
                "Monte Carlo scenario estimates and "
                "are not guarantees of crop success."
            ),
        },
    }