"""
CropLogic-Saathi decision engine.

Core flow:

    calibrated rainfall transitions
        -> Monte Carlo rainfall scenarios
        -> soil-water balance
        -> crop establishment probability
        -> SOW / WAIT / SWITCH decision

Important:
    Simulation probabilities are scenario estimates, not guarantees.
    Economic assumptions are kept separate from the physical simulation.
"""

import numpy as np

from src.crop_data import crops
from src.soil_data import soils
from src.monte_carlo_weather import generate_monte_carlo_scenarios
from src.soil_water import simulate_soil_water
from src.crop_establishment import evaluate_establishment
from src.economic_engine import compare_all_decisions


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

    if current_moisture_mm < 0:
        raise ValueError(
            "current_moisture_mm cannot be negative."
        )

    field_capacity = soils[soil_type]["field_capacity_mm"]

    if current_moisture_mm > field_capacity:
        raise ValueError(
            "current_moisture_mm cannot exceed field capacity."
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

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    if days_to_simulate <= 0:
        raise ValueError(
            "days_to_simulate must be greater than zero."
        )


def calculate_daily_et(
    rainfall_scenario,
):
    """
    Estimate daily ET for a rainfall scenario.

    This is intentionally a simple transparent assumption because
    the current rainfall scenario does not contain temperature data.

    Returns:
        list[float]: daily ET values in mm.
    """

    et_mm_per_day = 5.0

    return [
        et_mm_per_day
        for _ in rainfall_scenario
    ]


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


def _scenario_establishment_probability(
    scenarios,
    crop_name,
    soil_type,
    initial_moisture_mm,
):
    """
    Calculate establishment probability across Monte Carlo scenarios.
    """

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

        trajectories.append(
            [
                initial_moisture_mm
            ]
            + [
                result["final_water_mm"]
                for result in soil_water_results
            ]
        )

    probability = successful / len(scenarios)

    return probability, np.asarray(trajectories)


def _get_initial_state(
    rainfall_yesterday_mm,
):
    """
    Convert recent rainfall into the starting Markov rainfall state.

    These thresholds are model assumptions and should eventually be
    calibrated against the project's rainfall-state preprocessing.
    """

    if rainfall_yesterday_mm <= 0:
        return "dry"

    if rainfall_yesterday_mm < 10:
        return "drizzle"

    return "rain"


def _calculate_confidence(
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
):
    """
    Calculate a simple decision confidence indicator.

    This represents separation between the best and competing
    establishment probabilities. It is not a statistical confidence
    interval.
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

    separation = ordered[0] - ordered[1]

    return float(
        min(max(separation * 2.0, 0.0), 1.0)
    )


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

    Returns:
        dict containing:

        - recommendation
        - establishment probabilities
        - Monte Carlo trajectories
        - uncertainty information
        - assumptions

    Notes:
        SOW / WAIT / SWITCH are decision-support outputs.
        They are not guarantees of crop success.
    """

    _validate_inputs(
        crop_name=crop_name,
        soil_type=soil_type,
        current_moisture_mm=current_moisture_mm,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        transition_matrix=transition_matrix,
        num_simulations=num_simulations,
        days_to_simulate=days_to_simulate,
    )

    initial_state = _get_initial_state(
        rainfall_yesterday_mm
    )

    # ---------------------------------------------------------
    # Scenario A: Sow today
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Scenario B: Wait
    # ---------------------------------------------------------

    wait_days = 5

    wait_scenarios = generate_monte_carlo_scenarios(
        transition_matrix=transition_matrix,
        num_days=wait_days + days_to_simulate,
        num_simulations=num_simulations,
        initial_state=initial_state,
        random_seed=random_seed + 1,
    )

    wait_initial_moistures = []

    for scenario in wait_scenarios:

        wait_soil_results = simulate_rainfall_soil_water(
            rainfall_scenario=scenario[:wait_days],
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )

        wait_initial_moistures.append(
            wait_soil_results[-1]["final_water_mm"]
        )

    wait_successes = 0
    wait_trajectories = []

    crop_germination_days = crops[crop_name]["germination_days"]

    for scenario, wait_moisture in zip(
        wait_scenarios,
        wait_initial_moistures,
    ):

        future_scenario = scenario[
            wait_days:
        ]

        if len(future_scenario) < crop_germination_days:
            continue

        soil_water_results = simulate_rainfall_soil_water(
            rainfall_scenario=future_scenario,
            soil_type=soil_type,
            initial_moisture_mm=wait_moisture,
        )

        establishment = evaluate_establishment(
            soil_water_results=soil_water_results,
            crop=crop_name,
            soil_type=soil_type,
        )

        if establishment["establishment_success"]:
            wait_successes += 1

        wait_trajectory = [
            current_moisture_mm
        ]

        wait_trajectory.extend(
            wait_initial_moistures[
                len(wait_trajectories):
            ]
            for _ in []
        )

        wait_trajectory.extend(
            result["final_water_mm"]
            for result in soil_water_results
        )

        wait_trajectories.append(
            wait_trajectory
        )

    germ_prob_wait = (
        wait_successes / num_simulations
    )

    # ---------------------------------------------------------
    # Scenario C: Switch to soybean
    # ---------------------------------------------------------

    soybean_scenarios = sow_scenarios

    germ_prob_soybean, _ = (
        _scenario_establishment_probability(
            scenarios=soybean_scenarios,
            crop_name="soybean",
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )
    )

        # ---------------------------------------------------------
    # Economic comparison
    # ---------------------------------------------------------
    #
    # The physical model produces establishment probabilities.
    # The economic engine converts those probabilities into
    # expected monetary outcomes.
    #
    # Economic assumptions are kept inside economic_engine.py
    # and are not treated as physical simulation probabilities.

    economic_comparison = compare_all_decisions(
        crop_name=crop_name,
        soil_type=soil_type,
        germ_prob_today=germ_prob_today,
        germ_prob_wait=germ_prob_wait,
        germ_prob_soybean=germ_prob_soybean,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        current_moisture_mm=current_moisture_mm,
    )

    economic_decision = economic_comparison["best_decision"]

    # Map the economic-engine labels to the public
    # decision-engine recommendation labels.
    if economic_decision == "Sow Today":
        decision = "SOW TODAY"

    elif economic_decision == "Wait 5 Days":
        decision = "WAIT 5 DAYS"

    elif economic_decision == "Switch to Soybean":
        decision = "SWITCH TO SOYBEAN"

    else:
        decision = "WAIT 5 DAYS"

    confidence = _calculate_confidence(
        germ_prob_today,
        germ_prob_wait,
        germ_prob_soybean,
    )

    if decision == "SOW TODAY":
        color = "🟢"
    elif decision == "WAIT 5 DAYS":
        color = "🟡"
    else:
        color = "🔴"

    crop = crops[crop_name]
    soil = soils[soil_type]

    min_moisture_mm = (
        soil["field_capacity_mm"]
        * crop["min_moisture_pct"]
        / 100.0
    )

    return {
        "decision": decision,
	"economic_comparison": economic_comparison,
        "color": color,
        "germ_prob_today": float(germ_prob_today),
        "germ_prob_wait": float(germ_prob_wait),
        "germ_prob_soybean": float(germ_prob_soybean),
        "confidence": confidence,
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
        "initial_rainfall_state": initial_state,
        "num_simulations": num_simulations,
        "days_to_simulate": days_to_simulate,
        "assumptions": {
            "daily_et_mm": 5.0,
            "wait_days": wait_days,
            "economic_decision_policy": (
                "Final recommendation is selected using "
                "expected economic outcome across SOW, WAIT "
                "and SWITCH."
            ),
            "confidence_definition": (
                "Probability separation between the "
                "best and second-best option; not a "
                "statistical confidence interval."
            ),
        },
    }