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

    if transition_matrix is not None:
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

        # Protect against an accidentally short scenario.
        scenario = list(scenario)

        if len(scenario) < crop_germination_days:
            scenario.extend(
                [{"rainfall_mm": 0.0}]
                * (
                    crop_germination_days
                    - len(scenario)
                )
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
    Convert recent rainfall into the starting rainfall state.

    These thresholds are model assumptions and should eventually
    be calibrated against project rainfall-state preprocessing.
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
    """

    wait_trajectories = []
    successful = 0

    crop_germination_days = int(
        crops[crop_name]["germination_days"]
    )

    for scenario in wait_scenarios:

        scenario = list(scenario)

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

        if len(future_scenario) < crop_germination_days:
            future_scenario = list(
                future_scenario
            )

            while len(future_scenario) < crop_germination_days:
                future_scenario.append(
                    {
                        "rainfall_mm": 0.0
                    }
                )

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

    # =========================================================
    # INITIAL WEATHER STATE
    # =========================================================

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
        )
    else:
        wait_scenarios = generate_monte_carlo_scenarios(
            transition_matrix=transition_matrix,
            num_days=wait_simulation_days,
            num_simulations=num_simulations,
            initial_state=initial_state,
            random_seed=random_seed + 1,
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
    # SWITCH TO SOYBEAN
    # =========================================================

    soybean_scenarios = sow_scenarios

    germ_prob_soybean, soybean_trajectories = (
        _scenario_establishment_probability(
            scenarios=soybean_scenarios,
            crop_name="soybean",
            soil_type=soil_type,
            initial_moisture_mm=current_moisture_mm,
        )
    )

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
    )

    economic_decision = economic_comparison.get(
        "best_decision",
        "Wait 5 Days",
    )

    if economic_decision == "Sow Today":
        decision = "SOW TODAY"

    elif economic_decision == "Wait 5 Days":
        decision = "WAIT 5 DAYS"

    elif economic_decision == "Switch to Soybean":
        decision = "SWITCH TO SOYBEAN"

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

            "simulation_note": (
                "Monte Carlo probabilities represent "
                "simulated scenarios and are not guarantees."
            ),
        },
    }