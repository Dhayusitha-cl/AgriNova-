import numpy as np

from .crop_data import crops
from .soil_data import soils
from .weather_simulator import (
    generate_weather_sequence,
    calculate_et_daily
)


def make_decision(
    crop_name,
    soil_type,
    current_moisture_mm,
    rainfall_yesterday_mm,
    transition_matrix,
    num_simulations=500,
    days_to_simulate=7
):
    """
    Compare three pre-sowing scenarios:

    1. SOW TODAY
    2. WAIT 5 DAYS
    3. SWITCH TO SOYBEAN

    Returns germination probabilities and the selected decision.
    """

    crop = crops[crop_name]
    soil = soils[soil_type]

    min_moisture = (
        soil["field_capacity_mm"]
        * (crop["min_moisture_pct"] / 100)
    )

    germination_period = crop["germination_days"]

    # ---------------------------------------------------------
    # SCENARIO A: SOW TODAY
    # ---------------------------------------------------------

    trajectories = simulate_soil_moisture(
        initial_moisture=current_moisture_mm,
        rainfall_today=rainfall_yesterday_mm,
        transition_matrix=transition_matrix,
        soil_type=soil_type,
        num_simulations=num_simulations,
        days=days_to_simulate
    )

    germ_prob_today = calculate_germination_probability(
        trajectories,
        min_moisture,
        germination_period
    )

    # ---------------------------------------------------------
    # SCENARIO B: WAIT 5 DAYS
    # ---------------------------------------------------------

    wait_simulations = simulate_with_wait(
        current_moisture=current_moisture_mm,
        transition_matrix=transition_matrix,
        soil_type=soil_type,
        wait_days=5,
        num_simulations=num_simulations
    )

    germ_prob_wait = calculate_germination_probability(
        wait_simulations,
        min_moisture,
        germination_period
    )

    # ---------------------------------------------------------
    # SCENARIO C: SWITCH TO SOYBEAN
    # ---------------------------------------------------------

    soybean = crops["soybean"]

    soybean_min_moisture = (
        soil["field_capacity_mm"]
        * (soybean["min_moisture_pct"] / 100)
    )

    germ_prob_soybean = calculate_germination_probability(
        trajectories,
        soybean_min_moisture,
        soybean["germination_days"]
    )

    # ---------------------------------------------------------
    # DECISION LOGIC
    # ---------------------------------------------------------

    if germ_prob_today >= 0.7:
        decision = "SOW TODAY"
        confidence = germ_prob_today

    elif germ_prob_wait > germ_prob_today + 0.2:
        decision = "WAIT 5 DAYS"
        confidence = germ_prob_wait

    elif germ_prob_soybean > germ_prob_today + 0.15:
        decision = "SWITCH TO SOYBEAN"
        confidence = germ_prob_soybean

    else:
        decision = "WAIT 5 DAYS"
        confidence = max(
            germ_prob_wait,
            germ_prob_today
        )

    return {
        "decision": decision,
        "germ_prob_today": germ_prob_today,
        "germ_prob_wait": germ_prob_wait,
        "germ_prob_soybean": germ_prob_soybean,
        "confidence": confidence,
        "trajectories": trajectories,
        "wait_simulations": wait_simulations,
        "current_moisture": current_moisture_mm,
        "min_moisture_required": min_moisture
    }


def simulate_soil_moisture(
    initial_moisture,
    rainfall_today,
    transition_matrix,
    soil_type,
    num_simulations=500,
    days=7
):
    """
    Run Monte Carlo simulations of soil moisture
    for the SOW TODAY scenario.
    """

    soil = soils[soil_type]

    all_trajectories = np.zeros(
        (num_simulations, days + 1)
    )

    # Initial soil moisture
    all_trajectories[:, 0] = initial_moisture

    for sim in range(num_simulations):

        weather = generate_weather_sequence(
            transition_matrix,
            days
        )

        moisture = initial_moisture

        for day in range(days):

            # Rainfall on the first simulation day
            if day == 0:
                moisture += rainfall_today * 0.8

            # Simulated rainfall on following days
            else:
                moisture += (
                    weather[day - 1]["rainfall"] * 0.8
                )

            # Evapotranspiration
            et = calculate_et_daily(
                weather[day]["tmax"],
                weather[day]["tmin"]
            )

            moisture -= et

            # Handle excess water above field capacity
            if moisture > soil["field_capacity_mm"]:
                excess = (
                    moisture
                    - soil["field_capacity_mm"]
                )

                moisture = (
                    soil["field_capacity_mm"]
                    + excess * 0.3
                )

            # Soil moisture cannot be negative
            moisture = max(0, moisture)

            all_trajectories[
                sim,
                day + 1
            ] = moisture

    return all_trajectories


def simulate_with_wait(
    current_moisture,
    transition_matrix,
    soil_type,
    wait_days=5,
    num_simulations=500
):
    """
    Simulate soil moisture when the farmer waits
    before sowing.
    """

    soil = soils[soil_type]

    # Wait period + 7 days after waiting
    total_days = wait_days + 7

    all_trajectories = np.zeros(
        (num_simulations, total_days)
    )

    for sim in range(num_simulations):

        weather = generate_weather_sequence(
            transition_matrix,
            total_days
        )

        moisture = current_moisture
        trajectory = []

        for day in range(total_days):

            # Add rainfall
            moisture += (
                weather[day]["rainfall"] * 0.8
            )

            # Remove evapotranspiration
            et = calculate_et_daily(
                weather[day]["tmax"],
                weather[day]["tmin"]
            )

            moisture -= et

            # Keep moisture within physical bounds
            moisture = max(
                0,
                min(
                    moisture,
                    soil["field_capacity_mm"]
                )
            )

            trajectory.append(moisture)

        all_trajectories[
            sim,
            :
        ] = trajectory

    return all_trajectories


def calculate_germination_probability(
    trajectories,
    min_moisture,
    germination_period
):
    """
    Calculate the fraction of simulations in which
    soil moisture remains above the required threshold
    for the required germination period.

    This is a simulation estimate, not a guarantee.
    """

    success_count = 0

    total = trajectories.shape[0]

    if total == 0:
        return 0.0

    for sim in range(total):

        moisture_ok = (
            trajectories[sim, :] >= min_moisture
        )

        for day in range(
            len(moisture_ok) - germination_period + 1
        ):

            if all(
                moisture_ok[
                    day:day + germination_period
                ]
            ):
                success_count += 1
                break

    return success_count / total