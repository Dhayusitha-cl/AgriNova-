import numpy as np
from economic_engine import calculate_expected_return, calculate_wait_return
def make_decision(
    crop_name,
    soil_type,
    current_moisture_mm,
    rainfall_yesterday_mm,
    transition_matrix,
    num_simulations=500,
    days_to_simulate=10
):
    """
    CropLogic-Saathi decision engine.

    Compares:
    1. SOW TODAY
    2. WAIT 5 DAYS
    3. SWITCH TO SOYBEAN

    The decision combines:
    - Monte Carlo weather scenarios
    - soil-water simulation
    - crop establishment probability
    - simplified economic risk

    Probabilities are simulation estimates, not guarantees.
    """

    from crop_data import crops
    from soil_data import soils

    crop = crops[crop_name]
    soil = soils[soil_type]

    # ---------------------------------------------------------
    # CROP REQUIREMENTS
    # ---------------------------------------------------------

    min_moisture = (
        soil["field_capacity_mm"]
        * (crop["min_moisture_pct"] / 100)
    )

    germination_period = crop["germination_days"]

    # ---------------------------------------------------------
    # SCENARIO A — SOW TODAY
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
    # SCENARIO B — WAIT 5 DAYS
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
    # SCENARIO C — SWITCH TO SOYBEAN
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
    # ECONOMIC EXPECTED RETURNS
    # ---------------------------------------------------------

    sow_expected_return = calculate_expected_return(
        crop_name,
        germ_prob_today
    )

    wait_expected_return = calculate_wait_return(
    crop_name,
    germ_prob_wait,
    wait_days=5
)
    soybean_expected_return = calculate_expected_return(
        "soybean",
        germ_prob_soybean
    )

    # ---------------------------------------------------------
    # ECONOMIC OPTIONS
    # ---------------------------------------------------------

    economic_options = {
        "SOW TODAY": sow_expected_return,
        "WAIT 5 DAYS": wait_expected_return,
        "SWITCH TO SOYBEAN": soybean_expected_return
    }

    best_economic_option = max(
        economic_options,
        key=economic_options.get
    )

    # ---------------------------------------------------------
    # DECISION LOGIC
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # We do NOT allow a low establishment probability
    # to produce SOW TODAY simply because the theoretical
    # economic return is high.
    #
    # 70% = minimum threshold for direct SOW recommendation.
    # 50% = minimum threshold for soybean switch.
    # Otherwise WAIT is preferred.
    # ---------------------------------------------------------

    if germ_prob_today >= 0.70:

        decision = "SOW TODAY"
        color = "🟢"
        confidence = germ_prob_today

    elif (
        germ_prob_soybean >= 0.50
        and soybean_expected_return > sow_expected_return
        and soybean_expected_return >= wait_expected_return
    ):

        decision = "SWITCH TO SOYBEAN"
        color = "🔴"
        confidence = germ_prob_soybean

    else:

        decision = "WAIT 5 DAYS"
        color = "🟡"
        confidence = max(
            germ_prob_today,
            germ_prob_wait,
            germ_prob_soybean
        )

    # ---------------------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------------------
        explanation = generate_decision_explanation(
        decision=decision,
        crop_name=crop_name,
        germ_prob_today=germ_prob_today,
        germ_prob_wait=germ_prob_wait,
        germ_prob_soybean=germ_prob_soybean,
        sow_expected_return=sow_expected_return,
        wait_expected_return=wait_expected_return,
        soybean_expected_return=soybean_expected_return
    )
    return {
        "decision": decision,
        "color": color,

        "germ_prob_today": germ_prob_today,
        "germ_prob_wait": germ_prob_wait,
        "germ_prob_soybean": germ_prob_soybean,

        "sow_expected_return": sow_expected_return,
        "wait_expected_return": wait_expected_return,
        "soybean_expected_return": soybean_expected_return,

        "best_economic_option": best_economic_option,

        "confidence": confidence,
        "explanation": explanation,

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
    Monte Carlo soil-water simulation.
    """

    from soil_data import soils
    from weather_simulator import generate_weather_sequence

    soil = soils[soil_type]

    all_trajectories = np.zeros(
        (num_simulations, days + 1)
    )

    all_trajectories[:, 0] = initial_moisture

    for sim in range(num_simulations):

        weather = generate_weather_sequence(
            transition_matrix,
            days
        )

        moisture = initial_moisture

        for day in range(days):

            # Today's observed rainfall
            if day == 0:
                moisture += rainfall_today * 0.8

            # Simulated future rainfall
            else:
                moisture += weather[day - 1]["rainfall"] * 0.8

            # Evapotranspiration
            et = calculate_et_daily(
                weather[day]["tmax"],
                weather[day]["tmin"]
            )

            moisture -= et

            # Field capacity / drainage approximation
            if moisture > soil["field_capacity_mm"]:

                excess = (
                    moisture
                    - soil["field_capacity_mm"]
                )

                moisture = (
                    soil["field_capacity_mm"]
                    + excess * 0.3
                )

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
    Simulate soil moisture when the farmer waits before sowing.
    """

    from soil_data import soils
    from weather_simulator import generate_weather_sequence

    soil = soils[soil_type]

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

            moisture += (
                weather[day]["rainfall"]
                * 0.8
            )

            et = calculate_et_daily(
                weather[day]["tmax"],
                weather[day]["tmin"]
            )

            moisture -= et

            moisture = max(
                0,
                min(
                    moisture,
                    soil["field_capacity_mm"]
                )
            )

            trajectory.append(moisture)

        all_trajectories[sim, :] = trajectory

    return all_trajectories


def calculate_germination_probability(
    trajectories,
    min_moisture,
    germination_period
):
    """
    Estimate probability that soil moisture remains
    above the crop threshold for the required germination period.
    """

    success_count = 0

    total = trajectories.shape[0]

    for sim in range(total):

        moisture_ok = (
            trajectories[sim, :]
            >= min_moisture
        )

        for day in range(
            len(moisture_ok)
            - germination_period
            + 1
        ):

            if all(
                moisture_ok[
                    day:
                    day + germination_period
                ]
            ):

                success_count += 1
                break

    return success_count / total


def calculate_et_daily(tmax, tmin):
    """
    Simplified daily evapotranspiration estimate.
    """

    tavg = (tmax + tmin) / 2

    et = 0.5 * (tavg - 10)

    return max(
        1,
        min(et, 10)
    )
def generate_decision_explanation(
    decision,
    crop_name,
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
    sow_expected_return,
    wait_expected_return,
    soybean_expected_return
):
    """
    Generate a deterministic explanation for the decision.

    The explanation is based only on model outputs.
    It does not introduce new predictions or assumptions.
    """

    explanation = []

    if decision == "SOW TODAY":
        explanation.append(
            f"{crop_name} has an estimated establishment probability "
            f"of {germ_prob_today:.1%} when sown today."
        )

        explanation.append(
            f"The estimated economic return from sowing today is "
            f"{sow_expected_return:.2f}."
        )

        explanation.append(
            "The establishment probability meets the model's "
            "threshold for sowing today."
        )

    elif decision == "WAIT 5 DAYS":
        explanation.append(
            f"{crop_name} has an estimated establishment probability "
            f"of {germ_prob_today:.1%} when sown today."
        )

        explanation.append(
            f"The estimated establishment probability after waiting "
            f"5 days is {germ_prob_wait:.1%}."
        )

        explanation.append(
            "The current conditions do not meet the model's threshold "
            "for recommending immediate sowing."
        )

        if soybean_expected_return > sow_expected_return:
            explanation.append(
                "An alternative crop has a higher estimated economic "
                "return, but the model does not recommend switching "
                "under the current decision rules."
            )

    elif decision == "SWITCH TO SOYBEAN":
        explanation.append(
            f"{crop_name} has an estimated establishment probability "
            f"of {germ_prob_today:.1%} when sown today."
        )

        explanation.append(
            f"Soybean has an estimated establishment probability "
            f"of {germ_prob_soybean:.1%}."
        )

        explanation.append(
            f"The estimated soybean return is {soybean_expected_return:.2f}, "
            f"compared with {sow_expected_return:.2f} for the current crop."
        )

        explanation.append(
            "The model therefore recommends switching to soybean "
            "under the current assumptions."
        )

    return explanation
