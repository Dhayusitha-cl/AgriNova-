import numpy as np

from .crop_data import crops
from .soil_data import soils
from .soil_water import simulate_soil_water
from .crop_establishment import evaluate_establishment


def simulate_climate_change(
    transition_matrix,
    temperature_increase=2.0,
    rainfall_change_pct=-10,
):
    """
    Adjust a Markov weather transition matrix for a climate scenario.

    This is a scenario transformation, not a climate projection model.
    """

    adjusted_matrix = np.asarray(
        transition_matrix,
        dtype=float,
    ).copy()

    if adjusted_matrix.shape != (3, 3):
        raise ValueError(
            "transition_matrix must have shape (3, 3)."
        )

    # Reduce transitions into rainfall states.
    # State 0 = dry, state 1 = drizzle, state 2 = rain.
    adjusted_matrix[:, 2] *= (
        1 + rainfall_change_pct / 100
    )

    adjusted_matrix[:, 1] *= (
        1 + rainfall_change_pct / 200
    )

    # Prevent negative probabilities.
    adjusted_matrix = np.maximum(
        adjusted_matrix,
        0.0,
    )

    # Re-normalize each row.
    row_sums = adjusted_matrix.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sums <= 0):
        raise ValueError(
            "Climate adjustment produced an invalid transition matrix."
        )

    adjusted_matrix /= row_sums

    return adjusted_matrix


def calculate_et_with_warming(
    tmax,
    tmin,
    temperature_increase=2.0,
):
    """
    Calculate simplified daily evapotranspiration under warming.

    This is a simplified scenario calculation and should not be
    interpreted as a calibrated ET model.
    """

    tmax_adj = tmax + temperature_increase
    tmin_adj = tmin + temperature_increase

    tavg = (
        tmax_adj + tmin_adj
    ) / 2

    et = 0.5 * (tavg - 10)

    return max(
        1,
        min(et, 12),
    )


def _sample_rainfall_series(
    transition_matrix,
    rainfall_yesterday,
    days,
    rng,
):
    """
    Generate one plausible rainfall sequence from the Markov model.

    Weather states:
        0 = dry
        1 = drizzle
        2 = rain

    The rainfall amount associated with each state is based on the
    supplied recent rainfall observation.
    """

    matrix = np.asarray(
        transition_matrix,
        dtype=float,
    )

    rainfall_yesterday = max(
        0.0,
        float(rainfall_yesterday),
    )

    # Simple scenario rainfall mapping.
    # This keeps the climate simulator compatible with the existing
    # 3-state Markov representation without inventing a new model.
    state_rainfall = np.array(
        [
            0.0,
            rainfall_yesterday * 0.5,
            rainfall_yesterday * 1.5,
        ],
        dtype=float,
    )

    # If yesterday was dry, provide small scenario amounts for the
    # wet states rather than making every simulated trajectory zero.
    if rainfall_yesterday == 0:
        state_rainfall = np.array(
            [0.0, 5.0, 15.0],
            dtype=float,
        )

    current_state = 0

    rainfall_series = []

    for _ in range(days):
        probabilities = matrix[current_state]

        next_state = rng.choice(
            3,
            p=probabilities,
        )

        rainfall_series.append(
            state_rainfall[next_state]
        )

        current_state = next_state

    return rainfall_series


def _run_single_scenario(
    crop_name,
    soil_type,
    current_moisture,
    rainfall_yesterday,
    transition_matrix,
    temperature_increase,
    num_simulations,
    days,
    rng,
):
    """
    Run Monte Carlo establishment simulation for one climate scenario.
    """

    if crop_name not in crops:
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

    if soil_type not in soils:
        raise ValueError(
            f"Unknown soil type: {soil_type}"
        )

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    if current_moisture < 0:
        raise ValueError(
            "current_moisture cannot be negative."
        )

    if current_moisture > field_capacity:
        raise ValueError(
            "current_moisture cannot exceed field capacity."
        )

    crop = crops[crop_name]

    germination_days = int(
        crop["germination_days"]
    )

    simulation_days = max(
        days,
        germination_days,
    )

    successful_runs = 0

    for _ in range(num_simulations):

        rainfall_series = _sample_rainfall_series(
            transition_matrix=transition_matrix,
            rainfall_yesterday=rainfall_yesterday,
            days=simulation_days,
            rng=rng,
        )

        # ET increases under warming.
        #
        # We use a simple baseline ET scenario because the current API
        # does not provide daily Tmax/Tmin observations.
        baseline_et = 6.0 + (
            0.5 * temperature_increase
        )

        baseline_et = max(
            1.0,
            min(baseline_et, 12.0),
        )

        et_series = [
            baseline_et
        ] * simulation_days

        soil_water_results = simulate_soil_water(
            rainfall_series=rainfall_series,
            et_series=et_series,
            soil_type=soil_type,
            initial_water_mm=current_moisture,
        )

        establishment = evaluate_establishment(
            soil_water_results=soil_water_results,
            crop=crop_name,
            soil_type=soil_type,
        )

        if establishment["establishment_success"]:
            successful_runs += 1

    probability = (
        successful_runs / num_simulations
    )

    return probability


def run_climate_projection(
    crop_name,
    soil_type,
    current_moisture,
    rainfall_yesterday,
    transition_matrix,
    climate_scenarios=None,
    num_simulations=300,
    days=7,
    random_seed=42,
):
    """
    Estimate crop-establishment probability under climate scenarios.

    Scenarios:
        Current
        2030
        2050

    Results are Monte Carlo scenario estimates, not guarantees or
    validated future climate predictions.
    """

    if climate_scenarios is None:
        climate_scenarios = [
            "Current",
            "2030",
            "2050",
        ]

    rng = np.random.default_rng(
        random_seed
    )

    results = {}

    for scenario in climate_scenarios:

        if scenario == "Current":

            adjusted_matrix = np.asarray(
                transition_matrix,
                dtype=float,
            )

            temp_increase = 0.0

        elif scenario == "2030":

            temp_increase = 1.0

            adjusted_matrix = (
                simulate_climate_change(
                    transition_matrix,
                    temperature_increase=1.0,
                    rainfall_change_pct=-5,
                )
            )

        elif scenario == "2050":

            temp_increase = 2.5

            adjusted_matrix = (
                simulate_climate_change(
                    transition_matrix,
                    temperature_increase=2.5,
                    rainfall_change_pct=-15,
                )
            )

        else:
            continue

        probability = _run_single_scenario(
            crop_name=crop_name,
            soil_type=soil_type,
            current_moisture=current_moisture,
            rainfall_yesterday=rainfall_yesterday,
            transition_matrix=adjusted_matrix,
            temperature_increase=temp_increase,
            num_simulations=num_simulations,
            days=days,
            rng=rng,
        )

        crop = crops[crop_name]
        soil = soils[soil_type]

        minimum_moisture = (
            float(
                soil["field_capacity_mm"]
            )
            * float(
                crop["min_moisture_pct"]
            )
            / 100
        )

        results[scenario] = {
            "germination_probability": probability,
            "temperature_increase": temp_increase,
            "min_moisture": minimum_moisture,
            "transition_matrix": adjusted_matrix,
        }

    return results


def get_climate_impact_summary(results):
    """
    Generate a concise explanation of climate-scenario results.
    """

    if "Current" not in results:
        return "No current baseline data available."

    current_prob = results[
        "Current"
    ]["germination_probability"]

    summary = (
        f"Current establishment probability: "
        f"{current_prob * 100:.0f}%\n\n"
    )

    if "2030" in results:

        prob_2030 = results[
            "2030"
        ]["germination_probability"]

        change = (
            prob_2030 - current_prob
        ) * 100

        summary += (
            f"2030 scenario: "
            f"{prob_2030 * 100:.0f}% "
            f"({change:+.0f} percentage points)\n\n"
        )

    if "2050" in results:

        prob_2050 = results[
            "2050"
        ]["germination_probability"]

        change = (
            prob_2050 - current_prob
        ) * 100

        summary += (
            f"2050 scenario: "
            f"{prob_2050 * 100:.0f}% "
            f"({change:+.0f} percentage points)"
        )

    return summary