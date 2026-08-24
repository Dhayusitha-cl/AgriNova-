"""
Monte Carlo weather scenario generation.

Generates many plausible rainfall scenarios using:
    - calibrated Markov rainfall-state transitions
    - empirical rainfall amounts from observed IMD data

Simulation outputs are scenarios, not forecasts or guarantees.
"""

import numpy as np

from src.weather_simulator import generate_rainfall_scenario

def generate_calibrated_monte_carlo_scenarios(
    start_date,
    num_days,
    num_simulations=1000,
    initial_state="dry",
    random_seed=None,
    rainfall_data=None,
):
    """
    Generate multiple rainfall scenarios using calibrated
    seasonal historical rainfall behaviour.

    Historical calibration is performed once and reused
    across all Monte Carlo simulations.

    The generated outputs are plausible scenarios, not
    deterministic weather forecasts or guarantees.
    """

    import pandas as pd

    from src.markov_calibration import (
        calculate_transition_matrix,
        get_monthly_transition_matrix_with_fallback,
    )
    from src.rainfall_amount_calibration import (
        load_processed_rainfall,
        sample_month_state_rainfall_amounts,
    )

    if num_days <= 0:
        raise ValueError(
            "num_days must be greater than zero."
        )

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    if initial_state not in ["dry", "drizzle", "rain"]:
        raise ValueError(
            f"Unknown initial state: {initial_state}"
        )

    simulation_start = pd.Timestamp(start_date)

    # ---------------------------------------------------------
    # CALIBRATION: perform expensive historical processing ONCE
    # ---------------------------------------------------------

    if rainfall_data is None:
        rainfall_data = load_processed_rainfall()

    fallback_matrix = calculate_transition_matrix(
        rainfall_data
    )

    monthly_matrices = {}

    for month in range(1, 13):
        monthly_matrices[month] = (
            get_monthly_transition_matrix_with_fallback(
                rainfall_data,
                month=month,
                fallback_matrix=fallback_matrix,
            )
        )

    # Pre-group observed rainfall amounts by month and state.
    rainfall_samples = {}

    for month in range(1, 13):
        month_data = rainfall_data[
            rainfall_data["date"].dt.month == month
        ]

        for state in ["dry", "drizzle", "rain"]:

            values = month_data.loc[
                month_data["rainfall_state"] == state,
                "rainfall_mm",
            ].to_numpy(dtype=float)

            if len(values) == 0:
                values = rainfall_data.loc[
                    rainfall_data["rainfall_state"] == state,
                    "rainfall_mm",
                ].to_numpy(dtype=float)

            if len(values) == 0:
                raise ValueError(
                    f"No rainfall observations found for "
                    f"state '{state}'."
                )

            rainfall_samples[(month, state)] = values

    # ---------------------------------------------------------
    # MONTE CARLO
    # ---------------------------------------------------------

    rng = np.random.default_rng(random_seed)

    scenarios = []

    states = ["dry", "drizzle", "rain"]

    for _ in range(num_simulations):

        current_state = initial_state
        simulation_date = simulation_start

        scenario = []

        for day in range(num_days):

            month = simulation_date.month

            matrix = monthly_matrices[month]

            current_index = states.index(
                current_state
            )

            if day == 0:
                simulated_state = current_state
            else:
                next_index = rng.choice(
                    len(states),
                    p=matrix[current_index],
                )

                simulated_state = states[next_index]

            if simulated_state == "dry":

                rainfall = 0.0

            else:

                values = rainfall_samples[
                    (month, simulated_state)
                ]

                rainfall = float(
                    rng.choice(values)
                )

            scenario.append(
                {
                    "day": day + 1,
                    "date": simulation_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "rainfall_state": simulated_state,
                    "rainfall_mm": rainfall,
                }
            )

            current_state = simulated_state
            simulation_date += pd.Timedelta(days=1)

        scenarios.append(scenario)

    return scenarios

def generate_monte_carlo_scenarios(
    transition_matrix,
    num_days,
    num_simulations=1000,
    initial_state="dry",
    rainfall_data=None,
    random_seed=None,
):
    """
    Generate multiple plausible rainfall scenarios.

    Parameters
    ----------
    transition_matrix : array-like
        3x3 calibrated Markov transition matrix.

    num_days : int
        Number of days in each simulated scenario.

    num_simulations : int
        Number of Monte Carlo scenarios.

    initial_state : str
        Initial rainfall state.

    rainfall_data : pandas.DataFrame or None
        Observed processed IMD rainfall data.

    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    list[list[dict]]
        A list containing multiple rainfall scenarios.

    Notes
    -----
    These are simulated scenarios based on historical observations.
    They are not weather forecasts.
    """

    if num_days <= 0:
        raise ValueError("num_days must be greater than zero.")

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    rng = np.random.default_rng(random_seed)

    scenarios = []

    for _ in range(num_simulations):

        scenario_seed = int(
            rng.integers(0, 2**32 - 1)
        )

        scenario = generate_rainfall_scenario(
            transition_matrix=transition_matrix,
            num_days=num_days,
            initial_state=initial_state,
            random_seed=scenario_seed,
            rainfall_data=rainfall_data,
        )

        scenarios.append(scenario)

    return scenarios


def summarize_monte_carlo_scenarios(scenarios):
    """
    Summarize rainfall totals across Monte Carlo scenarios.

    Returns
    -------
    dict
        Distribution statistics for total rainfall.
    """

    if not scenarios:
        raise ValueError("No scenarios supplied.")

    totals = np.array(
        [
            sum(day["rainfall_mm"] for day in scenario)
            for scenario in scenarios
        ],
        dtype=float,
    )

    return {
        "num_simulations": int(len(totals)),
        "min_total_mm": float(np.min(totals)),
        "max_total_mm": float(np.max(totals)),
        "mean_total_mm": float(np.mean(totals)),
        "median_total_mm": float(np.median(totals)),
        "p10_total_mm": float(np.percentile(totals, 10)),
        "p25_total_mm": float(np.percentile(totals, 25)),
        "p75_total_mm": float(np.percentile(totals, 75)),
        "p90_total_mm": float(np.percentile(totals, 90)),
    }