"""
Monte Carlo weather scenario generation.

Generates many plausible rainfall scenarios using:
    - calibrated Markov rainfall-state transitions
    - empirical rainfall amounts from observed IMD data

Simulation outputs are scenarios, not forecasts or guarantees.
"""

import numpy as np

from src.weather_simulator import generate_rainfall_scenario


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