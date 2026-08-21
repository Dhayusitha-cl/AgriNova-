"""
Weather simulation utilities for AgriNova.

This module provides:
    1. Markov-chain rainfall-state simulation
    2. Empirical rainfall-amount sampling

The simulated rainfall values are scenarios, not forecasts
or guarantees.

Historical rainfall data is used to represent observed
rainfall amounts, while the Markov transition matrix
represents rainfall-state persistence.
"""

import numpy as np

from src.rainfall_amount_calibration import (
    load_processed_rainfall,
    sample_rainfall_amounts,
    sample_month_state_rainfall_amounts,
)


RAINFALL_STATES = ["dry", "drizzle", "rain"]


def create_transition_matrix(
    week_number,
    district="yavatmal",
    use_calibrated=False,
):
    """
    Return the rainfall-state transition matrix.

    Parameters
    ----------
    week_number : int
        Forecast week number.

    district : str
        District identifier.

    use_calibrated : bool
        If True, use the transition matrix calibrated from
        historical processed rainfall data.

        If False, use the existing project transition matrices.

    Returns
    -------
    numpy.ndarray
        3x3 transition matrix.

    Notes
    -----
    The existing week-based matrices are retained as fallback
    assumptions.

    When use_calibrated=True, the matrix is estimated from
    historical rainfall observations using markov_calibration.py.
    """

    if use_calibrated:

        if district.lower() != "yavatmal":
            raise ValueError(
                "Historical calibration is currently available "
                "only for Yavatmal."
            )

        from src.markov_calibration import (
            calculate_multi_year_transition_matrix
        )

        return calculate_multi_year_transition_matrix(
            processed_dir="data/processed"
        )

    matrices = {
        1: np.array([
            [0.75, 0.18, 0.07],
            [0.55, 0.30, 0.15],
            [0.40, 0.35, 0.25],
        ]),

        2: np.array([
            [0.65, 0.25, 0.10],
            [0.45, 0.35, 0.20],
            [0.35, 0.35, 0.30],
        ]),

        3: np.array([
            [0.55, 0.30, 0.15],
            [0.40, 0.35, 0.25],
            [0.30, 0.35, 0.35],
        ]),

        4: np.array([
            [0.50, 0.32, 0.18],
            [0.35, 0.38, 0.27],
            [0.28, 0.35, 0.37],
        ]),
    }

    if week_number in matrices:
        return matrices[week_number].copy()

    return np.array([
        [0.45, 0.35, 0.20],
        [0.30, 0.40, 0.30],
        [0.25, 0.35, 0.40],
    ])

def validate_transition_matrix(transition_matrix):
    """
    Validate a rainfall-state transition matrix.

    Parameters
    ----------
    transition_matrix : array-like
        Expected shape is (3, 3).

    Returns
    -------
    numpy.ndarray
        Validated floating-point matrix.
    """

    matrix = np.asarray(
        transition_matrix,
        dtype=float,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "Transition matrix must have shape (3, 3)."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Transition matrix must contain finite values."
        )

    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(
            "Transition probabilities must be between 0 and 1."
        )

    row_sums = matrix.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Each transition-matrix row must sum to 1."
        )

    return matrix / row_sums[:, np.newaxis]


def generate_markov_states(
    transition_matrix,
    num_days,
    initial_state="dry",
    random_seed=None,
):
    """
    Generate future rainfall states using a Markov chain.

    Parameters
    ----------
    transition_matrix : array-like
        3x3 rainfall-state transition matrix.

    num_days : int
        Number of days to simulate.

    initial_state : str
        Initial rainfall state.

    random_seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    list[str]
        Simulated rainfall states.
    """

    if num_days <= 0:
        raise ValueError(
            "num_days must be greater than zero."
        )

    if initial_state not in RAINFALL_STATES:
        raise ValueError(
            f"Unknown initial state: {initial_state}"
        )

    matrix = validate_transition_matrix(
        transition_matrix
    )

    rng = np.random.default_rng(random_seed)

    current_state = RAINFALL_STATES.index(
        initial_state
    )

    sequence = [initial_state]

    for _ in range(num_days - 1):
        current_state = rng.choice(
            len(RAINFALL_STATES),
            p=matrix[current_state],
        )

        sequence.append(
            RAINFALL_STATES[current_state]
        )

    return sequence


def rainfall_amount_from_state(
    state,
    random_seed=None,
):
    """
    Generate a rainfall amount from a rainfall state.

    This function is retained as a simple utility.

    IMPORTANT
    ---------
    These ranges are simulation assumptions.
    They are not presented as observed IMD rainfall
    distributions.

    For calibrated scenarios, use
    ``generate_rainfall_scenario`` instead.
    """

    if state not in RAINFALL_STATES:
        raise ValueError(
            f"Unknown rainfall state: {state}"
        )

    rng = np.random.default_rng(random_seed)

    if state == "dry":
        return 0.0

    if state == "drizzle":
        return float(
            rng.uniform(0.1, 10.0)
        )

    return float(
        rng.uniform(10.0, 40.0)
    )


def generate_rainfall_scenario(
    transition_matrix,
    num_days,
    initial_state="dry",
    random_seed=None,
    rainfall_data=None,
):
    """
    Generate one future rainfall scenario.

    The process is:

        Markov chain
            ↓
        rainfall states
            ↓
        empirical rainfall amounts

    Parameters
    ----------
    transition_matrix : array-like
        3x3 rainfall-state transition matrix.

    num_days : int
        Number of days to simulate.

    initial_state : str
        Starting rainfall state.

    random_seed : int or None
        Random seed for reproducibility.

    rainfall_data : pandas.DataFrame or None
        Processed historical rainfall data.

        If None, the project rainfall dataset is loaded
        automatically.

    Returns
    -------
    list[dict]
        Daily rainfall scenario.

    Notes
    -----
    The output represents a plausible simulated scenario.
    It is not a deterministic weather forecast.
    """

    if num_days <= 0:
        raise ValueError(
            "num_days must be greater than zero."
        )

    states = generate_markov_states(
        transition_matrix=transition_matrix,
        num_days=num_days,
        initial_state=initial_state,
        random_seed=random_seed,
    )

    if rainfall_data is None:
        rainfall_data = load_processed_rainfall()

    rng = np.random.default_rng(
        random_seed
    )

    scenario = []

    for day, state in enumerate(
        states,
        start=1,
    ):

        if state == "dry":
            rainfall = 0.0

        else:
            seed = int(
                rng.integers(
                    0,
                    2**32 - 1,
                )
            )

            rainfall = float(
                sample_rainfall_amounts(
                    rainfall_data,
                    state,
                    size=1,
                    random_seed=seed,
                )[0]
            )

        scenario.append({
            "day": day,
            "rainfall_state": state,
            "rainfall_mm": rainfall,
        })

    return scenario
def generate_calibrated_rainfall_scenario(
    start_date,
    num_days,
    initial_state="dry",
    district="yavatmal",
    rainfall_data=None,
    random_seed=None,
):
    """
    Generate a rainfall scenario using historical seasonal calibration.

    The transition matrix is selected internally from historical
    rainfall behaviour. The caller does not provide a transition matrix.

    Each simulated transition uses the calendar month of the
    current simulation date. Sparse month/state combinations use
    the state-level historical fallback.

    Rainfall amounts are sampled empirically using month + rainfall
    state.

    Parameters
    ----------
    start_date : str or pandas.Timestamp
        Date on which the simulation starts.

    num_days : int
        Number of days to simulate.

    initial_state : str
        Current rainfall state at the start of simulation.

    district : str
        District identifier.

    rainfall_data : pandas.DataFrame or None
        Historical processed rainfall data.

    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    list[dict]
        Simulated daily rainfall scenario.
    """

    import pandas as pd

    if num_days <= 0:
        raise ValueError(
            "num_days must be greater than zero."
        )

    if district.lower() != "yavatmal":
        raise ValueError(
            "Historical calibration is currently available "
            "only for Yavatmal."
        )

    if initial_state not in RAINFALL_STATES:
        raise ValueError(
            f"Unknown initial state: {initial_state}"
        )

    simulation_date = pd.Timestamp(start_date)

    if rainfall_data is None:
        rainfall_data = load_processed_rainfall()

    from src.markov_calibration import (
        calculate_transition_matrix,
        get_monthly_transition_matrix_with_fallback,
    )

    fallback_matrix = calculate_transition_matrix(
        rainfall_data
    )

    rng = np.random.default_rng(random_seed)

    current_state = initial_state
    scenario = []

    for day in range(num_days):

        current_month = simulation_date.month

        transition_matrix = (
            get_monthly_transition_matrix_with_fallback(
                rainfall_data,
                month=current_month,
                fallback_matrix=fallback_matrix,
            )
        )

        current_index = RAINFALL_STATES.index(
            current_state
        )

        if day == 0:
            simulated_state = current_state
        else:
            next_index = rng.choice(
                len(RAINFALL_STATES),
                p=transition_matrix[current_index],
            )

            simulated_state = RAINFALL_STATES[
                next_index
            ]

        if simulated_state == "dry":
            rainfall = 0.0
        else:
            seed = int(
                rng.integers(
                    0,
                    2**32 - 1,
                )
            )

            rainfall = float(
                sample_month_state_rainfall_amounts(
                    rainfall_data,
                    month=current_month,
                    state=simulated_state,
                    size=1,
                    random_seed=seed,
                )[0]
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

    return scenario