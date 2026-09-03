"""
Historical backtesting utilities for CropLogic-Saathi.

The key rule is:

    TRAINING DATA = information available before the decision date
    ACTUAL FUTURE = used only for evaluation

This prevents future-data leakage.
"""


import pandas as pd
import numpy as np

from src.rainfall_preprocessing import (
    classify_rainfall,
    validate_daily_rainfall_observations,
)

def validate_rainfall_dataframe(dataframe):
    """
    Backward-compatible wrapper for the canonical daily rainfall
    observation validator.
    """

    return validate_daily_rainfall_observations(dataframe)



def get_training_data(dataframe, decision_date):
    """
    Return only information available before the decision date.

    The decision date itself is excluded because the model should
    represent information known before the decision.
    """

    dataframe = validate_daily_rainfall_observations(dataframe)

    decision_date = pd.Timestamp(decision_date)

    training_data = dataframe[
        dataframe["date"] < decision_date
    ].copy()

    if training_data.empty:
        raise ValueError(
            "No historical training data exists before "
            f"decision date {decision_date.date()}."
        )

    return training_data.reset_index(drop=True)


def get_actual_future_data(
    dataframe,
    decision_date,
    horizon_days,
):
    """
    Return actual rainfall observations after the decision date.

    These observations are NOT used for model calibration.
    They are reserved for evaluating the simulation.
    """

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    dataframe = validate_daily_rainfall_observations(dataframe)

    decision_date = pd.Timestamp(decision_date)

    future_end = (
        decision_date
        + pd.Timedelta(days=horizon_days)
    )

    future_data = dataframe[
        (dataframe["date"] >= decision_date)
        & (dataframe["date"] < future_end)
    ].copy()

    if future_data.empty:
        raise ValueError(
            "No actual future observations found for "
            f"decision date {decision_date.date()}."
        )

    expected_dates = pd.date_range(
        start=decision_date,
        periods=horizon_days,
        freq="D",
    )

    actual_dates = future_data["date"].reset_index(
        drop=True
    )

    if len(future_data) != horizon_days:
        raise ValueError(
            "Actual future observations do not cover the "
            f"complete {horizon_days}-day horizon starting "
            f"on {decision_date.date()}."
        )

    if not actual_dates.equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError(
            "Actual future observations contain missing or "
            "non-consecutive dates."
        )

    return future_data.reset_index(drop=True)


def create_backtest_split(
    dataframe,
    decision_date,
    horizon_days,
):
    """
    Create a leakage-safe historical backtesting split.

    Returns
    -------
    dict
        training_data:
            observations before decision date

        actual_future:
            observations from decision date onward
            for the requested horizon
    """

    training_data = get_training_data(
        dataframe,
        decision_date,
    )

    actual_future = get_actual_future_data(
        dataframe,
        decision_date,
        horizon_days,
    )

    return {
        "decision_date": pd.Timestamp(decision_date),
        "horizon_days": horizon_days,
        "training_data": training_data,
        "actual_future": actual_future,
    }
from src.markov_calibration import (
    calculate_transition_matrix,
)


def calibrate_backtest_transition_matrix(training_data):
    """
    Calibrate a Markov transition matrix using only the
    historical training data available before the decision date.

    This function must never receive future observations.
    """

    training_data = validate_daily_rainfall_observations(
        training_data
    )

    matrix = calculate_transition_matrix(
        training_data
    )

    return matrix
def calculate_actual_future_total(actual_future):
    """
    Calculate the total rainfall actually observed during
    the held-out future period.
    """

    actual_future = validate_daily_rainfall_observations(
        actual_future
    )

    return float(
        actual_future["rainfall_mm"].sum()
    )


def calculate_monte_carlo_total_distribution(scenarios):
    """
    Calculate total rainfall for each Monte Carlo scenario.

    Returns
    -------
    numpy.ndarray
        One total-rainfall value per simulation.
    """

    import numpy as np

    if not scenarios:
        raise ValueError(
            "No Monte Carlo scenarios supplied."
        )

    totals = np.array(
        [
            sum(
                day["rainfall_mm"]
                for day in scenario
            )
            for scenario in scenarios
        ],
        dtype=float,
    )

    return totals


def evaluate_actual_against_distribution(
    actual_total_mm,
    simulated_totals,
):
    """
    Evaluate where the actual future rainfall falls within
    the Monte Carlo simulated distribution.

    This is a descriptive validation metric, not a guarantee
    or a forecast accuracy claim.
    """

    import numpy as np

    simulated_totals = np.asarray(
        simulated_totals,
        dtype=float,
    )

    if simulated_totals.size == 0:
        raise ValueError(
            "No simulated totals supplied."
        )

    percentile = (
        np.mean(
            simulated_totals
            <= actual_total_mm
        )
        * 100.0
    )

    return {
        "actual_total_mm": float(actual_total_mm),
        "simulated_mean_mm": float(
            np.mean(simulated_totals)
        ),
        "simulated_median_mm": float(
            np.median(simulated_totals)
        ),
        "simulated_p10_mm": float(
            np.percentile(simulated_totals, 10)
        ),
        "simulated_p90_mm": float(
            np.percentile(simulated_totals, 90)
        ),
        "actual_percentile": float(percentile),
    }
def get_initial_state(dataframe, decision_date):
    """
    Get the rainfall state immediately before the decision date.

    The state is recomputed from the observed rainfall amount
    using the current classification rule rather than trusting
    a potentially stale stored rainfall_state value.

    Only observations strictly before the decision date are used.
    """

    dataframe = validate_daily_rainfall_observations(dataframe)

    decision_date = pd.Timestamp(decision_date)

    previous_data = dataframe[
        dataframe["date"] < decision_date
    ]

    if previous_data.empty:
        raise ValueError(
            "No observation exists before the decision date."
        )

    previous_rainfall = float(
        previous_data.iloc[-1]["rainfall_mm"]
    )

    initial_state = classify_rainfall(
        previous_rainfall
    )

    return initial_state
def run_single_backtest(
    dataframe,
    decision_date,
    horizon_days=14,
    num_simulations=1000,
    random_seed=42,
):
    """
    Run one leakage-safe historical backtest.

    Training data contains only observations before the
    decision date. Future observations are used only for
    evaluation.
    """

    from src.monte_carlo_weather import (
        generate_monte_carlo_scenarios,
    )

    backtest = create_backtest_split(
        dataframe=dataframe,
        decision_date=decision_date,
        horizon_days=horizon_days,
    )

    training_data = backtest["training_data"]
    actual_future = backtest["actual_future"]

    transition_matrix = (
        calibrate_backtest_transition_matrix(
            training_data
        )
    )

    initial_state = get_initial_state(
        dataframe,
        decision_date,
    )

    scenarios = generate_monte_carlo_scenarios(
        transition_matrix=transition_matrix,
        num_days=horizon_days,
        num_simulations=num_simulations,
        initial_state=initial_state,
        rainfall_data=training_data,
        random_seed=random_seed,
    )

    actual_total = calculate_actual_future_total(
        actual_future
    )

    simulated_totals = (
        calculate_monte_carlo_total_distribution(
            scenarios
        )
    )

    evaluation = evaluate_actual_against_distribution(
        actual_total_mm=actual_total,
        simulated_totals=simulated_totals,
    )

    return {
        "decision_date": pd.Timestamp(
            decision_date
        ),
        "initial_state": initial_state,
        "training_rows": len(training_data),
        "actual_future_rows": len(actual_future),
        **evaluation,
    }


def run_multi_date_backtest(
    dataframe,
    decision_dates,
    horizon_days=14,
    num_simulations=1000,
    random_seed=42,
):
    """
    Run leakage-safe backtesting across multiple
    historical decision dates.

    Returns
    -------
    pandas.DataFrame
        One row per historical decision date.
    """

    results = []

    for i, decision_date in enumerate(
        decision_dates
    ):

        result = run_single_backtest(
            dataframe=dataframe,
            decision_date=decision_date,
            horizon_days=horizon_days,
            num_simulations=num_simulations,
            random_seed=random_seed + i,
        )

        results.append(result)

    return pd.DataFrame(results)
def calibrate_backtest_monthly_transition_matrix(
    training_data,
    decision_date,
):
    """
    Calibrate a month-specific Markov transition matrix
    using only training observations available before the
    historical decision date.

    This prevents future-data leakage.

    Parameters
    ----------
    training_data : pandas.DataFrame
        Historical observations strictly before the decision date.

    decision_date : str or pandas.Timestamp
        Historical decision date.

    Returns
    -------
    numpy.ndarray
        3x3 transition matrix for the decision month.

    Notes
    -----
    If insufficient observations exist for the decision month,
    the function falls back to the full training-data matrix.
    """

    import numpy as np

    training_data = validate_daily_rainfall_observations(
        training_data
    )

    decision_date = pd.Timestamp(decision_date)

    month = decision_date.month

    monthly_training_data = training_data[
        training_data["date"].dt.month == month
    ].copy()

    # Use the normal training-data matrix if the
    # month-specific dataset is too small.
    if len(monthly_training_data) < 30:
        return calibrate_backtest_transition_matrix(
            training_data
        )

    matrix = calculate_transition_matrix(
        monthly_training_data
    )

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    return matrix


def prepare_month_aware_calibration(
    training_data,
):
    """
    Precompute month-specific transition matrices and rainfall
    samples needed by the month-aware Monte Carlo simulator.

    All calibration uses training_data only.

    Important:
    - Only consecutive observations within the same calendar month
      and same year are used for month-specific transitions.
    - A transition from the last day of one year/month to the first
      day of another month/year is never created.
    - If a month does not have enough transition observations,
      the full training-data matrix is used as fallback.
    """

    import numpy as np

    training_data = validate_daily_rainfall_observations(
        training_data
    )

    valid_states = [
        "dry",
        "drizzle",
        "rain",
    ]

    calibration = {}

    from src.markov_calibration import (
        transition_counts_dataframe,
    )

    # ---------------------------------------------------------
    # Build a leakage-safe full-training fallback matrix.
    #
    # Some small/synthetic datasets may not contain every state.
    # Therefore calculate the matrix manually instead of calling
    # calculate_transition_matrix(), which requires every state
    # to have an observed outgoing transition.
    # ---------------------------------------------------------

    full_counts = transition_counts_dataframe(
        training_data
    ).to_numpy(dtype=float)

    full_transition_matrix = np.zeros_like(
        full_counts,
        dtype=float,
    )

    for row_index in range(len(valid_states)):

        row_total = full_counts[row_index].sum()

        if row_total > 0:

            full_transition_matrix[row_index] = (
                full_counts[row_index]
                / row_total
            )

        else:
            # Completely unobserved state:
            # keep the process in that state rather than inventing
            # unsupported transitions.
            full_transition_matrix[
                row_index,
                row_index,
            ] = 1.0

    # ---------------------------------------------------------
    # Build calibration for every month.
    # ---------------------------------------------------------

    for month in range(1, 13):

        monthly_training = training_data[
            training_data["date"].dt.month == month
        ].copy()

        monthly_transition_data = []

        # Group by year first so that we never create a transition
        # between July 31 of one year and July 1 of another year.
        for year, year_data in monthly_training.groupby(
            monthly_training["date"].dt.year
        ):

            year_data = (
                year_data
                .sort_values("date")
                .reset_index(drop=True)
            )

            if len(year_data) < 2:
                continue

            date_difference = (
                year_data["date"].diff().dt.days
            )

            # Keep only genuinely consecutive observations.
            consecutive_data = year_data[
                date_difference.eq(1)
                | date_difference.isna()
            ].copy()

            if len(consecutive_data) >= 2:
                monthly_transition_data.append(
                    consecutive_data
                )

        if monthly_transition_data:

            monthly_training_for_transitions = pd.concat(
                monthly_transition_data,
                ignore_index=True,
            )

        else:

            monthly_training_for_transitions = (
                pd.DataFrame()
            )

        # Full training matrix is always the safe fallback.
        transition_matrix = (
            full_transition_matrix.copy()
        )

        # Use month-specific transitions only when there are
        # sufficient consecutive observations.
        if len(monthly_training_for_transitions) >= 30:

            monthly_counts = (
                transition_counts_dataframe(
                    monthly_training_for_transitions
                ).to_numpy(dtype=float)
            )

            monthly_matrix = np.zeros_like(
                monthly_counts,
                dtype=float,
            )

            for row_index in range(len(valid_states)):

                row_total = monthly_counts[
                    row_index
                ].sum()

                if row_total > 0:

                    monthly_matrix[row_index] = (
                        monthly_counts[row_index]
                        / row_total
                    )

                else:

                    # If this state has no outgoing transition
                    # in the month-specific data, retain the
                    # corresponding full-training fallback row.
                    monthly_matrix[row_index] = (
                        full_transition_matrix[
                            row_index
                        ]
                    )

            transition_matrix = monthly_matrix

        # -----------------------------------------------------
        # Rainfall amount distributions by state.
        # -----------------------------------------------------

        rainfall_values = {}

        for state in valid_states:

            if state == "dry":

                rainfall_values[state] = np.array(
                    [0.0],
                    dtype=float,
                )

                continue

            # Prefer rainfall amounts from the current month.
            monthly_state_values = (
                monthly_training.loc[
                    monthly_training[
                        "rainfall_state"
                    ] == state,
                    "rainfall_mm",
                ]
                .to_numpy(dtype=float)
            )

            # Fall back to all training observations for that state.
            if len(monthly_state_values) == 0:

                monthly_state_values = (
                    training_data.loc[
                        training_data[
                            "rainfall_state"
                        ] == state,
                        "rainfall_mm",
                    ]
                    .to_numpy(dtype=float)
                )

            # If the state has never been observed in the training
            # data, use zero rainfall rather than inventing a
            # rainfall distribution.
            #
            # Normally this state should also have zero probability
            # from observed-state transition rows.
            if len(monthly_state_values) == 0:

                monthly_state_values = np.array(
                    [0.0],
                    dtype=float,
                )

            rainfall_values[state] = (
                monthly_state_values
            )

        calibration[month] = {
            "transition_matrix": transition_matrix,
            "rainfall_values": rainfall_values,
        }

    return calibration

def generate_month_aware_backtest_scenario(
    training_data,
    start_date,
    horizon_days,
    initial_state,
    random_seed=None,
    calibration=None,
):
    """
    Generate one leakage-safe rainfall scenario for backtesting.

    Calibration is derived only from training_data.

    The transition matrix and rainfall distribution are selected
    according to the simulated calendar month.
    """

    import numpy as np

    training_data = validate_daily_rainfall_observations(
        training_data
    )

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    valid_states = [
        "dry",
        "drizzle",
        "rain",
    ]

    if initial_state not in valid_states:
        raise ValueError(
            f"Invalid initial state: {initial_state}"
        )

    start_date = pd.Timestamp(start_date)

    if calibration is None:
        calibration = prepare_month_aware_calibration(
            training_data
        )

    rng = np.random.default_rng(
        random_seed
    )

    current_state = initial_state

    scenario = []

    for day_offset in range(horizon_days):

        simulation_date = (
            start_date
            + pd.Timedelta(days=day_offset)
        )

        month = simulation_date.month

        month_calibration = calibration[
            month
        ]

        transition_matrix = (
            month_calibration[
                "transition_matrix"
            ]
        )

        current_state_index = valid_states.index(
            current_state
        )

        next_state_index = rng.choice(
            len(valid_states),
            p=transition_matrix[
                current_state_index
            ],
        )

        next_state = valid_states[
            next_state_index
        ]

        if next_state == "dry":

            rainfall = 0.0

        else:

            rainfall_values = (
                month_calibration[
                    "rainfall_values"
                ][next_state]
            )

            rainfall = float(
                rng.choice(
                    rainfall_values
                )
            )

        scenario.append(
            {
                "day": day_offset + 1,
                "date": simulation_date,
                "rainfall_state": next_state,
                "rainfall_mm": rainfall,
            }
        )

        current_state = next_state

    return scenario

def generate_month_aware_monte_carlo_scenarios(
    training_data,
    start_date,
    horizon_days,
    initial_state,
    num_simulations=1000,
    random_seed=42,
):
    """
    Generate multiple month-aware Monte Carlo rainfall scenarios.

    Calibration is performed once and reused across all simulations.
    """

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    calibration = (
        prepare_month_aware_calibration(
            training_data
        )
    )

    scenarios = []

    for i in range(num_simulations):

        scenario = (
            generate_month_aware_backtest_scenario(
                training_data=training_data,
                start_date=start_date,
                horizon_days=horizon_days,
                initial_state=initial_state,
                random_seed=random_seed + i,
                calibration=calibration,
            )
        )

        scenarios.append(scenario)

    return scenarios
def run_single_month_aware_backtest(
    dataframe,
    decision_date,
    horizon_days=14,
    num_simulations=1000,
    random_seed=42,
):
    """
    Run one leakage-safe backtest using month-aware
    Monte Carlo rainfall scenarios.
    """

    backtest = create_backtest_split(
        dataframe=dataframe,
        decision_date=decision_date,
        horizon_days=horizon_days,
    )

    training_data = backtest["training_data"]
    actual_future = backtest["actual_future"]

    initial_state = get_initial_state(
        dataframe,
        decision_date,
    )

    scenarios = generate_month_aware_monte_carlo_scenarios(
        training_data=training_data,
        start_date=decision_date,
        horizon_days=horizon_days,
        initial_state=initial_state,
        num_simulations=num_simulations,
        random_seed=random_seed,
    )

    actual_total = calculate_actual_future_total(
        actual_future
    )

    simulated_totals = (
        calculate_monte_carlo_total_distribution(
            scenarios
        )
    )

    evaluation = evaluate_actual_against_distribution(
        actual_total_mm=actual_total,
        simulated_totals=simulated_totals,
    )

    return {
        "decision_date": pd.Timestamp(decision_date),
        "initial_state": initial_state,
        "training_rows": len(training_data),
        "actual_future_rows": len(actual_future),
        **evaluation,
    }
def run_multi_date_month_aware_backtest(
    dataframe,
    decision_dates,
    horizon_days=14,
    num_simulations=1000,
    random_seed=42,
):
    """
    Run leakage-safe month-aware backtesting across
    multiple historical decision dates.

    Returns
    -------
    pandas.DataFrame
        One row per historical decision date.
    """

    results = []

    for i, decision_date in enumerate(
        decision_dates
    ):

        result = run_single_month_aware_backtest(
            dataframe=dataframe,
            decision_date=decision_date,
            horizon_days=horizon_days,
            num_simulations=num_simulations,
            random_seed=random_seed + i,
        )

        results.append(result)

    return pd.DataFrame(results)

def evaluate_rainfall_simulation(
    training_data,
    actual_future,
    start_date,
    initial_state,
    horizon_days=14,
    num_simulations=1000,
    random_seed=42,
):
    """
    Evaluate Monte Carlo rainfall scenarios against held-out rainfall.

    Training data is used only to generate simulated scenarios.
    Actual future observations are used only for evaluation.

    Returns
    -------
    dict
        Simulation summary and comparison metrics.

    Notes
    -----
    This evaluates rainfall-simulation behaviour only.
    It is not a crop-establishment or decision-quality metric.
    """

    if training_data.empty:
        raise ValueError(
            "training_data cannot be empty."
        )

    if actual_future.empty:
        raise ValueError(
            "actual_future cannot be empty."
        )

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    if num_simulations <= 0:
        raise ValueError(
            "num_simulations must be greater than zero."
        )

    scenarios = generate_month_aware_monte_carlo_scenarios(
        training_data=training_data,
        start_date=pd.Timestamp(start_date),
        horizon_days=horizon_days,
        initial_state=initial_state,
        num_simulations=num_simulations,
        random_seed=random_seed,
    )

    if not scenarios:
        raise ValueError(
            "No Monte Carlo scenarios were generated."
        )

    simulated_totals = np.array(
        [
            sum(
                float(day.get("rainfall_mm", 0.0))
                for day in scenario
            )
            for scenario in scenarios
        ],
        dtype=float,
    )

    actual_rainfall = actual_future[
        "rainfall_mm"
    ].astype(float)

    actual_total = float(
        actual_rainfall.sum()
    )

    p10 = float(
        np.percentile(
            simulated_totals,
            10,
        )
    )

    p50 = float(
        np.percentile(
            simulated_totals,
            50,
        )
    )

    p90 = float(
        np.percentile(
            simulated_totals,
            90,
        )
    )

    simulated_mean = float(
        simulated_totals.mean()
    )

    absolute_error = abs(
        simulated_mean - actual_total
    )

    covered_by_p10_p90 = (
        p10 <= actual_total <= p90
    )

    return {
        "actual_total_mm": actual_total,
        "simulated_mean_mm": simulated_mean,
        "p10_mm": p10,
        "p50_mm": p50,
        "p90_mm": p90,
        "absolute_error_mm": float(
            absolute_error
        ),
        "covered_by_p10_p90": bool(
            covered_by_p10_p90
        ),
        "num_simulations": int(
            len(simulated_totals)
        ),
    }