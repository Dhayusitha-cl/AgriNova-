"""
Markov-chain calibration from observed rainfall states.

This module estimates weather-state transition probabilities
from historical rainfall observations.

States:
    dry
    drizzle
    rain

The resulting matrix uses:

    rows    = current state
    columns = next state

Matrix order:
    [dry, drizzle, rain]
"""

from collections import defaultdict

import numpy as np
import pandas as pd


STATES = ["dry", "drizzle", "rain"]


def count_transitions(dataframe):
    """
    Count transitions between consecutive rainfall states.

    Missing states are ignored.

    Returns
    -------
    dict
        Nested transition counts.
    """

    counts = defaultdict(
        lambda: {state: 0 for state in STATES}
    )

    states = dataframe["rainfall_state"].tolist()

    for current_state, next_state in zip(states[:-1], states[1:]):
        if current_state not in STATES:
            continue

        if next_state not in STATES:
            continue

        counts[current_state][next_state] += 1

    return dict(counts)


def calculate_transition_matrix(dataframe):
    """
    Estimate a row-normalized Markov transition matrix.

    Returns
    -------
    numpy.ndarray
        3x3 transition matrix.

    Raises
    ------
    ValueError
        If a state has no observed outgoing transitions.
    """

    counts = count_transitions(dataframe)

    matrix = np.zeros(
        (len(STATES), len(STATES)),
        dtype=float,
    )

    for i, current_state in enumerate(STATES):
        total = sum(counts.get(current_state, {}).values())

        if total == 0:
            raise ValueError(
                f"No outgoing transitions observed for state "
                f"'{current_state}'."
            )

        for j, next_state in enumerate(STATES):
            matrix[i, j] = (
                counts[current_state][next_state] / total
            )

    return matrix


def transition_counts_dataframe(dataframe):
    """
    Return transition counts as a readable DataFrame.
    """

    counts = count_transitions(dataframe)

    return pd.DataFrame(
        counts
    ).T.reindex(
        index=STATES,
        columns=STATES,
        fill_value=0,
    )


def transition_matrix_dataframe(dataframe):
    """
    Return the transition matrix as a labelled DataFrame.
    """

    matrix = calculate_transition_matrix(dataframe)

    return pd.DataFrame(
        matrix,
        index=STATES,
        columns=STATES,
    )


def validate_transition_matrix(matrix):
    """
    Validate that a transition matrix is a proper stochastic matrix.

    Every probability must be between 0 and 1 and
    every row must sum approximately to 1.
    """

    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        raise ValueError(
            "Transition matrix must have shape (3, 3)."
        )

    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(
            "Transition probabilities must be between 0 and 1."
        )

    if not np.allclose(
        matrix.sum(axis=1),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Each transition-matrix row must sum to 1."
        )

    return True
def combine_historical_rainfall_data(processed_dir="data/processed"):
    """
    Combine multiple yearly processed rainfall CSV files.

    Expected filenames:
        rainfall_yavatmal_YYYY.csv

    Returns
    -------
    pandas.DataFrame
        Combined historical rainfall observations sorted by date.

    Notes
    -----
    Each year's data is kept as a separate chronological block.
    Transitions are calculated within each year only so that the
    final day of one year is not incorrectly connected to the first
    day of the next year.
    """

    from pathlib import Path

    processed_path = Path(processed_dir)

    files = sorted(
        processed_path.glob("rainfall_yavatmal_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No processed rainfall CSV files found in {processed_path}"
        )

    dataframes = []

    for file_path in files:
        dataframe = pd.read_csv(file_path)

        required_columns = {
            "date",
            "rainfall_mm",
            "rainfall_state",
        }

        missing = required_columns - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: {sorted(missing)}"
            )

        dataframe["date"] = pd.to_datetime(
            dataframe["date"]
        )

        dataframe["_source_file"] = file_path.name

        dataframes.append(dataframe)

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["date", "_source_file"]
    ).reset_index(drop=True)

    return combined


def calculate_multi_year_transition_matrix(
    processed_dir="data/processed"
):
    """
    Calculate a Markov transition matrix using multiple
    historical rainfall years.

    Transitions are counted separately within each yearly
    dataset, preventing artificial transitions across year
    boundaries.

    Returns
    -------
    numpy.ndarray
        3x3 row-normalized transition matrix.
    """

    from pathlib import Path

    processed_path = Path(processed_dir)

    files = sorted(
        processed_path.glob("rainfall_yavatmal_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No processed rainfall CSV files found in {processed_path}"
        )

    total_counts = pd.DataFrame(
        0,
        index=STATES,
        columns=STATES,
        dtype=int,
    )

    for file_path in files:
        dataframe = pd.read_csv(file_path)

        counts = transition_counts_dataframe(dataframe)

        total_counts = total_counts.add(
            counts,
            fill_value=0,
        )

    total_counts = total_counts.astype(int)

    matrix = np.zeros(
        (len(STATES), len(STATES)),
        dtype=float,
    )

    for i, state in enumerate(STATES):

        total = total_counts.loc[state].sum()

        if total == 0:
            raise ValueError(
                f"No outgoing transitions observed for state '{state}'."
            )

        for j, next_state in enumerate(STATES):

            matrix[i, j] = (
                total_counts.loc[state, next_state]
                / total
            )

    validate_transition_matrix(matrix)

    return matrix


def multi_year_transition_counts_dataframe(
    processed_dir="data/processed"
):
    """
    Return aggregated transition counts across all
    processed historical years.
    """

    from pathlib import Path

    processed_path = Path(processed_dir)

    files = sorted(
        processed_path.glob("rainfall_yavatmal_*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No processed rainfall CSV files found in {processed_path}"
        )

    total_counts = pd.DataFrame(
        0,
        index=STATES,
        columns=STATES,
        dtype=int,
    )

    for file_path in files:

        dataframe = pd.read_csv(file_path)

        counts = transition_counts_dataframe(dataframe)

        total_counts = total_counts.add(
            counts,
            fill_value=0,
        )

    return total_counts.astype(int)


def multi_year_transition_matrix_dataframe(
    processed_dir="data/processed"
):
    """
    Return the multi-year transition matrix as a labelled DataFrame.
    """

    matrix = calculate_multi_year_transition_matrix(
        processed_dir
    )

    return pd.DataFrame(
        matrix,
        index=STATES,
        columns=STATES,
    )



def calculate_monthly_transition_matrices_from_training(df):
    """
    Calculate leakage-safe month-specific transition matrices.

    Only observations supplied in df are used.

    A transition is counted only when observations are
    consecutive calendar days.

    This function is intended for historical backtesting,
    where df must contain only observations available before
    the historical decision date.

    State order:
        dry
        drizzle
        rain
    """

    required_columns = {
        "date",
        "rainfall_state",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values("date").reset_index(drop=True)

    states = ["dry", "drizzle", "rain"]

    monthly_counts = {
        month: pd.DataFrame(
            0,
            index=states,
            columns=states,
            dtype=int,
        )
        for month in range(1, 13)
    }

    for current_row, next_row in zip(
        data.iloc[:-1].itertuples(index=False),
        data.iloc[1:].itertuples(index=False),
    ):

        current_date = pd.Timestamp(current_row.date)
        next_date = pd.Timestamp(next_row.date)

        # Prevent transitions across missing dates,
        # year boundaries, or unrelated observations.
        if next_date != current_date + pd.Timedelta(days=1):
            continue

        current_state = current_row.rainfall_state
        next_state = next_row.rainfall_state

        if (
            current_state not in states
            or next_state not in states
        ):
            continue

        month = current_date.month

        monthly_counts[
            month
        ].loc[
            current_state,
            next_state,
        ] += 1

    monthly_matrices = {}

    for month in range(1, 13):

        counts = monthly_counts[month]

        row_totals = counts.sum(axis=1)

        matrix = counts.div(
            row_totals.replace(0, np.nan),
            axis=0,
        ).fillna(0.0)

        monthly_matrices[month] = matrix.to_numpy()

    return monthly_matrices
def calculate_monthly_transition_matrices(df):
    """
    Calculate rainfall-state transition matrices separately
    for each calendar month.

    A transition is counted only when observations are on
    consecutive calendar days. This prevents artificial
    transitions across missing dates, months, or year boundaries.

    State order:
        dry
        drizzle
        rain

    Returns
    -------
    dict
        Mapping month number -> 3x3 transition matrix.
    """

    required_columns = {
        "date",
        "rainfall_state",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = (
        data.sort_values("date")
        .reset_index(drop=True)
    )

    states = [
        "dry",
        "drizzle",
        "rain",
    ]

    monthly_counts = {
        month: pd.DataFrame(
            0,
            index=states,
            columns=states,
            dtype=int,
        )
        for month in range(1, 13)
    }

    for current_row, next_row in zip(
        data.iloc[:-1].itertuples(index=False),
        data.iloc[1:].itertuples(index=False),
    ):

        current_date = pd.Timestamp(
            current_row.date
        )

        next_date = pd.Timestamp(
            next_row.date
        )

        # Only consecutive calendar-day observations
        # are valid Markov transitions.
        if next_date != (
            current_date + pd.Timedelta(days=1)
        ):
            continue

        current_state = current_row.rainfall_state
        next_state = next_row.rainfall_state

        if (
            current_state not in states
            or next_state not in states
        ):
            continue

        # Attribute transition to the month in which
        # the current observation occurred.
        month = current_date.month

        monthly_counts[
            month
        ].loc[
            current_state,
            next_state,
        ] += 1

    monthly_matrices = {}

    for month in range(1, 13):

        counts = monthly_counts[month]

        row_totals = counts.sum(axis=1)

        matrix = counts.div(
            row_totals.replace(0, np.nan),
            axis=0,
        ).fillna(0.0)

        monthly_matrices[month] = matrix.to_numpy()

    return monthly_matrices
def get_monthly_transition_matrix_with_fallback(
    df,
    month,
    fallback_matrix=None,
):
    """
    Return a monthly rainfall transition matrix with
    state-level fallback to historical behaviour.

    A monthly row is used when that current state has at
    least one observed outgoing transition in the month.
    If the monthly row has no observations, the corresponding
    row from fallback_matrix is used.
    """

    if not 1 <= month <= 12:
        raise ValueError(
            "month must be between 1 and 12."
        )

    required_columns = {
        "date",
        "rainfall_state",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    if fallback_matrix is None:
        fallback_matrix = calculate_transition_matrix(
            data
        )

    fallback_matrix = np.asarray(
        fallback_matrix,
        dtype=float,
    )

    validate_transition_matrix(
        fallback_matrix
    )

    counts = pd.DataFrame(
        0,
        index=STATES,
        columns=STATES,
        dtype=int,
    )

    dates = data["date"].to_numpy()
    states = data["rainfall_state"].to_numpy()

    for i in range(len(data) - 1):

        current_date = pd.Timestamp(
            dates[i]
        )

        next_date = pd.Timestamp(
            dates[i + 1]
        )

        if next_date != (
            current_date + pd.Timedelta(days=1)
        ):
            continue

        if current_date.month != month:
            continue

        current_state = states[i]
        next_state = states[i + 1]

        if (
            current_state not in STATES
            or next_state not in STATES
        ):
            continue

        counts.loc[
            current_state,
            next_state,
        ] += 1

    row_totals = counts.sum(axis=1)

    monthly_matrix = counts.div(
        row_totals.replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    result = monthly_matrix.to_numpy(
        dtype=float
    )

    for i, state in enumerate(STATES):

        if row_totals.loc[state] == 0:
            result[i] = fallback_matrix[i]

    validate_transition_matrix(
        result
    )

    return result