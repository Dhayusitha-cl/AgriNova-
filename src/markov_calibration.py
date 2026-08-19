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