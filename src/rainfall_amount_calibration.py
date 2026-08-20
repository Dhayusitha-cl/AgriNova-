"""
Calibration of rainfall amounts by rainfall state.

Uses observed IMD rainfall data from the processed multi-year
rainfall CSV files.

Important:
These are empirical distributions from the available dataset.
They are not forecasts or guarantees.
"""

from pathlib import Path

import pandas as pd


STATES = ["dry", "drizzle", "rain"]


def load_processed_rainfall(
    data_dir="data/processed",
    pattern="rainfall_yavatmal_*.csv",
):
    """
    Load and combine processed multi-year rainfall data.
    """

    data_path = Path(data_dir)
    files = sorted(data_path.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No processed rainfall files found in {data_path}"
        )

    frames = []

    for file in files:
        dataframe = pd.read_csv(file)

        required_columns = {
            "date",
            "rainfall_mm",
            "rainfall_state",
        }

        missing = required_columns - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{file} is missing columns: {sorted(missing)}"
            )

        frames.append(dataframe)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(
        combined["date"]
    )

    return combined.sort_values("date").reset_index(drop=True)


def calculate_state_statistics(dataframe):
    """
    Calculate observed rainfall statistics for each state.
    """

    results = {}

    for state in STATES:
        values = dataframe.loc[
            dataframe["rainfall_state"] == state,
            "rainfall_mm",
        ]

        if values.empty:
            raise ValueError(
                f"No rainfall observations found for state '{state}'."
            )

        results[state] = {
            "count": int(values.count()),
            "mean_mm": float(values.mean()),
            "median_mm": float(values.median()),
            "std_mm": float(values.std(ddof=0)),
            "min_mm": float(values.min()),
            "max_mm": float(values.max()),
            "p25_mm": float(values.quantile(0.25)),
            "p75_mm": float(values.quantile(0.75)),
            "p90_mm": float(values.quantile(0.90)),
            "p95_mm": float(values.quantile(0.95)),
        }

    return results
def sample_rainfall_amounts(
    dataframe,
    state,
    size=1,
    random_seed=None,
):
    """
    Sample rainfall amounts directly from observed IMD values
    belonging to a rainfall state.

    This is empirical resampling, not a fitted parametric
    probability distribution.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Combined processed IMD rainfall data.
    state : str
        dry, drizzle, or rain.
    size : int
        Number of rainfall amounts to sample.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Sampled rainfall amounts in mm.
    """

    import numpy as np

    if state not in STATES:
        raise ValueError(
            f"Unknown rainfall state: {state}"
        )

    if size <= 0:
        raise ValueError(
            "size must be greater than zero."
        )

    values = dataframe.loc[
        dataframe["rainfall_state"] == state,
        "rainfall_mm",
    ].to_numpy(dtype=float)

    if len(values) == 0:
        raise ValueError(
            f"No rainfall observations found for state '{state}'."
        )

    rng = np.random.default_rng(random_seed)

    return rng.choice(
        values,
        size=size,
        replace=True,
    )
def sample_month_state_rainfall_amounts(
    dataframe,
    month,
    state,
    size=1,
    random_seed=None,
):
    """
    Sample rainfall amounts conditioned on both
    calendar month and rainfall state.

    This is empirical resampling from observed
    training data.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Historical training rainfall data.

    month : int
        Calendar month, 1-12.

    state : str
        dry, drizzle, or rain.

    size : int
        Number of rainfall amounts to sample.

    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Sampled rainfall amounts in mm.

    Notes
    -----
    Only observations belonging to the requested
    month and rainfall state are sampled.

    During backtesting, dataframe must contain
    training data only.
    """

    import numpy as np

    if state not in STATES:
        raise ValueError(
            f"Unknown rainfall state: {state}"
        )

    if not 1 <= month <= 12:
        raise ValueError(
            "month must be between 1 and 12."
        )

    if size <= 0:
        raise ValueError(
            "size must be greater than zero."
        )

    data = dataframe.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    monthly_values = data.loc[
        (
            data["date"].dt.month == month
        )
        &
        (
            data["rainfall_state"] == state
        ),
        "rainfall_mm",
    ].to_numpy(dtype=float)

    # If the month/state combination has no
    # observations, fall back to the state-level
    # empirical distribution.
    if len(monthly_values) == 0:

        monthly_values = data.loc[
            data["rainfall_state"] == state,
            "rainfall_mm",
        ].to_numpy(dtype=float)

    if len(monthly_values) == 0:
        raise ValueError(
            f"No rainfall observations found for "
            f"state '{state}'."
        )

    rng = np.random.default_rng(
        random_seed
    )

    return rng.choice(
        monthly_values,
        size=size,
        replace=True,
    )