"""
Simple historical rainfall baseline for CropLogic-Saathi.

The baseline estimates future rainfall using the historical
average rainfall for the same calendar month.

This is intentionally simple and is used as a benchmark
against the probabilistic Monte Carlo approach.
"""

import pandas as pd


def validate_baseline_dataframe(dataframe):
    """Validate the minimum rainfall columns required."""

    required_columns = {
        "date",
        "rainfall_mm",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["date"] = pd.to_datetime(dataframe["date"])

    dataframe = dataframe.sort_values("date").reset_index(
        drop=True
    )

    return dataframe


def calculate_monthly_daily_rainfall_mean(
    training_data,
):
    """
    Calculate mean daily rainfall for each calendar month.

    Only training_data should be supplied.

    Returns
    -------
    dict
        Mapping:
            month number -> mean daily rainfall in mm
    """

    training_data = validate_baseline_dataframe(
        training_data
    )

    training_data["month"] = (
        training_data["date"].dt.month
    )

    monthly_means = (
        training_data
        .groupby("month")["rainfall_mm"]
        .mean()
        .to_dict()
    )

    return {
        int(month): float(mean_rainfall)
        for month, mean_rainfall
        in monthly_means.items()
    }


def calculate_historical_mean_baseline(
    training_data,
    start_date,
    horizon_days,
):
    """
    Estimate total rainfall over a future horizon using
    historical calendar-month rainfall means.

    IMPORTANT:
    Only training_data is used.

    Parameters
    ----------
    training_data : pandas.DataFrame
        Historical observations available before the
        decision date.

    start_date : str or pandas.Timestamp
        Historical decision date.

    horizon_days : int
        Number of future days to estimate.

    Returns
    -------
    float
        Expected rainfall total in mm.
    """

    if horizon_days <= 0:
        raise ValueError(
            "horizon_days must be greater than zero."
        )

    training_data = validate_baseline_dataframe(
        training_data
    )

    start_date = pd.Timestamp(start_date)

    monthly_means = (
        calculate_monthly_daily_rainfall_mean(
            training_data
        )
    )

    total_rainfall = 0.0

    for day_offset in range(horizon_days):

        simulation_date = (
            start_date
            + pd.Timedelta(days=day_offset)
        )

        month = simulation_date.month

        if month not in monthly_means:
            raise ValueError(
                f"No historical rainfall data available "
                f"for month {month}."
            )

        total_rainfall += monthly_means[month]

    return float(total_rainfall)