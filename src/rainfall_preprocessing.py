"""
Rainfall preprocessing for AgriNova.

Converts IMD gridded rainfall data into a clean daily rainfall
time series for a selected location.

This module performs data preparation only.
It does not perform weather simulation or decision making.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_daily_rainfall(
    dataset_path,
    latitude,
    longitude,
):
    """
    Load daily IMD rainfall for the nearest available grid point.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the IMD NetCDF rainfall file.

    latitude : float
        Requested latitude in decimal degrees.

    longitude : float
        Requested longitude in decimal degrees.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - date
        - rainfall_mm
        - latitude
        - longitude
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Rainfall dataset not found: {dataset_path}"
        )

    with xr.open_dataset(dataset_path) as ds:
        rainfall = ds["RAINFALL"].sel(
            LATITUDE=latitude,
            LONGITUDE=longitude,
            method="nearest",
        )

        selected_latitude = float(rainfall.LATITUDE)
        selected_longitude = float(rainfall.LONGITUDE)

        dates = pd.to_datetime(rainfall["TIME"].values)
        values = rainfall.values.astype(float)

    dataframe = pd.DataFrame(
        {
            "date": dates,
            "rainfall_mm": values,
        }
    )

    dataframe["latitude"] = selected_latitude
    dataframe["longitude"] = selected_longitude

    return dataframe


def validate_rainfall(dataframe):
    """
    Validate and clean a daily rainfall DataFrame.

    Negative rainfall values are treated as invalid.
    Missing rainfall values are preserved as NaN for visibility.

    Returns
    -------
    pandas.DataFrame
        Validated rainfall data.
    """

    dataframe = dataframe.copy()

    if "rainfall_mm" not in dataframe.columns:
        raise ValueError("Missing required column: rainfall_mm")

    negative_values = dataframe["rainfall_mm"] < 0

    if negative_values.any():
        raise ValueError(
            "Rainfall dataset contains negative rainfall values."
        )

    dataframe = dataframe.sort_values("date").reset_index(drop=True)

    return dataframe


def classify_rainfall(rainfall_mm):
    """
    Convert rainfall amount into a simple weather state.

    Current thresholds are modelling assumptions for the
    first implementation and must be validated later.

    States
    ------
    dry:
        rainfall < 1 mm

    drizzle:
        1 mm <= rainfall < 10 mm

    rain:
        rainfall >= 10 mm
    """

    if pd.isna(rainfall_mm):
        return "missing"

    if rainfall_mm < 1.0:
        return "dry"

    if rainfall_mm < 10.0:
        return "drizzle"

    return "rain"


def add_rainfall_state(dataframe):
    """
    Add a rainfall_state column.
    """

    dataframe = dataframe.copy()

    dataframe["rainfall_state"] = dataframe["rainfall_mm"].apply(
        classify_rainfall
    )

    return dataframe


def preprocess_rainfall(
    dataset_path,
    latitude,
    longitude,
):
    """
    Complete rainfall preprocessing pipeline.
    """

    dataframe = load_daily_rainfall(
        dataset_path=dataset_path,
        latitude=latitude,
        longitude=longitude,
    )

    dataframe = validate_rainfall(dataframe)

    dataframe = add_rainfall_state(dataframe)

    return dataframe


def save_processed_rainfall(dataframe, output_path):
    """
    Save processed rainfall data to CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    dataset = Path(
        "data/raw/RF25_ind2024_rfp25.nc"
    )

    output = Path(
        "data/processed/rainfall_yavatmal_2024.csv"
    )

    rainfall = preprocess_rainfall(
        dataset_path=dataset,
        latitude=20.39,
        longitude=78.13,
    )

    save_processed_rainfall(
        rainfall,
        output,
    )

    print("Rainfall preprocessing completed.")
    print(
        f"Selected grid point: "
        f"{rainfall['latitude'].iloc[0]}, "
        f"{rainfall['longitude'].iloc[0]}"
    )
    print(f"Days processed: {len(rainfall)}")
    print(
        "Rainfall states:"
    )
    print(
        rainfall["rainfall_state"].value_counts()
    )
    print(f"Saved to: {output}")