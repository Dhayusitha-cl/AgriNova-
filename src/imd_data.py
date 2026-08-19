"""IMD gridded rainfall data ingestion utilities.

This module reads IMD 0.25-degree gridded rainfall NetCDF files
and extracts a daily rainfall series for the nearest grid cell.

The extracted observations are data inputs. They are not forecasts.
"""

from pathlib import Path

import numpy as np
import xarray as xr


def open_rainfall_dataset(file_path):
    """Open an IMD gridded rainfall NetCDF dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to the IMD NetCDF file.

    Returns
    -------
    xarray.Dataset
        Open rainfall dataset.

    Raises
    ------
    FileNotFoundError
        If the supplied file does not exist.
    ValueError
        If the expected RAINFALL variable or dimensions are missing.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Rainfall dataset not found: {path}")

    dataset = xr.open_dataset(path)

    required_dimensions = {"TIME", "LATITUDE", "LONGITUDE"}

    if not required_dimensions.issubset(dataset.dims):
        dataset.close()
        raise ValueError(
            "Dataset must contain TIME, LATITUDE and LONGITUDE dimensions."
        )

    if "RAINFALL" not in dataset.data_vars:
        dataset.close()
        raise ValueError("Dataset does not contain the expected RAINFALL variable.")

    return dataset


def extract_daily_rainfall(dataset, latitude, longitude):
    """Extract rainfall from the nearest IMD grid cell.

    Parameters
    ----------
    dataset : xarray.Dataset
        Open IMD rainfall dataset.
    latitude : float
        Requested latitude in decimal degrees.
    longitude : float
        Requested longitude in decimal degrees.

    Returns
    -------
    dict
        Daily rainfall information including the selected grid cell
        and rainfall observations.
    """

    rainfall = dataset["RAINFALL"].sel(
        LATITUDE=latitude,
        LONGITUDE=longitude,
        method="nearest",
    )

    values = np.asarray(rainfall.values, dtype=float)

    if np.isnan(values).any():
        raise ValueError("Extracted rainfall series contains missing values.")

    return {
        "requested_latitude": float(latitude),
        "requested_longitude": float(longitude),
        "selected_latitude": float(rainfall.LATITUDE),
        "selected_longitude": float(rainfall.LONGITUDE),
        "time": rainfall.TIME.values,
        "rainfall_mm": values,
    }


def summarize_rainfall(extracted_data):
    """Return basic quality statistics for an extracted rainfall series."""

    rainfall = extracted_data["rainfall_mm"]

    return {
        "days": int(len(rainfall)),
        "missing_values": int(np.isnan(rainfall).sum()),
        "min_mm": float(np.min(rainfall)),
        "max_mm": float(np.max(rainfall)),
        "mean_mm": float(np.mean(rainfall)),
        "zero_rain_days": int(np.sum(rainfall == 0)),
    }