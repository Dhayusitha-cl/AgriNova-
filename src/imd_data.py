"""IMD gridded rainfall data ingestion utilities.

This module reads IMD 0.25-degree gridded rainfall NetCDF files
and extracts a daily rainfall series for the nearest grid cell.

The extracted observations are data inputs. They are not forecasts.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from src.location import (
    ClimateGridLocation,
    GeographicLocation,
    resolve_climate_grid,
)


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


def resolve_imd_climate_grid(
    dataset,
    location: GeographicLocation,
) -> ClimateGridLocation:
    """Resolve a geographic location using the coordinates of an IMD dataset.

    The dataset supplies the authoritative latitude and longitude grid.
    """

    if not isinstance(location, GeographicLocation):
        raise TypeError(
            "location must be a GeographicLocation instance."
        )

    return resolve_climate_grid(
        location,
        dataset["LATITUDE"].values,
        dataset["LONGITUDE"].values,
        source="imd_gridded_rainfall",
    )


def extract_daily_rainfall(
    dataset,
    location: GeographicLocation,
):
    """Extract rainfall from the canonical IMD grid cell.

    Parameters
    ----------
    dataset : xarray.Dataset
        Open IMD rainfall dataset.
    location : GeographicLocation
        Requested geographic location.

    Returns
    -------
    dict
        Daily rainfall information including the requested geographic
        location, selected climate grid cell, climate-grid key, and
        rainfall observations.
    """

    climate_grid = resolve_imd_climate_grid(
        dataset,
        location,
    )

    rainfall = dataset["RAINFALL"].sel(
        LATITUDE=climate_grid.latitude,
        LONGITUDE=climate_grid.longitude,
    )

    values = np.asarray(rainfall.values, dtype=float)

    if np.isnan(values).any():
        raise ValueError(
            "Extracted rainfall series contains missing values."
        )

    return {
        "requested_latitude": float(location.latitude),
        "requested_longitude": float(location.longitude),
        "selected_latitude": float(climate_grid.latitude),
        "selected_longitude": float(climate_grid.longitude),
        "climate_source": climate_grid.source,
        "climate_grid_key": climate_grid.key,
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