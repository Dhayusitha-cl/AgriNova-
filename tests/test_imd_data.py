import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.imd_data import (
    extract_daily_rainfall,
    resolve_imd_climate_grid,
)
from src.location import GeographicLocation


def make_test_dataset():
    times = pd.date_range(
        "2024-01-01",
        periods=3,
        freq="D",
    )

    latitudes = np.array(
        [20.0, 20.25, 20.5],
        dtype=float,
    )

    longitudes = np.array(
        [77.75, 78.0, 78.25],
        dtype=float,
    )

    rainfall = np.arange(
        len(times) * len(latitudes) * len(longitudes),
        dtype=float,
    ).reshape(
        len(times),
        len(latitudes),
        len(longitudes),
    )

    return xr.Dataset(
        {
            "RAINFALL": (
                ("TIME", "LATITUDE", "LONGITUDE"),
                rainfall,
            ),
        },
        coords={
            "TIME": times,
            "LATITUDE": latitudes,
            "LONGITUDE": longitudes,
        },
    )


def test_resolve_imd_climate_grid_uses_dataset_coordinates():
    dataset = make_test_dataset()

    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    result = resolve_imd_climate_grid(
        dataset,
        location,
    )

    assert result.source == "imd_gridded_rainfall"
    assert result.latitude == 20.5
    assert result.longitude == 78.0
    assert result.key == "imd_gridded_rainfall:20.50:78.00"


def test_extract_daily_rainfall_uses_canonical_grid():
    dataset = make_test_dataset()

    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    result = extract_daily_rainfall(
        dataset,
        location,
    )

    assert result["requested_latitude"] == 20.39
    assert result["requested_longitude"] == 78.12

    assert result["selected_latitude"] == 20.5
    assert result["selected_longitude"] == 78.0

    assert result["climate_source"] == "imd_gridded_rainfall"
    assert result["climate_grid_key"] == (
        "imd_gridded_rainfall:20.50:78.00"
    )

    assert len(result["time"]) == 3
    assert len(result["rainfall_mm"]) == 3


def test_extract_daily_rainfall_rejects_missing_values():
    dataset = make_test_dataset()

    dataset["RAINFALL"][0, 2, 1] = np.nan

    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        extract_daily_rainfall(
            dataset,
            location,
        )


def test_resolve_imd_climate_grid_requires_geographic_location():
    dataset = make_test_dataset()

    with pytest.raises(
        TypeError,
        match="GeographicLocation",
    ):
        resolve_imd_climate_grid(
            dataset,
            (20.39, 78.12),
        )