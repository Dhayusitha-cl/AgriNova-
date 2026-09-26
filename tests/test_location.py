import math

import pytest

from src.location import (
    ClimateGridLocation,
    GeographicLocation,
    resolve_climate_grid,
)


LATITUDES = [20.0, 20.25, 20.5, 20.75]
LONGITUDES = [77.75, 78.0, 78.25, 78.5]


def test_geographic_location_accepts_valid_coordinates():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    assert location.latitude == 20.39
    assert location.longitude == 78.12


@pytest.mark.parametrize(
    "latitude",
    [-90.1, 90.1, math.nan, math.inf, -math.inf],
)
def test_geographic_location_rejects_invalid_latitude(latitude):
    with pytest.raises(ValueError):
        GeographicLocation(
            latitude=latitude,
            longitude=78.12,
        )


@pytest.mark.parametrize(
    "longitude",
    [-180.1, 180.1, math.nan, math.inf, -math.inf],
)
def test_geographic_location_rejects_invalid_longitude(longitude):
    with pytest.raises(ValueError):
        GeographicLocation(
            latitude=20.39,
            longitude=longitude,
        )


def test_resolves_to_nearest_grid_cell():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    result = resolve_climate_grid(
        location,
        LATITUDES,
        LONGITUDES,
    )

    assert isinstance(result, ClimateGridLocation)
    assert result.latitude == 20.5
    assert result.longitude == 78.0
    assert result.source == "imd_gridded_rainfall"
    assert result.key == "imd_gridded_rainfall:20.50:78.00"

def test_exact_grid_coordinate_is_preserved():
    location = GeographicLocation(
        latitude=20.5,
        longitude=78.0,
    )

    result = resolve_climate_grid(
        location,
        LATITUDES,
        LONGITUDES,
    )

    assert result.latitude == 20.5
    assert result.longitude == 78.0


def test_same_grid_cell_has_same_key():
    first = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    second = GeographicLocation(
        latitude=20.41,
        longitude=78.09,
    )

    first_result = resolve_climate_grid(
        first,
        LATITUDES,
        LONGITUDES,
    )

    second_result = resolve_climate_grid(
        second,
        LATITUDES,
        LONGITUDES,
    )

    assert first_result.key == second_result.key


def test_different_grid_cells_have_different_keys():
    first = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    second = GeographicLocation(
        latitude=20.70,
        longitude=78.30,
    )

    first_result = resolve_climate_grid(
        first,
        LATITUDES,
        LONGITUDES,
    )

    second_result = resolve_climate_grid(
        second,
        LATITUDES,
        LONGITUDES,
    )

    assert first_result.key != second_result.key


def test_rejects_empty_latitude_grid():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    with pytest.raises(ValueError):
        resolve_climate_grid(
            location,
            [],
            LONGITUDES,
        )


def test_rejects_empty_longitude_grid():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    with pytest.raises(ValueError):
        resolve_climate_grid(
            location,
            LATITUDES,
            [],
        )


def test_source_is_normalized():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    result = resolve_climate_grid(
        location,
        LATITUDES,
        LONGITUDES,
        source="  IMD_Gridded_Rainfall  ",
    )

    assert result.source == "imd_gridded_rainfall"

def test_resolves_numpy_coordinate_arrays():
    import numpy as np

    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    latitudes = np.array([20.0, 20.25, 20.5, 20.75])
    longitudes = np.array([77.75, 78.0, 78.25, 78.5])

    result = resolve_climate_grid(
        location,
        latitudes,
        longitudes,
    )

    assert result.latitude == 20.5
    assert result.longitude == 78.0