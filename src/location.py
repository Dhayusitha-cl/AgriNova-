"""
Canonical geographic and climate-location resolution.

This module separates:
- the caller's geographic coordinates
- the climate grid location used for historical calibration

The climate grid is resolved from the actual coordinates supplied by the
climate-data source rather than assuming a fixed grid in this module.
"""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class GeographicLocation:
    """A requested geographic location."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude):
            raise ValueError("Latitude must be finite.")

        if not math.isfinite(self.longitude):
            raise ValueError("Longitude must be finite.")

        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(
                "Longitude must be between -180 and 180 degrees."
            )


@dataclass(frozen=True)
class ClimateGridLocation:
    """Canonical climate-grid location resolved from geographic coordinates."""

    source: str
    latitude: float
    longitude: float
    key: str


def _nearest_coordinate(
    value: float,
    coordinates: Sequence[float],
) -> float:
    """Return the nearest coordinate from an available source grid."""

    if len(coordinates) == 0:
        raise ValueError("Coordinate grid must not be empty.")

    finite_coordinates = [
        float(coordinate)
        for coordinate in coordinates
        if math.isfinite(float(coordinate))
    ]

    if not finite_coordinates:
        raise ValueError("Coordinate grid must contain finite values.")

    return min(
        finite_coordinates,
        key=lambda coordinate: (abs(coordinate - value), coordinate),
    )


def resolve_climate_grid(
    location: GeographicLocation,
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    *,
    source: str = "imd_gridded_rainfall",
) -> ClimateGridLocation:
    """
    Resolve a geographic location to the nearest available climate grid cell.

    The actual latitude/longitude arrays from the source dataset are supplied
    by the caller. This avoids assuming that every future climate source uses
    the same spatial resolution or coordinate convention.
    """

    if not isinstance(location, GeographicLocation):
        raise TypeError(
            "location must be a GeographicLocation instance."
        )

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Climate source must not be empty.")

    selected_latitude = _nearest_coordinate(
        location.latitude,
        latitudes,
    )
    selected_longitude = _nearest_coordinate(
        location.longitude,
        longitudes,
    )

    normalized_source = source.strip().lower()

    key = (
        f"{normalized_source}:"
        f"{selected_latitude:.2f}:"
        f"{selected_longitude:.2f}"
    )

    return ClimateGridLocation(
        source=normalized_source,
        latitude=selected_latitude,
        longitude=selected_longitude,
        key=key,
    )