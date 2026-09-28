"""
Runtime resolution of validated calibration artifacts from geographic input.

This module connects:
    geographic coordinates
        -> authoritative climate-data grid
        -> registered calibration artifact

It does not download data, build calibration artifacts, or run decisions.
"""

from dataclasses import dataclass

from src.calibration_artifact import CalibrationArtifact
from src.calibration_registry import get_calibration_artifact_for_grid
from src.climate_data_provider import ClimateDataProvider
from src.imd_data import resolve_imd_climate_grid
from src.location import ClimateGridLocation, GeographicLocation


@dataclass(frozen=True)
class CalibrationResolution:
    """Resolved runtime climate-grid and calibration identity."""

    climate_grid: ClimateGridLocation
    artifact: CalibrationArtifact


def resolve_calibration_resolution(
    location: GeographicLocation,
    climate_dataset,
) -> CalibrationResolution:
    """
    Resolve geographic coordinates to a climate grid and calibration artifact.

    The climate dataset supplies the authoritative IMD coordinate grid.
    The calibration registry supplies the registered artifact for the
    resolved climate source/grid identity.
    """

    if not isinstance(location, GeographicLocation):
        raise TypeError(
            "location must be a GeographicLocation instance."
        )

    climate_grid = resolve_imd_climate_grid(
        climate_dataset,
        location,
    )

    artifact = get_calibration_artifact_for_grid(
        climate_source=climate_grid.source,
        climate_grid_key=climate_grid.key,
    )

    return CalibrationResolution(
        climate_grid=climate_grid,
        artifact=artifact,
    )


def resolve_calibration_artifact(
    location: GeographicLocation,
    climate_dataset,
) -> CalibrationArtifact:
    """
    Resolve geographic coordinates to a validated calibration artifact.
    """

    resolution = resolve_calibration_resolution(
        location=location,
        climate_dataset=climate_dataset,
    )

    return resolution.artifact


def resolve_calibration_resolution_from_provider(
    location: GeographicLocation,
    climate_provider: ClimateDataProvider,
) -> CalibrationResolution:
    """
    Resolve geographic coordinates using a climate-data provider.

    The provider owns climate-dataset acquisition. This function owns only
    the runtime geographic-to-calibration resolution.
    """

    if not isinstance(location, GeographicLocation):
        raise TypeError(
            "location must be a GeographicLocation instance."
        )

    if not isinstance(climate_provider, ClimateDataProvider):
        raise TypeError(
            "climate_provider must be a ClimateDataProvider instance."
        )

    dataset = climate_provider.get_dataset()

    try:
        return resolve_calibration_resolution(
            location=location,
            climate_dataset=dataset,
        )
    finally:
        close = getattr(dataset, "close", None)

        if callable(close):
            close()


def resolve_calibration_artifact_from_provider(
    location: GeographicLocation,
    climate_provider: ClimateDataProvider,
) -> CalibrationArtifact:
    """
    Resolve geographic coordinates using a climate-data provider and return
    only the validated calibration artifact.

    Kept as a compatibility wrapper for existing callers.
    """

    resolution = resolve_calibration_resolution_from_provider(
        location=location,
        climate_provider=climate_provider,
    )

    return resolution.artifact