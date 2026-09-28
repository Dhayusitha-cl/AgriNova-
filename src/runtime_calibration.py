"""
Runtime resolution of validated calibration artifacts from geographic input.

This module connects:
    geographic coordinates
        -> authoritative climate-data grid
        -> registered calibration artifact

It does not download data, build calibration artifacts, or run decisions.
"""

from src.calibration_artifact import CalibrationArtifact
from src.calibration_registry import get_calibration_artifact_for_grid
from src.climate_data_provider import ClimateDataProvider
from src.imd_data import resolve_imd_climate_grid
from src.location import GeographicLocation


def resolve_calibration_artifact(
    location: GeographicLocation,
    climate_dataset,
) -> CalibrationArtifact:
    """
    Resolve geographic coordinates to a validated calibration artifact.

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

    return get_calibration_artifact_for_grid(
        climate_source=climate_grid.source,
        climate_grid_key=climate_grid.key,
    )


def resolve_calibration_artifact_from_provider(
    location: GeographicLocation,
    climate_provider: ClimateDataProvider,
) -> CalibrationArtifact:
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
        return resolve_calibration_artifact(
            location=location,
            climate_dataset=dataset,
        )
    finally:
        close = getattr(dataset, "close", None)

        if callable(close):
            close()