from dataclasses import dataclass
from importlib.resources import files

from src.calibration_artifact import CalibrationArtifact


@dataclass(frozen=True)
class CalibrationRegistryEntry:
    """
    Registry configuration for a public calibration location.

    The public location ID remains separate from the climate-grid identity.
    """

    artifact_path: object
    climate_source: str
    climate_grid_key: str


CALIBRATION_ARTIFACTS = {
    "yavatmal": CalibrationRegistryEntry(
        artifact_path=(
            files("data")
            / "calibration"
            / "yavatmal_rainfall_calibration_v1.json"
        ),
        climate_source="imd_gridded_rainfall",
        climate_grid_key="imd_gridded_rainfall:20.50:78.00",
    ),
}


def get_calibration_artifact(location: str) -> CalibrationArtifact:
    """
    Load the versioned rainfall calibration artifact for a supported location.

    Public location IDs remain semantic identifiers. Climate-grid identity is
    stored separately in the registry entry.
    """
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Calibration location must not be empty.")

    normalized_location = location.strip().lower()

    if normalized_location not in CALIBRATION_ARTIFACTS:
        supported = ", ".join(sorted(CALIBRATION_ARTIFACTS))
        raise ValueError(
            f"No calibration artifact is configured for location "
            f"'{location}'. Supported locations: {supported}."
        )

    entry = CALIBRATION_ARTIFACTS[normalized_location]
    artifact_path = entry.artifact_path

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found for location "
            f"'{normalized_location}': {artifact_path}"
        )

    artifact = CalibrationArtifact.load(artifact_path)

    if artifact.location.lower() != normalized_location:
        raise ValueError(
            f"Calibration artifact location mismatch: expected "
            f"'{normalized_location}', got '{artifact.location}'."
        )

    return artifact


def get_calibration_artifact_for_grid(
    climate_source: str,
    climate_grid_key: str,
) -> CalibrationArtifact:
    """
    Load the validated calibration artifact registered for a climate grid.

    Climate-grid identity is resolved independently from the public location
    identifier. An unknown grid never falls back to another calibration.
    """
    if not isinstance(climate_source, str) or not climate_source.strip():
        raise ValueError("Climate source must not be empty.")

    if not isinstance(climate_grid_key, str) or not climate_grid_key.strip():
        raise ValueError("Climate grid key must not be empty.")

    normalized_source = climate_source.strip().lower()
    normalized_grid_key = climate_grid_key.strip().lower()

    for location, entry in CALIBRATION_ARTIFACTS.items():
        if (
            entry.climate_source.strip().lower() == normalized_source
            and entry.climate_grid_key.strip().lower()
            == normalized_grid_key
        ):
            return get_calibration_artifact(location)

    raise ValueError(
        "No calibration artifact is configured for climate grid "
        f"'{climate_grid_key}' from source '{climate_source}'."
    )
