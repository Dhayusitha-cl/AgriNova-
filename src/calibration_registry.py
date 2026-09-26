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