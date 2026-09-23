from pathlib import Path

from src.calibration_artifact import CalibrationArtifact


BASE_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = BASE_DIR / "data" / "calibration"

CALIBRATION_ARTIFACTS = {
    "yavatmal": CALIBRATION_DIR / "yavatmal_rainfall_calibration_v1.json",
}


def get_calibration_artifact(location: str) -> CalibrationArtifact:
    """
    Load the versioned rainfall calibration artifact for a supported location.

    This registry intentionally supports only locations that have a validated
    production calibration artifact.
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

    artifact_path = CALIBRATION_ARTIFACTS[normalized_location]

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
