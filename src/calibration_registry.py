import json
from dataclasses import dataclass
from importlib.resources import files

from src.calibration_artifact import CalibrationArtifact


REGISTRY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CalibrationRegistryEntry:
    """
    Registry configuration for a public calibration location.

    The public location ID remains separate from the climate-grid identity.
    """

    artifact_path: object
    climate_source: str
    climate_grid_key: str


def _load_registry() -> dict[str, CalibrationRegistryEntry]:
    """
    Load the packaged calibration registry configuration.

    The registry is configuration data, while this module owns the runtime
    loading and validation behavior.
    """
    registry_path = files("data") / "calibration" / "registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Calibration registry not found: {registry_path}"
        )

    try:
        data = json.loads(
            registry_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid calibration registry JSON: {registry_path}"
        ) from exc

    if data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported calibration registry schema version: "
            f"{data.get('schema_version')!r}"
        )

    entries = data.get("entries")

    if not isinstance(entries, dict):
        raise ValueError(
            "Calibration registry 'entries' must be an object."
        )

    calibration_dir = files("data") / "calibration"
    registry = {}

    for location, config in entries.items():
        if not isinstance(location, str) or not location.strip():
            raise ValueError(
                "Calibration registry contains an empty location ID."
            )

        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid registry entry for location '{location}'."
            )

        artifact_path = config.get("artifact_path")
        climate_source = config.get("climate_source")
        climate_grid_key = config.get("climate_grid_key")

        if (
            not isinstance(artifact_path, str)
            or not artifact_path.strip()
        ):
            raise ValueError(
                f"Registry entry '{location}' has an invalid "
                "artifact_path."
            )

        if (
            not isinstance(climate_source, str)
            or not climate_source.strip()
        ):
            raise ValueError(
                f"Registry entry '{location}' has an invalid "
                "climate_source."
            )

        if (
            not isinstance(climate_grid_key, str)
            or not climate_grid_key.strip()
        ):
            raise ValueError(
                f"Registry entry '{location}' has an invalid "
                "climate_grid_key."
            )

        registry[location.strip().lower()] = CalibrationRegistryEntry(
            artifact_path=calibration_dir / artifact_path,
            climate_source=climate_source,
            climate_grid_key=climate_grid_key,
        )

    return registry

CALIBRATION_ARTIFACTS = _load_registry()


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
