"""
Offline calibration onboarding workflow for CropLogic-Saathi.

This module orchestrates existing preprocessing and calibration functions.
It does not perform runtime decision making or automatically register
calibration artifacts for production use.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.build_calibration_artifact import (
    build_and_save_calibration_artifact,
)
from src.calibration_artifact import CalibrationArtifact
from src.process_imd_years import (
    process_year,
)


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def onboard_calibration(
    *,
    location_id: str,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    output_path: str | Path,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
):
    """
    Build a validated rainfall calibration artifact for a new location.

    Historical rainfall is first extracted from the supplied IMD yearly
    datasets using the existing preprocessing path. Only the processed files
    generated for the requested year range are staged for calibration.

    Runtime registry registration remains a separate explicit operation.
    """

    if not isinstance(location_id, str) or not location_id.strip():
        raise ValueError("location_id must not be empty.")

    if start_year > end_year:
        raise ValueError(
            "start_year must not be greater than end_year."
        )

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    output_path = Path(output_path)

    processed_files = []
    resolved_grids = set()

    for year in range(start_year, end_year + 1):
        dataframe = process_year(
            year,
            latitude=latitude,
            longitude=longitude,
            location_id=location_id,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )

        processed_file = (
            processed_dir / f"rainfall_{location_id}_{year}.csv"
        )

        if not processed_file.exists():
            raise FileNotFoundError(
                "Expected processed rainfall file was not created: "
                f"{processed_file}"
            )

        processed_files.append(processed_file)

        resolved_grids.add(
            (
                float(dataframe["latitude"].iloc[0]),
                float(dataframe["longitude"].iloc[0]),
            )
        )

    if len(resolved_grids) != 1:
        raise ValueError(
            "Historical datasets resolved to inconsistent climate grids: "
            f"{sorted(resolved_grids)}"
        )

    with tempfile.TemporaryDirectory(
        prefix="croplogic_calibration_"
    ) as staging_dir:
        staging_dir = Path(staging_dir)

        for processed_file in processed_files:
            shutil.copy2(
                processed_file,
                staging_dir / processed_file.name,
            )

        artifact = build_and_save_calibration_artifact(
            output_path=output_path,
            data_dir=staging_dir,
            pattern=f"rainfall_{location_id}_*.csv",
            location=location_id,
        )

    selected_latitude, selected_longitude = next(
        iter(resolved_grids)
    )

    climate_grid_key = (
        f"imd_gridded_rainfall:"
        f"{selected_latitude:.2f}:"
        f"{selected_longitude:.2f}"
    )

    return {
        "artifact": artifact,
        "artifact_path": output_path,
        "processed_files": processed_files,
        "climate_source": "imd_gridded_rainfall",
        "climate_grid_key": climate_grid_key,
        "selected_latitude": selected_latitude,
        "selected_longitude": selected_longitude,
    }


def register_calibration(
    *,
    registry_path: str | Path,
    location_id: str,
    artifact_path: str | Path,
    climate_source: str,
    climate_grid_key: str,
    overwrite: bool = False,
):
    """
    Explicitly register a previously validated calibration artifact.

    Registration is an offline/deployment configuration operation.
    It does not perform calibration or runtime decision making.

    The registry is replaced atomically only after all validation succeeds.
    """

    if not isinstance(location_id, str) or not location_id.strip():
        raise ValueError("Calibration location must not be empty.")

    if not isinstance(climate_source, str) or not climate_source.strip():
        raise ValueError("Climate source must not be empty.")

    if not isinstance(climate_grid_key, str) or not climate_grid_key.strip():
        raise ValueError("Climate grid key must not be empty.")

    registry_path = Path(registry_path)
    source_path = Path(artifact_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Calibration artifact not found: {source_path}"
        )

    if not source_path.is_file():
        raise ValueError(
            f"Calibration artifact path is not a file: {source_path}"
        )

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Calibration registry not found: {registry_path}"
        )

    if not registry_path.is_file():
        raise ValueError(
            f"Calibration registry path is not a file: {registry_path}"
        )

    normalized_location = location_id.strip().lower()
    normalized_source = climate_source.strip().lower()
    normalized_grid_key = climate_grid_key.strip().lower()

    artifact = CalibrationArtifact.load(source_path)

    if artifact.location.strip().lower() != normalized_location:
        raise ValueError(
            f"Calibration artifact location mismatch: expected "
            f"'{normalized_location}', got '{artifact.location}'."
        )

    try:
        registry_data = json.loads(
            registry_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid calibration registry JSON: {registry_path}"
        ) from exc

    if registry_data.get("schema_version") != "1.0":
        raise ValueError(
            "Unsupported calibration registry schema version: "
            f"{registry_data.get('schema_version')!r}"
        )

    entries = registry_data.get("entries")

    if not isinstance(entries, dict):
        raise ValueError(
            "Calibration registry 'entries' must be an object."
        )

    if normalized_location in entries and not overwrite:
        raise ValueError(
            f"Calibration location '{normalized_location}' is already "
            "registered. Set overwrite=True to replace it explicitly."
        )

    try:
        relative_artifact_path = source_path.resolve().relative_to(
            registry_path.parent.resolve()
        )
    except ValueError as exc:
        raise ValueError(
            "Calibration artifact must be located inside the "
            "calibration registry directory."
        ) from exc

    entries[normalized_location] = {
        "artifact_path": str(relative_artifact_path).replace("\\", "/"),
        "climate_source": normalized_source,
        "climate_grid_key": normalized_grid_key,
    }

    serialized = json.dumps(
        registry_data,
        indent=2,
        sort_keys=True,
    ) + "\n"

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=registry_path.parent,
        prefix=".registry.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(serialized)

    try:
        temporary_path.replace(registry_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise