"""
Offline calibration onboarding workflow for CropLogic-Saathi.

This module orchestrates existing preprocessing and calibration functions.
It does not perform runtime decision making or automatically register
calibration artifacts for production use.
"""

from __future__ import annotations

import math
import re

import pandas as pd
import json
import shutil
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.build_calibration_artifact import (
    build_and_save_calibration_artifact,
)
from src.calibration_artifact import CalibrationArtifact
from src.calibration_registry import REGISTRY_SCHEMA_VERSION
from src.process_imd_years import (
    process_year,
)


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def _validate_location_and_coordinates(
    location_id: str,
    latitude: float,
    longitude: float,
) -> tuple[str, float, float]:
    """Validate the identifier and requested geographic coordinates."""

    if not isinstance(location_id, str):
        raise ValueError("location_id must be a string.")

    location_id = location_id.strip().lower()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", location_id):
        raise ValueError(
            "location_id must contain only lowercase letters, "
            "numbers, underscores, and hyphens, and must start "
            "with a letter or number."
        )

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Latitude and longitude must be numeric."
        ) from exc

    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("latitude must be finite and between -90 and 90.")

    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError(
            "longitude must be finite and between -180 and 180."
        )

    return location_id, latitude, longitude

def _get_single_csv_grid_from_dataframe(
    dataframe: pd.DataFrame,
    expected_year: int,
) -> tuple[float, float]:
    """Check that the returned preprocessing DataFrame matches its CSV."""

    required = {"date", "latitude", "longitude"}
    missing = required.difference(dataframe.columns)

    if missing or dataframe.empty:
        raise ValueError(
            "Preprocessing returned an empty DataFrame or one missing "
            "date/latitude/longitude columns."
        )

    dates = pd.to_datetime(dataframe["date"], errors="coerce")

    if dates.isna().any() or not (dates.dt.year == expected_year).all():
        raise ValueError(
            f"Preprocessing returned invalid dates for year {expected_year}."
        )

    latitudes = pd.to_numeric(dataframe["latitude"], errors="coerce")
    longitudes = pd.to_numeric(dataframe["longitude"], errors="coerce")

    if (
        latitudes.isna().any()
        or longitudes.isna().any()
        or not latitudes.map(math.isfinite).all()
        or not longitudes.map(math.isfinite).all()
    ):
        raise ValueError(
            "Preprocessing returned invalid climate-grid coordinates."
        )

    grids = set(
        zip(
            latitudes.astype(float),
            longitudes.astype(float),
        )
    )

    if len(grids) != 1:
        raise ValueError(
            "Preprocessing returned inconsistent climate grids: "
            f"{sorted(grids)}"
        )

    return next(iter(grids))



def _get_single_csv_grid(
    processed_file: Path,
    expected_year: int,
) -> tuple[float, float]:
    """
    Validate the generated CSV's date year and single resolved grid.

    Full daily coverage and rainfall-state validation remain the
    responsibility of the artifact builder.
    """

    try:
        dataframe = pd.read_csv(processed_file)
    except Exception as exc:
        raise ValueError(
            f"Could not read processed rainfall file: {processed_file}"
        ) from exc

    required = {"date", "latitude", "longitude"}
    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"{processed_file.name} is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if dataframe.empty:
        raise ValueError(
            f"Processed rainfall file is empty: {processed_file}"
        )

    dates = pd.to_datetime(dataframe["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError(
            f"{processed_file.name} contains invalid dates."
        )

    if not (dates.dt.year == expected_year).all():
        raise ValueError(
            f"{processed_file.name} contains dates outside "
            f"the requested year {expected_year}."
        )

    latitude_values = pd.to_numeric(
        dataframe["latitude"],
        errors="coerce",
    )
    longitude_values = pd.to_numeric(
        dataframe["longitude"],
        errors="coerce",
    )

    if (
        not latitude_values.map(math.isfinite).all()
        or not longitude_values.map(math.isfinite).all()
    ):
        raise ValueError(
            f"{processed_file.name} contains invalid coordinates."
        )

    grids = set(
        zip(
            latitude_values.astype(float),
            longitude_values.astype(float),
        )
    )

    if len(grids) != 1:
        raise ValueError(
            f"{processed_file.name} contains inconsistent climate grids: "
            f"{sorted(grids)}"
        )

    latitude, longitude = next(iter(grids))

    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError(
            "Preprocessing returned coordinates outside "
            "valid geographic bounds."
        )

    return latitude, longitude



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

    location_id, latitude, longitude = (
        _validate_location_and_coordinates(
            location_id,
            latitude,
            longitude,
        )
    )

    if (
        isinstance(start_year, bool)
        or isinstance(end_year, bool)
        or not isinstance(start_year, int)
        or not isinstance(end_year, int)
    ):
        raise ValueError("start_year and end_year must be integers.")

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

        csv_grid = _get_single_csv_grid(
            processed_file,
            expected_year=year,
        )

        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError(
                f"Preprocessing returned no valid data for year {year}."
            )

        returned_grid = _get_single_csv_grid_from_dataframe(
            dataframe,
            year,
        )

        if returned_grid != csv_grid:
            raise ValueError(
                f"Preprocessing return data and generated CSV disagree "
                f"on the climate grid for year {year}."
            )

        resolved_grids.add(csv_grid)

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

    artifact_hash = artifact.content_hash()

    try:
        registry_data = json.loads(
            registry_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid calibration registry JSON: {registry_path}"
        ) from exc

    if registry_data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
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
        "artifact_hash": artifact_hash,
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