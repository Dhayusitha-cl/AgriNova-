"""
Offline calibration onboarding workflow for CropLogic-Saathi.

This module orchestrates existing preprocessing and calibration functions.
It does not perform runtime decision making or automatically register
calibration artifacts for production use.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.build_calibration_artifact import (
    build_and_save_calibration_artifact,
)
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