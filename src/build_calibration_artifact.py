"""
Build a versioned rainfall calibration artifact from processed data.

This is an offline operation.

It reuses the existing calibration logic and does not perform
Monte Carlo simulation or decision making.
"""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import pandas as pd

from src.calibration_artifact import (
    CALIBRATION_METHOD_VERSION,
    PREPROCESSING_VERSION,
    SOURCE_DATASET,
    CalibrationArtifact,
    RAINFALL_STATES,
)

from src.markov_calibration import (
    calculate_transition_matrix,
    get_monthly_transition_matrix_with_fallback,
)
from src.rainfall_amount_calibration import load_processed_rainfall


DEFAULT_DATA_DIR = "data/processed"


def build_calibration_artifact(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    pattern: str | None = None,
    location: str = "yavatmal",
) -> CalibrationArtifact:
    """
    Build a calibration artifact from processed rainfall data.

    Calibration is performed once from the supplied historical dataset.
    """

    data_dir = Path(data_dir)

    if pattern is None:
        pattern = f"rainfall_{location}_*.csv"

    rainfall_data = load_processed_rainfall(
        data_dir=data_dir,
        pattern=pattern,
    )

    if rainfall_data.empty:
        raise ValueError(
            "Cannot build calibration artifact from empty rainfall data."
        )

    rainfall_data = rainfall_data.copy()

    rainfall_data["date"] = pd.to_datetime(
        rainfall_data["date"],
        errors="raise",
    )

    rainfall_data = rainfall_data.sort_values(
        "date"
    ).reset_index(drop=True)

    required_columns = {
        "date",
        "rainfall_mm",
        "rainfall_state",
    }

    missing_columns = required_columns.difference(
        rainfall_data.columns
    )

    if missing_columns:
        raise ValueError(
            "Rainfall data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    fallback_matrix = calculate_transition_matrix(
        rainfall_data
    )

    monthly_transition_matrices = {}

    for month in range(1, 13):
        monthly_transition_matrices[month] = (
            get_monthly_transition_matrix_with_fallback(
                rainfall_data,
                month=month,
                fallback_matrix=fallback_matrix,
            )
        )

    rainfall_samples = {}

    for month in range(1, 13):
        month_data = rainfall_data[
            rainfall_data["date"].dt.month == month
        ]

        for state in RAINFALL_STATES:
            values = month_data.loc[
                month_data["rainfall_state"] == state,
                "rainfall_mm",
            ].to_numpy(dtype=float)

            if len(values) == 0:
                values = rainfall_data.loc[
                    rainfall_data["rainfall_state"] == state,
                    "rainfall_mm",
                ].to_numpy(dtype=float)

            if len(values) == 0:
                raise ValueError(
                    f"No rainfall observations available for "
                    f"state={state!r}, month={month}."
                )

            rainfall_samples[(month, state)] = values

    source_files = sorted(
        path.name
        for path in data_dir.glob(pattern)
        if path.is_file()
    )

    if not source_files:
        raise ValueError(
            "No source rainfall files matched the supplied pattern."
        )

    artifact = CalibrationArtifact(
        location=location,
        source_dataset=SOURCE_DATASET,
        preprocessing_version=PREPROCESSING_VERSION,
        calibration_method_version=CALIBRATION_METHOD_VERSION,
        source_files=source_files,
        source_start_date=(
            rainfall_data["date"].min().date().isoformat()
        ),
        source_end_date=(
            rainfall_data["date"].max().date().isoformat()
        ),
        monthly_transition_matrices=monthly_transition_matrices,
        rainfall_samples=rainfall_samples,
        fallback_transition_matrix=fallback_matrix,
    )

    artifact.validate()

    return artifact


def build_and_save_calibration_artifact(
    output_path: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    pattern: str | None = None,
    location: str = "yavatmal",
) -> CalibrationArtifact:
    """
    Build, validate, and atomically publish a calibration artifact.

    The artifact is first written to a temporary file beside the final
    destination and loaded back for validation. The final path is replaced
    only after the serialized artifact has been successfully validated.
    """

    artifact = build_calibration_artifact(
        data_dir=data_dir,
        pattern=pattern,
        location=location,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        artifact.save(temporary_path)

        # Validate the serialized artifact before publication.
        validated_artifact = CalibrationArtifact.load(
            temporary_path
        )

        if (
            validated_artifact.content_hash()
            != artifact.content_hash()
        ):
            raise ValueError(
                "Serialized calibration artifact does not match "
                "the in-memory artifact."
            )

        os.replace(temporary_path, output_path)
        temporary_path = None

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return artifact
