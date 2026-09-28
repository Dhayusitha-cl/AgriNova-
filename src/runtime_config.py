"""
Runtime configuration for CropLogic-Saathi.

Configuration is supplied by the deployment environment rather than
hard-coded into API or decision-engine code.
"""

import os
from pathlib import Path


IMD_RAINFALL_DATASET_ENV = "CROPLOGIC_IMD_RAINFALL_DATASET"


def get_imd_rainfall_dataset_path() -> Path:
    """Return the configured IMD rainfall dataset path.

    Raises
    ------
    ValueError
        If the environment variable is missing or empty.
    FileNotFoundError
        If the configured path does not exist.
    """
    configured_path = os.getenv(IMD_RAINFALL_DATASET_ENV)

    if configured_path is None or not configured_path.strip():
        raise ValueError(
            f"{IMD_RAINFALL_DATASET_ENV} is not configured."
        )

    path = Path(configured_path.strip())

    if not path.exists():
        raise FileNotFoundError(
            f"Configured IMD rainfall dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Configured IMD rainfall dataset path is not a file: {path}"
        )

    return path