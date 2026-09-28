"""
Climate-data provider boundary for runtime geographic resolution.

This module separates:
    runtime decision logic
        -> climate-data acquisition
        -> authoritative climate dataset

The provider does not perform calibration or make decisions.
"""

from pathlib import Path

import xarray as xr


class ClimateDataProvider:
    """Interface for obtaining an authoritative climate dataset."""

    def get_dataset(self):
        """
        Return an open climate dataset.

        Implementations may obtain the dataset from a local cache,
        packaged data, object storage, or another authoritative source.
        """
        raise NotImplementedError


class IMDClimateDataProvider(ClimateDataProvider):
    """
    Provider for an IMD gridded rainfall NetCDF dataset.

    The dataset path is supplied by configuration rather than hard-coded.
    """

    def __init__(self, dataset_path: str | Path):
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"IMD rainfall dataset not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"IMD rainfall dataset path is not a file: {path}"
            )

        self._dataset_path = path

    def get_dataset(self):
        """Open and return the configured IMD rainfall dataset."""

        return xr.open_dataset(self._dataset_path)