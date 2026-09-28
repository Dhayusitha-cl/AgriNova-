import pytest
import xarray as xr

from src.climate_data_provider import (
    ClimateDataProvider,
    IMDClimateDataProvider,
)


def test_imd_provider_rejects_missing_dataset():
    with pytest.raises(FileNotFoundError):
        IMDClimateDataProvider(
            "does-not-exist.nc"
        )


def test_imd_provider_opens_dataset(tmp_path):
    dataset_path = tmp_path / "rainfall.nc"

    dataset = xr.Dataset(
        data_vars={
            "RAINFALL": (
                ("TIME", "LATITUDE", "LONGITUDE"),
                [[[0.0]]],
            ),
        },
        coords={
            "TIME": ["2024-01-01"],
            "LATITUDE": [20.50],
            "LONGITUDE": [78.25],
        },
    )

    dataset.to_netcdf(dataset_path)

    provider = IMDClimateDataProvider(dataset_path)

    opened = provider.get_dataset()

    try:
        assert "RAINFALL" in opened.data_vars
        assert "LATITUDE" in opened.coords
        assert "LONGITUDE" in opened.coords
    finally:
        opened.close()


def test_provider_contract_is_explicit():
    provider = ClimateDataProvider()

    with pytest.raises(NotImplementedError):
        provider.get_dataset()