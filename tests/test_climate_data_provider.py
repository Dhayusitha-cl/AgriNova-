import pytest
import xarray as xr

from src.climate_data_provider import (
    ClimateDataProvider,
    IMDClimateDataProvider,
    create_configured_climate_data_provider,
)
from src.runtime_config import IMD_RAINFALL_DATASET_ENV


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

def test_configured_provider_uses_runtime_configuration(
    monkeypatch,
    tmp_path,
):
    dataset_path = tmp_path / "rainfall.nc"

    dataset = xr.Dataset(
        {
            "RAINFALL": (
                ("TIME", "LATITUDE", "LONGITUDE"),
                [[[10.0]]],
            )
        },
        coords={
            "TIME": ["2024-01-01"],
            "LATITUDE": [20.50],
            "LONGITUDE": [78.25],
        },
    )

    dataset.to_netcdf(dataset_path)
    dataset.close()

    monkeypatch.setenv(
        IMD_RAINFALL_DATASET_ENV,
        str(dataset_path),
    )

    provider = create_configured_climate_data_provider()

    assert isinstance(provider, IMDClimateDataProvider)

    opened_dataset = provider.get_dataset()

    try:
        assert "RAINFALL" in opened_dataset
        assert opened_dataset.sizes["TIME"] == 1
        assert opened_dataset.sizes["LATITUDE"] == 1
        assert opened_dataset.sizes["LONGITUDE"] == 1
    finally:
        opened_dataset.close()

def test_configured_provider_requires_runtime_configuration(
    monkeypatch,
):
    monkeypatch.delenv(
        IMD_RAINFALL_DATASET_ENV,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="is not configured",
    ):
        create_configured_climate_data_provider()