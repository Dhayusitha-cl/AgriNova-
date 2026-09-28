from pathlib import Path

import pytest

from src.runtime_config import (
    IMD_RAINFALL_DATASET_ENV,
    get_imd_rainfall_dataset_path,
)


def test_missing_dataset_configuration(monkeypatch):
    monkeypatch.delenv(IMD_RAINFALL_DATASET_ENV, raising=False)

    with pytest.raises(
        ValueError,
        match="is not configured",
    ):
        get_imd_rainfall_dataset_path()


def test_empty_dataset_configuration(monkeypatch):
    monkeypatch.setenv(IMD_RAINFALL_DATASET_ENV, "   ")

    with pytest.raises(
        ValueError,
        match="is not configured",
    ):
        get_imd_rainfall_dataset_path()


def test_missing_configured_dataset(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.nc"

    monkeypatch.setenv(
        IMD_RAINFALL_DATASET_ENV,
        str(missing_path),
    )

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        get_imd_rainfall_dataset_path()


def test_configured_dataset_path(monkeypatch, tmp_path):
    dataset_path = tmp_path / "rainfall.nc"
    dataset_path.write_bytes(b"placeholder")

    monkeypatch.setenv(
        IMD_RAINFALL_DATASET_ENV,
        str(dataset_path),
    )

    result = get_imd_rainfall_dataset_path()

    assert result == Path(dataset_path)


def test_configured_directory_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(
        IMD_RAINFALL_DATASET_ENV,
        str(tmp_path),
    )

    with pytest.raises(
        ValueError,
        match="is not a file",
    ):
        get_imd_rainfall_dataset_path()