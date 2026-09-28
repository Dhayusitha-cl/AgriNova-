from unittest.mock import Mock

import pytest

from src.calibration_artifact import CalibrationArtifact
from src.location import ClimateGridLocation, GeographicLocation
from src.runtime_calibration import resolve_calibration_artifact


def test_resolves_location_to_registered_calibration_artifact(
    monkeypatch,
):
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    climate_dataset = Mock()

    expected_grid = ClimateGridLocation(
        source="imd_gridded_rainfall",
        latitude=20.50,
        longitude=78.25,
        key="imd_gridded_rainfall:20.50:78.25",
    )

    expected_artifact = Mock(spec=CalibrationArtifact)

    monkeypatch.setattr(
        "src.runtime_calibration.resolve_imd_climate_grid",
        lambda dataset, requested_location: expected_grid,
    )

    calls = {}

    def fake_get_artifact(
        climate_source,
        climate_grid_key,
    ):
        calls["climate_source"] = climate_source
        calls["climate_grid_key"] = climate_grid_key
        return expected_artifact

    monkeypatch.setattr(
        "src.runtime_calibration.get_calibration_artifact_for_grid",
        fake_get_artifact,
    )

    result = resolve_calibration_artifact(
        location,
        climate_dataset,
    )

    assert result is expected_artifact

    assert calls == {
        "climate_source": "imd_gridded_rainfall",
        "climate_grid_key": (
            "imd_gridded_rainfall:20.50:78.25"
        ),
    }


def test_rejects_invalid_location_type():
    with pytest.raises(TypeError):
        resolve_calibration_artifact(
            location="yavatmal",
            climate_dataset=Mock(),
        )


def test_grid_resolution_error_is_not_hidden(
    monkeypatch,
):
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    climate_dataset = Mock()

    def failing_resolution(dataset, requested_location):
        raise ValueError("Unable to resolve climate grid.")

    monkeypatch.setattr(
        "src.runtime_calibration.resolve_imd_climate_grid",
        failing_resolution,
    )

    with pytest.raises(
        ValueError,
        match="Unable to resolve climate grid",
    ):
        resolve_calibration_artifact(
            location,
            climate_dataset,
        )


def test_registry_resolution_error_is_not_hidden(
    monkeypatch,
):
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    climate_dataset = Mock()

    climate_grid = ClimateGridLocation(
        source="imd_gridded_rainfall",
        latitude=20.50,
        longitude=78.25,
        key="imd_gridded_rainfall:20.50:78.25",
    )

    monkeypatch.setattr(
        "src.runtime_calibration.resolve_imd_climate_grid",
        lambda dataset, requested_location: climate_grid,
    )

    monkeypatch.setattr(
        "src.runtime_calibration.get_calibration_artifact_for_grid",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("No calibration artifact is configured.")
        ),
    )

    with pytest.raises(
        ValueError,
        match="No calibration artifact is configured",
    ):
        resolve_calibration_artifact(
            location,
            climate_dataset,
        )

def test_resolves_real_yavatmal_calibration_artifact():
    import xarray as xr

    from src.runtime_calibration import resolve_calibration_artifact

    location = GeographicLocation(
        latitude=20.50,
        longitude=78.25,
    )

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

    artifact = resolve_calibration_artifact(
        location,
        dataset,
    )

    assert artifact.location == "yavatmal"
    assert artifact.content_hash() == (
        "7211729434149487a1913e1bfc2ebe77d66b93fa7fdbefb13dc0a74dec63ac4d"
    )

def test_resolves_calibration_artifact_from_provider():
    import xarray as xr

    from src.climate_data_provider import ClimateDataProvider
    from src.runtime_calibration import (
        resolve_calibration_artifact_from_provider,
    )

    class FakeProvider(ClimateDataProvider):
        def get_dataset(self):
            return xr.Dataset(
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

    location = GeographicLocation(
        latitude=20.50,
        longitude=78.25,
    )

    artifact = resolve_calibration_artifact_from_provider(
        location,
        FakeProvider(),
    )

    assert artifact.location == "yavatmal"
    assert artifact.content_hash() == (
        "7211729434149487a1913e1bfc2ebe77d66b93fa7fdbefb13dc0a74dec63ac4d"
    )