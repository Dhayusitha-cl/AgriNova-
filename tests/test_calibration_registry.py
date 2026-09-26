import json

import pytest

from src.location import GeographicLocation, resolve_climate_grid
from src.calibration_registry import (
    CALIBRATION_ARTIFACTS,
    CalibrationRegistryEntry,
    get_calibration_artifact,
    get_calibration_artifact_for_grid,
)


def test_yavatmal_artifact_is_registered():
    assert "yavatmal" in CALIBRATION_ARTIFACTS


def test_load_yavatmal_artifact():
    artifact = get_calibration_artifact("yavatmal")

    assert artifact.location == "yavatmal"
    assert artifact.source_start_date == "2019-01-01"
    assert artifact.source_end_date == "2024-12-31"


def test_location_is_case_insensitive():
    artifact = get_calibration_artifact("YAVATMAL")

    assert artifact.location == "yavatmal"


def test_unknown_location_is_rejected():
    with pytest.raises(ValueError, match="No calibration artifact"):
        get_calibration_artifact("unknown_location")


def test_empty_location_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        get_calibration_artifact("")


def test_missing_artifact_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setitem(
        CALIBRATION_ARTIFACTS,
        "missing",
        CalibrationRegistryEntry(
            artifact_path=tmp_path / "missing.json",
            climate_source="test_source",
            climate_grid_key="test_source:test",
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Calibration artifact not found",
    ):
        get_calibration_artifact("missing")

def test_artifact_location_mismatch_is_rejected(monkeypatch, tmp_path):
    artifact_path = tmp_path / "artifact.json"

    source = CALIBRATION_ARTIFACTS["yavatmal"].artifact_path
    artifact_data = json.loads(source.read_text(encoding="utf-8"))
    artifact_data["location"] = "other_location"

    artifact_path.write_text(
        json.dumps(artifact_data),
        encoding="utf-8",
    )

    monkeypatch.setitem(
        CALIBRATION_ARTIFACTS,
        "yavatmal",
        CalibrationRegistryEntry(
            artifact_path=artifact_path,
            climate_source="imd_gridded_rainfall",
            climate_grid_key="imd_gridded_rainfall:20.50:78.25",
        ),
    )

    with pytest.raises(ValueError, match="location mismatch"):
        get_calibration_artifact("yavatmal")

def test_yavatmal_registry_contains_climate_metadata():
    entry = CALIBRATION_ARTIFACTS["yavatmal"]

    assert entry.climate_source == "imd_gridded_rainfall"
    assert entry.climate_grid_key == "imd_gridded_rainfall:20.50:78.25"


def test_climate_grid_identity_is_separate_from_artifact_location():
    entry = CALIBRATION_ARTIFACTS["yavatmal"]
    artifact = get_calibration_artifact("yavatmal")

    assert artifact.location == "yavatmal"
    assert entry.climate_grid_key != artifact.location


def test_load_artifact_for_climate_grid():
    artifact = get_calibration_artifact_for_grid(
        "imd_gridded_rainfall",
        "imd_gridded_rainfall:20.50:78.25",
    )

    assert artifact.location == "yavatmal"


def test_climate_grid_lookup_is_case_insensitive():
    artifact = get_calibration_artifact_for_grid(
        "IMD_GRIDDED_RAINFALL",
        "IMD_GRIDDED_RAINFALL:20.50:78.25",
    )

    assert artifact.location == "yavatmal"


def test_unknown_climate_grid_is_rejected():
    with pytest.raises(
        ValueError,
        match="No calibration artifact is configured for climate grid",
    ):
        get_calibration_artifact_for_grid(
            "imd_gridded_rainfall",
            "imd_gridded_rainfall:20.75:78.00",
        )


def test_wrong_climate_source_is_rejected():
    with pytest.raises(
        ValueError,
        match="No calibration artifact is configured for climate grid",
    ):
        get_calibration_artifact_for_grid(
            "other_source",
            "imd_gridded_rainfall:20.50:78.00",
        )


@pytest.mark.parametrize(
    "climate_source, climate_grid_key",
    [
        ("", "imd_gridded_rainfall:20.50:78.00"),
        ("imd_gridded_rainfall", ""),
    ],
)
def test_climate_grid_lookup_rejects_empty_inputs(
    climate_source,
    climate_grid_key,
):
    with pytest.raises(ValueError, match="must not be empty"):
        get_calibration_artifact_for_grid(
            climate_source,
            climate_grid_key,
        )


def test_geographic_location_resolves_to_registered_calibration():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.13,
    )

    climate_grid = resolve_climate_grid(
        location,
        [20.0, 20.25, 20.5, 20.75],
        [77.75, 78.0, 78.25, 78.5],
    )

    artifact = get_calibration_artifact_for_grid(
        climate_grid.source,
        climate_grid.key,
    )

    assert climate_grid.key == "imd_gridded_rainfall:20.50:78.25"
    assert artifact.location == "yavatmal"


def test_unregistered_geographic_grid_does_not_fallback():
    location = GeographicLocation(
        latitude=20.39,
        longitude=78.12,
    )

    climate_grid = resolve_climate_grid(
        location,
        [20.75],
        [78.25],
    )

    with pytest.raises(
        ValueError,
        match="No calibration artifact is configured for climate grid",
    ):
        get_calibration_artifact_for_grid(
            climate_grid.source,
            climate_grid.key,
        )