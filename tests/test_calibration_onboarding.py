import json
from pathlib import Path

import pandas as pd
import pytest

import src.calibration_onboarding as onboarding
from src.calibration_onboarding import register_calibration


def _write_raw_file(raw_dir: Path, year: int) -> None:
    """Create a placeholder raw IMD file for the mocked preprocessing path."""
    (raw_dir / f"RF25_ind{year}_rfp25.nc").write_text(
        "placeholder",
        encoding="utf-8",
    )


def _processed_dataframe(
    latitude: float = 20.50,
    longitude: float = 78.25,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-01"]),
            "rainfall_mm": [10.0],
            "latitude": [latitude],
            "longitude": [longitude],
            "rainfall_state": ["rain"],
        }
    )


def test_onboard_calibration_builds_artifact_and_returns_metadata(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = tmp_path / "calibration.json"

    raw_dir.mkdir()
    processed_dir.mkdir()

    _write_raw_file(raw_dir, 2023)
    _write_raw_file(raw_dir, 2024)

    processed_calls = []

    def fake_process_year(
        year,
        *,
        latitude,
        longitude,
        location_id,
        raw_dir,
        processed_dir,
    ):
        processed_calls.append(year)

        dataframe = _processed_dataframe()
        dataframe.to_csv(
            processed_dir / f"rainfall_{location_id}_{year}.csv",
            index=False,
        )
        return dataframe

    class FakeArtifact:
        location = "test_location"

    captured = {}

    def fake_build_and_save(
        *,
        output_path,
        data_dir,
        pattern,
        location,
    ):
        captured["output_path"] = output_path
        captured["data_dir"] = data_dir
        captured["pattern"] = pattern
        captured["location"] = location

        captured["staged_files"] = sorted(
            path.name
            for path in Path(data_dir).glob(pattern)
        )

        return FakeArtifact()

    monkeypatch.setattr(
        onboarding,
        "process_year",
        fake_process_year,
    )
    monkeypatch.setattr(
        onboarding,
        "build_and_save_calibration_artifact",
        fake_build_and_save,
    )

    result = onboarding.onboard_calibration(
        location_id="test_location",
        latitude=20.39,
        longitude=78.13,
        start_year=2023,
        end_year=2024,
        output_path=output_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert processed_calls == [2023, 2024]

    assert result["artifact"].location == "test_location"
    assert result["artifact_path"] == output_path
    assert result["processed_files"] == [
        processed_dir / "rainfall_test_location_2023.csv",
        processed_dir / "rainfall_test_location_2024.csv",
    ]

    assert result["climate_source"] == "imd_gridded_rainfall"
    assert result["climate_grid_key"] == (
        "imd_gridded_rainfall:20.50:78.25"
    )
    assert result["selected_latitude"] == 20.50
    assert result["selected_longitude"] == 78.25

    assert captured["output_path"] == output_path
    assert captured["data_dir"] != processed_dir
    assert captured["pattern"] == "rainfall_test_location_*.csv"
    assert captured["staged_files"] == [
        "rainfall_test_location_2023.csv",
        "rainfall_test_location_2024.csv",
    ]
    assert captured["location"] == "test_location"


def test_onboard_calibration_rejects_invalid_year_range(tmp_path):
    with pytest.raises(
        ValueError,
        match="start_year must not be greater than end_year",
    ):
        onboarding.onboard_calibration(
            location_id="test_location",
            latitude=20.39,
            longitude=78.13,
            start_year=2024,
            end_year=2023,
            output_path=tmp_path / "calibration.json",
        )


def test_onboard_calibration_rejects_inconsistent_climate_grids(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = tmp_path / "calibration.json"

    raw_dir.mkdir()
    processed_dir.mkdir()

    grids = {
        2023: (20.50, 78.25),
        2024: (20.75, 78.25),
    }

    def fake_process_year(
        year,
        *,
        latitude,
        longitude,
        location_id,
        raw_dir,
        processed_dir,
    ):
        grid_latitude, grid_longitude = grids[year]
        dataframe = _processed_dataframe(
            grid_latitude,
            grid_longitude,
        )

        dataframe.to_csv(
            processed_dir / f"rainfall_{location_id}_{year}.csv",
            index=False,
        )

        return dataframe

    def fail_if_called(**kwargs):
        pytest.fail(
            "Calibration artifact builder must not run "
            "when climate grids are inconsistent."
        )

    monkeypatch.setattr(
        onboarding,
        "process_year",
        fake_process_year,
    )
    monkeypatch.setattr(
        onboarding,
        "build_and_save_calibration_artifact",
        fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match="inconsistent climate grids",
    ):
        onboarding.onboard_calibration(
            location_id="test_location",
            latitude=20.39,
            longitude=78.13,
            start_year=2023,
            end_year=2024,
            output_path=output_path,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )


def test_onboard_calibration_does_not_import_registry(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = tmp_path / "calibration.json"

    raw_dir.mkdir()
    processed_dir.mkdir()

    def fake_process_year(
        year,
        *,
        latitude,
        longitude,
        location_id,
        raw_dir,
        processed_dir,
    ):
        dataframe = _processed_dataframe()

        dataframe.to_csv(
            processed_dir / f"rainfall_{location_id}_{year}.csv",
            index=False,
        )

        return dataframe

    class FakeArtifact:
        location = "new_location"

    monkeypatch.setattr(
        onboarding,
        "process_year",
        fake_process_year,
    )
    monkeypatch.setattr(
        onboarding,
        "build_and_save_calibration_artifact",
        lambda **kwargs: FakeArtifact(),
    )

    result = onboarding.onboard_calibration(
        location_id="new_location",
        latitude=20.39,
        longitude=78.13,
        start_year=2024,
        end_year=2024,
        output_path=output_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert result["artifact"].location == "new_location"
    assert not hasattr(onboarding, "CALIBRATION_ARTIFACTS")

def test_onboard_calibration_does_not_include_processed_years_outside_range(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_path = tmp_path / "calibration.json"

    raw_dir.mkdir()
    processed_dir.mkdir()

    # An older already-processed year that must NOT enter this onboarding run.
    existing_file = (
        processed_dir / "rainfall_test_location_2022.csv"
    )
    existing_file.write_text(
        "date,rainfall_mm,latitude,longitude,rainfall_state\n"
        "2022-06-01,10.0,20.50,78.25,rain\n",
        encoding="utf-8",
    )

    def fake_process_year(
        year,
        *,
        latitude,
        longitude,
        location_id,
        raw_dir,
        processed_dir,
    ):
        dataframe = _processed_dataframe()

        dataframe.to_csv(
            processed_dir / f"rainfall_{location_id}_{year}.csv",
            index=False,
        )

        return dataframe

    class FakeArtifact:
        location = "test_location"

    captured = {}

    def fake_build_and_save(
        *,
        output_path,
        data_dir,
        pattern,
        location,
    ):
        captured["data_dir"] = data_dir
        captured["pattern"] = pattern
        captured["files"] = sorted(
            path.name
            for path in Path(data_dir).glob(pattern)
        )
        return FakeArtifact()

    monkeypatch.setattr(
        onboarding,
        "process_year",
        fake_process_year,
    )
    monkeypatch.setattr(
        onboarding,
        "build_and_save_calibration_artifact",
        fake_build_and_save,
    )

    onboarding.onboard_calibration(
        location_id="test_location",
        latitude=20.39,
        longitude=78.13,
        start_year=2023,
        end_year=2024,
        output_path=output_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    # This test intentionally documents the requirement:
    # only 2023 and 2024 should be supplied to calibration.
    assert captured["data_dir"] != processed_dir
    assert captured["pattern"] == "rainfall_test_location_*.csv"
    assert captured["files"] == [
        "rainfall_test_location_2023.csv",
        "rainfall_test_location_2024.csv",
    ]

def test_register_calibration_adds_new_location(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()

    registry_path = calibration_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    artifact_source = (
        Path("data/calibration/yavatmal_rainfall_calibration_v1.json")
    )

    artifact_data = json.loads(
        artifact_source.read_text(encoding="utf-8")
    )
    artifact_data["location"] = "new_location"

    artifact_path = calibration_dir / "new_location_v1.json"
    artifact_path.write_text(
        json.dumps(artifact_data),
        encoding="utf-8",
    )

    register_calibration(
        registry_path=registry_path,
        location_id="new_location",
        artifact_path=artifact_path,
        climate_source="imd_gridded_rainfall",
        climate_grid_key="imd_gridded_rainfall:20.50:78.25",
    )

    registry = json.loads(
        registry_path.read_text(encoding="utf-8")
    )

    assert registry["entries"]["new_location"] == {
        "artifact_path": "new_location_v1.json",
        "climate_source": "imd_gridded_rainfall",
        "climate_grid_key": "imd_gridded_rainfall:20.50:78.25",
    }


def test_register_calibration_rejects_location_mismatch(tmp_path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()

    registry_path = calibration_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    artifact_source = (
        Path("data/calibration/yavatmal_rainfall_calibration_v1.json")
    )

    artifact_path = calibration_dir / "artifact.json"
    artifact_path.write_text(
        artifact_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="location mismatch"):
        register_calibration(
            registry_path=registry_path,
            location_id="different_location",
            artifact_path=artifact_path,
            climate_source="imd_gridded_rainfall",
            climate_grid_key="imd_gridded_rainfall:20.50:78.25",
        )


def test_register_calibration_rejects_duplicate_without_overwrite(
    tmp_path,
):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()

    registry_path = calibration_dir / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": {
                    "new_location": {
                        "artifact_path": "existing.json",
                        "climate_source": "imd_gridded_rainfall",
                        "climate_grid_key": (
                            "imd_gridded_rainfall:20.50:78.25"
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    artifact_source = (
        Path("data/calibration/yavatmal_rainfall_calibration_v1.json")
    )

    artifact_data = json.loads(
        artifact_source.read_text(encoding="utf-8")
    )
    artifact_data["location"] = "new_location"

    artifact_path = calibration_dir / "new_location_v2.json"
    artifact_path.write_text(
        json.dumps(artifact_data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already registered"):
        register_calibration(
            registry_path=registry_path,
            location_id="new_location",
            artifact_path=artifact_path,
            climate_source="imd_gridded_rainfall",
            climate_grid_key="imd_gridded_rainfall:20.50:78.25",
        )

def test_registration_does_not_mutate_runtime_registry():
    from src.calibration_registry import CALIBRATION_ARTIFACTS

    assert "yavatmal" in CALIBRATION_ARTIFACTS