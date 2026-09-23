import json

import pytest

from src.calibration_registry import (
    CALIBRATION_ARTIFACTS,
    get_calibration_artifact,
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
        tmp_path / "missing.json",
    )

    with pytest.raises(FileNotFoundError, match="Calibration artifact not found"):
        get_calibration_artifact("missing")


def test_artifact_location_mismatch_is_rejected(monkeypatch, tmp_path):
    artifact_path = tmp_path / "artifact.json"

    source = CALIBRATION_ARTIFACTS["yavatmal"]
    artifact_data = json.loads(source.read_text(encoding="utf-8"))
    artifact_data["location"] = "other_location"

    artifact_path.write_text(
        json.dumps(artifact_data),
        encoding="utf-8",
    )

    monkeypatch.setitem(
        CALIBRATION_ARTIFACTS,
        "yavatmal",
        artifact_path,
    )

    with pytest.raises(ValueError, match="location mismatch"):
        get_calibration_artifact("yavatmal")
