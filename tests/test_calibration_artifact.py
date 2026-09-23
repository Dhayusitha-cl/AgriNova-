import numpy as np
import pandas as pd
import pytest

from src.calibration_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    CalibrationArtifact,
    RAINFALL_STATES,
)

from src.build_calibration_artifact import (
    build_calibration_artifact,
)

from src.monte_carlo_weather import (
    generate_calibrated_monte_carlo_scenarios,
)


def make_test_artifact():
    matrix = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ],
        dtype=float,
    )

    monthly_matrices = {
        month: matrix.copy()
        for month in range(1, 13)
    }

    rainfall_samples = {
        (month, state): np.array(
            [0.0, 0.5, 1.0],
            dtype=float,
        )
        for month in range(1, 13)
        for state in RAINFALL_STATES
    }

    return CalibrationArtifact(
        location="test_location",
        source_files=[
            "rainfall_test_2020.csv",
            "rainfall_test_2021.csv",
        ],
        source_start_date="2020-06-01",
        source_end_date="2021-09-30",
        monthly_transition_matrices=monthly_matrices,
        rainfall_samples=rainfall_samples,
        fallback_transition_matrix=matrix.copy(),
    )


def test_valid_artifact_passes_validation():
    artifact = make_test_artifact()

    artifact.validate()


def test_artifact_round_trip(tmp_path):
    artifact = make_test_artifact()

    path = tmp_path / "calibration.json"

    artifact.save(path)

    loaded = CalibrationArtifact.load(path)

    assert loaded.location == artifact.location
    assert loaded.source_files == artifact.source_files
    assert loaded.source_start_date == artifact.source_start_date
    assert loaded.source_end_date == artifact.source_end_date

    np.testing.assert_allclose(
        loaded.fallback_transition_matrix,
        artifact.fallback_transition_matrix,
    )

    for month in range(1, 13):
        np.testing.assert_allclose(
            loaded.monthly_transition_matrices[month],
            artifact.monthly_transition_matrices[month],
        )

        for state in RAINFALL_STATES:
            np.testing.assert_allclose(
                loaded.rainfall_samples[(month, state)],
                artifact.rainfall_samples[(month, state)],
            )


def test_serialized_artifact_contains_schema_metadata():
    artifact = make_test_artifact()

    data = artifact.to_dict()

    assert data["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert data["artifact_type"] == "rainfall_calibration"
    assert data["rainfall_states"] == list(RAINFALL_STATES)


def test_missing_month_is_rejected():
    artifact = make_test_artifact()

    del artifact.monthly_transition_matrices[12]

    with pytest.raises(ValueError, match="months 1 through 12"):
        artifact.validate()


def test_invalid_transition_matrix_shape_is_rejected():
    artifact = make_test_artifact()

    artifact.fallback_transition_matrix = np.ones(
        (2, 2),
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="shape \\(3, 3\\)",
    ):
        artifact.validate()


def test_transition_matrix_with_invalid_row_sum_is_rejected():
    artifact = make_test_artifact()

    artifact.fallback_transition_matrix = np.array(
        [
            [0.5, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="rows must sum approximately to 1",
    ):
        artifact.validate()


def test_missing_rainfall_samples_are_rejected():
    artifact = make_test_artifact()

    del artifact.rainfall_samples[(6, "rain")]

    with pytest.raises(
        ValueError,
        match="Missing rainfall samples",
    ):
        artifact.validate()


def test_negative_rainfall_sample_is_rejected():
    artifact = make_test_artifact()

    artifact.rainfall_samples[(6, "rain")] = np.array(
        [-1.0, 2.0],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="negative rainfall",
    ):
        artifact.validate()


def test_unknown_schema_version_is_rejected():
    artifact = make_test_artifact()

    data = artifact.to_dict()
    data["schema_version"] = "999.0"

    with pytest.raises(
        ValueError,
        match="Unsupported calibration artifact schema version",
    ):
        CalibrationArtifact.from_dict(data)


def test_missing_artifact_file_is_rejected(tmp_path):
    path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        CalibrationArtifact.load(path)

def make_processed_rainfall_files(tmp_path):
    rows = []

    state_values = {
        "dry": 0.0,
        "drizzle": 5.0,
        "rain": 20.0,
    }

    for year in (2020, 2021):
        dates = pd.date_range(
            f"{year}-01-01",
            periods=366 if year % 4 == 0 else 365,
            freq="D",
        )

        states = (
            ["dry", "drizzle", "rain"]
            * ((len(dates) // 3) + 1)
        )[:len(dates)]

        for date, state in zip(dates, states):
            rows.append(
                {
                    "date": date,
                    "rainfall_mm": state_values[state],
                    "rainfall_state": state,
                }
            )

    dataframe = pd.DataFrame(rows)

    dataframe[
        dataframe["date"].dt.year == 2020
    ].to_csv(
        tmp_path / "rainfall_test_2020.csv",
        index=False,
    )

    dataframe[
        dataframe["date"].dt.year == 2021
    ].to_csv(
        tmp_path / "rainfall_test_2021.csv",
        index=False,
    )

    return tmp_path


def test_build_calibration_artifact_from_processed_files(tmp_path):
    data_dir = make_processed_rainfall_files(tmp_path)

    artifact = build_calibration_artifact(
        data_dir=data_dir,
        pattern="rainfall_test_*.csv",
        location="test_location",
    )

    assert artifact.location == "test_location"

    assert artifact.source_files == [
        "rainfall_test_2020.csv",
        "rainfall_test_2021.csv",
    ]

    assert artifact.source_start_date == "2020-01-01"
    assert artifact.source_end_date == "2021-12-31"

    assert set(
        artifact.monthly_transition_matrices
    ) == set(range(1, 13))

    for month in range(1, 13):
        matrix = artifact.monthly_transition_matrices[month]

        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(
            matrix.sum(axis=1),
            np.ones(3),
        )

    for month in range(1, 13):
        for state in RAINFALL_STATES:
            values = artifact.rainfall_samples[
                (month, state)
            ]

            assert len(values) > 0
            assert np.all(values >= 0)


def test_build_calibration_artifact_can_be_saved(tmp_path):
    data_dir = make_processed_rainfall_files(tmp_path)

    artifact = build_calibration_artifact(
        data_dir=data_dir,
        pattern="rainfall_test_*.csv",
    )

    output_path = tmp_path / "calibration.json"

    artifact.save(output_path)

    loaded = CalibrationArtifact.load(output_path)

    assert loaded.location == artifact.location
    assert loaded.source_files == artifact.source_files

    np.testing.assert_allclose(
        loaded.fallback_transition_matrix,
        artifact.fallback_transition_matrix,
    )


def test_monte_carlo_can_consume_calibration_artifact(tmp_path):
    data_dir = make_processed_rainfall_files(tmp_path)

    artifact = build_calibration_artifact(
        data_dir=data_dir,
        pattern="rainfall_test_*.csv",
        location="test_location",
    )

    scenarios = generate_calibrated_monte_carlo_scenarios(
        start_date="2021-06-15",
        num_days=7,
        num_simulations=10,
        initial_state="dry",
        random_seed=42,
        calibration_artifact=artifact,
    )

    assert len(scenarios) == 10
    assert all(len(scenario) == 7 for scenario in scenarios)

    for scenario in scenarios:
        assert scenario[0]["rainfall_state"] == "dry"

        for day in scenario:
            assert day["rainfall_state"] in {
                "dry",
                "drizzle",
                "rain",
            }
            assert day["rainfall_mm"] >= 0

def test_monte_carlo_rejects_invalid_calibration_artifact():
    with pytest.raises(TypeError, match="calibration_artifact"):
        generate_calibrated_monte_carlo_scenarios(
            start_date="2021-06-15",
            num_days=7,
            num_simulations=10,
            calibration_artifact=object(),
        )
