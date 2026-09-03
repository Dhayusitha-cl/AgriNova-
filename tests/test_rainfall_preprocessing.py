import pandas as pd
import pytest

from src.rainfall_preprocessing import (
    classify_rainfall,
    validate_rainfall,
    validate_daily_rainfall_observations,
    add_rainfall_state,
)


def test_classify_rainfall():
    assert classify_rainfall(0) == "dry"
    assert classify_rainfall(0.5) == "dry"
    assert classify_rainfall(1.0) == "drizzle"
    assert classify_rainfall(9.9) == "drizzle"
    assert classify_rainfall(10.0) == "rain"
    assert classify_rainfall(50.0) == "rain"


def test_classify_missing_rainfall():
    assert classify_rainfall(float("nan")) == "missing"


def test_validate_rejects_negative_rainfall():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "rainfall_mm": [-1.0],
        }
    )

    with pytest.raises(ValueError):
        validate_rainfall(dataframe)


def test_validate_sorts_dates():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-01"]
            ),
            "rainfall_mm": [5.0, 0.0],
        }
    )

    result = validate_rainfall(dataframe)

    assert result.iloc[0]["date"] == pd.Timestamp("2024-01-01")
    assert result.iloc[1]["date"] == pd.Timestamp("2024-01-02")


def test_add_rainfall_state():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "rainfall_mm": [0.0, 5.0, 20.0],
        }
    )

    result = add_rainfall_state(dataframe)

    assert list(result["rainfall_state"]) == [
        "dry",
        "drizzle",
        "rain",
    ]


def test_validate_daily_observations_accepts_valid_data():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-01"]
            ),
            "rainfall_mm": [5.0, 0.0],
            "rainfall_state": ["drizzle", "dry"],
        }
    )

    result = validate_daily_rainfall_observations(
        dataframe
    )

    assert list(result["date"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]


def test_validate_daily_observations_rejects_duplicate_dates():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01"]
            ),
            "rainfall_mm": [0.0, 5.0],
            "rainfall_state": ["dry", "drizzle"],
        }
    )

    with pytest.raises(ValueError):
        validate_daily_rainfall_observations(dataframe)


def test_validate_daily_observations_rejects_missing_rainfall():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "rainfall_mm": [float("nan")],
            "rainfall_state": ["dry"],
        }
    )

    with pytest.raises(ValueError):
        validate_daily_rainfall_observations(dataframe)


def test_validate_daily_observations_rejects_invalid_state():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "rainfall_mm": [5.0],
            "rainfall_state": ["heavy_rain"],
        }
    )

    with pytest.raises(ValueError):
        validate_daily_rainfall_observations(dataframe)


def test_validate_daily_observations_rejects_state_mismatch():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "rainfall_mm": [5.0],
            "rainfall_state": ["rain"],
        }
    )

    with pytest.raises(ValueError):
        validate_daily_rainfall_observations(dataframe)