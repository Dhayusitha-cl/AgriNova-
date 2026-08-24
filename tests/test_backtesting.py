import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    validate_rainfall_dataframe,
    get_training_data,
    get_actual_future_data,
    create_backtest_split,
    get_initial_state,
    calculate_actual_future_total,
    calculate_monte_carlo_total_distribution,
    evaluate_actual_against_distribution,
    calibrate_backtest_transition_matrix,
)


def make_test_rainfall_data():

    dates = pd.date_range(
        "2024-06-01",
        periods=40,
        freq="D",
    )

    states = [
        "dry",
        "drizzle",
        "rain",
        "rain",
        "drizzle",
        "dry",
        "rain",
        "drizzle",
    ]

    rainfall = [
        0,
        5,
        20,
        15,
        4,
        0,
        25,
        3,
    ]

    return pd.DataFrame(
        {
            "date": dates,
            "rainfall_mm": [
                rainfall[i % len(rainfall)]
                for i in range(len(dates))
            ],
            "rainfall_state": [
                states[i % len(states)]
                for i in range(len(dates))
            ],
        }
    )


def test_validate_rainfall_dataframe():

    dataframe = make_test_rainfall_data()

    result = validate_rainfall_dataframe(
        dataframe
    )

    assert list(result.columns) == [
        "date",
        "rainfall_mm",
        "rainfall_state",
    ]

    assert result["date"].is_monotonic_increasing


def test_validate_rainfall_dataframe_rejects_missing_columns():

    dataframe = pd.DataFrame(
        {
            "date": ["2024-06-01"],
            "rainfall_mm": [10],
        }
    )

    with pytest.raises(ValueError):
        validate_rainfall_dataframe(
            dataframe
        )


def test_training_data_excludes_decision_date():

    dataframe = make_test_rainfall_data()

    decision_date = "2024-06-20"

    training = get_training_data(
        dataframe,
        decision_date,
    )

    assert training["date"].max() < pd.Timestamp(
        decision_date
    )


def test_actual_future_contains_decision_date():

    dataframe = make_test_rainfall_data()

    decision_date = "2024-06-20"

    future = get_actual_future_data(
        dataframe,
        decision_date,
        horizon_days=5,
    )

    assert future["date"].min() == pd.Timestamp(
        decision_date
    )

    assert future["date"].max() == pd.Timestamp(
        "2024-06-24"
    )


def test_backtest_split_has_no_overlap():

    dataframe = make_test_rainfall_data()

    split = create_backtest_split(
        dataframe=dataframe,
        decision_date="2024-06-20",
        horizon_days=5,
    )

    training = split["training_data"]
    future = split["actual_future"]

    assert training["date"].max() < future["date"].min()


def test_initial_state_uses_previous_observation():

    dataframe = make_test_rainfall_data()

    initial_state = get_initial_state(
        dataframe,
        "2024-06-20",
    )

    expected = dataframe.loc[
        dataframe["date"]
        < pd.Timestamp("2024-06-20"),
        "rainfall_state",
    ].iloc[-1]

    assert initial_state == expected


def test_actual_future_total():

    dataframe = make_test_rainfall_data()

    future = get_actual_future_data(
        dataframe,
        "2024-06-20",
        horizon_days=5,
    )

    total = calculate_actual_future_total(
        future
    )

    assert total == future["rainfall_mm"].sum()


def test_monte_carlo_total_distribution():

    scenarios = [
        [
            {"rainfall_mm": 10},
            {"rainfall_mm": 20},
        ],
        [
            {"rainfall_mm": 5},
            {"rainfall_mm": 15},
        ],
    ]

    totals = calculate_monte_carlo_total_distribution(
        scenarios
    )

    assert isinstance(totals, np.ndarray)
    assert np.array_equal(
        totals,
        np.array([30.0, 20.0]),
    )


def test_evaluate_actual_against_distribution():

    simulated = np.array(
        [10, 20, 30, 40, 50],
        dtype=float,
    )

    result = evaluate_actual_against_distribution(
        actual_total_mm=30,
        simulated_totals=simulated,
    )

    assert result["actual_total_mm"] == 30
    assert result["simulated_mean_mm"] == 30
    assert result["simulated_median_mm"] == 30
    assert result["simulated_p10_mm"] == 14
    assert result["simulated_p90_mm"] == 46
    assert result["actual_percentile"] == 60


def test_monte_carlo_distribution_rejects_empty_input():

    with pytest.raises(ValueError):
        calculate_monte_carlo_total_distribution(
            []
        )


def test_evaluation_rejects_empty_distribution():

    with pytest.raises(ValueError):
        evaluate_actual_against_distribution(
            actual_total_mm=20,
            simulated_totals=[],
        )


def test_backtest_transition_calibration_uses_training_data():

    dataframe = make_test_rainfall_data()

    training = get_training_data(
        dataframe,
        "2024-06-30",
    )

    matrix = calibrate_backtest_transition_matrix(
        training
    )

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    assert matrix.shape == (3, 3)
    assert np.allclose(
        matrix.sum(axis=1),
        1.0,
    )
def test_run_single_backtest():

    from src.backtesting import run_single_backtest

    dataframe = make_test_rainfall_data()

    result = run_single_backtest(
        dataframe=dataframe,
        decision_date="2024-06-20",
        horizon_days=5,
        num_simulations=20,
        random_seed=42,
    )

    assert result["decision_date"] == pd.Timestamp(
        "2024-06-20"
    )

    assert result["initial_state"] in [
        "dry",
        "drizzle",
        "rain",
    ]

    assert result["training_rows"] == 19
    assert result["actual_future_rows"] == 5

    assert result["actual_total_mm"] >= 0
    assert result["simulated_mean_mm"] >= 0

    assert 0 <= result["actual_percentile"] <= 100


def test_run_single_backtest_is_reproducible():

    from src.backtesting import run_single_backtest

    dataframe = make_test_rainfall_data()

    result_1 = run_single_backtest(
        dataframe=dataframe,
        decision_date="2024-06-20",
        horizon_days=5,
        num_simulations=20,
        random_seed=42,
    )

    result_2 = run_single_backtest(
        dataframe=dataframe,
        decision_date="2024-06-20",
        horizon_days=5,
        num_simulations=20,
        random_seed=42,
    )

    assert result_1 == result_2


def test_run_multi_date_backtest():

    from src.backtesting import run_multi_date_backtest

    dataframe = make_test_rainfall_data()

    decision_dates = [
        "2024-06-20",
        "2024-06-25",
        "2024-06-30",
    ]

    result = run_multi_date_backtest(
        dataframe=dataframe,
        decision_dates=decision_dates,
        horizon_days=5,
        num_simulations=20,
        random_seed=42,
    )

    assert len(result) == 3

    assert list(result["decision_date"]) == [
        pd.Timestamp("2024-06-20"),
        pd.Timestamp("2024-06-25"),
        pd.Timestamp("2024-06-30"),
    ]

    assert (
        result["actual_percentile"]
        .between(0, 100)
        .all()
    )