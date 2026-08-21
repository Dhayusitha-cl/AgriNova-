import numpy as np
import pandas as pd
import pytest

from src.markov_calibration import (
    STATES,
    calculate_transition_matrix,
    count_transitions,
    transition_counts_dataframe,
    validate_transition_matrix,
)


def sample_rainfall_data():
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=6,
            ),
            "rainfall_state": [
                "dry",
                "dry",
                "drizzle",
                "rain",
                "rain",
                "dry",
            ],
        }
    )


def test_count_transitions():
    dataframe = sample_rainfall_data()

    counts = count_transitions(dataframe)

    assert counts["dry"]["dry"] == 1
    assert counts["dry"]["drizzle"] == 1
    assert counts["drizzle"]["rain"] == 1
    assert counts["rain"]["rain"] == 1
    assert counts["rain"]["dry"] == 1


def test_transition_matrix_shape():
    dataframe = sample_rainfall_data()

    matrix = calculate_transition_matrix(dataframe)

    assert matrix.shape == (3, 3)


def test_transition_matrix_rows_sum_to_one():
    dataframe = sample_rainfall_data()

    matrix = calculate_transition_matrix(dataframe)

    assert np.allclose(
        matrix.sum(axis=1),
        1.0,
    )


def test_transition_matrix_contains_valid_probabilities():
    dataframe = sample_rainfall_data()

    matrix = calculate_transition_matrix(dataframe)

    assert np.all(matrix >= 0)
    assert np.all(matrix <= 1)


def test_transition_counts_dataframe():
    dataframe = sample_rainfall_data()

    result = transition_counts_dataframe(dataframe)

    assert list(result.index) == STATES
    assert list(result.columns) == STATES
    assert result.loc["dry", "dry"] == 1


def test_validate_transition_matrix():
    valid_matrix = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ]
    )

    assert validate_transition_matrix(valid_matrix)


def test_validate_rejects_wrong_shape():
    invalid_matrix = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    with pytest.raises(ValueError):
        validate_transition_matrix(invalid_matrix)


def test_validate_rejects_invalid_row_sum():
    invalid_matrix = np.array(
        [
            [0.5, 0.2, 0.2],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ]
    )

    with pytest.raises(ValueError):
        validate_transition_matrix(invalid_matrix)
def test_multi_year_transition_counts():
    from src.markov_calibration import (
        multi_year_transition_counts_dataframe,
    )

    counts = multi_year_transition_counts_dataframe()

    assert counts.shape == (3, 3)
    assert list(counts.index) == ["dry", "drizzle", "rain"]
    assert list(counts.columns) == ["dry", "drizzle", "rain"]

    assert counts.values.sum() > 0
    assert (counts.values >= 0).all()


def test_multi_year_transition_matrix():
    from src.markov_calibration import (
        multi_year_transition_matrix_dataframe,
    )

    matrix = multi_year_transition_matrix_dataframe()

    assert matrix.shape == (3, 3)

    assert list(matrix.index) == ["dry", "drizzle", "rain"]
    assert list(matrix.columns) == ["dry", "drizzle", "rain"]

    assert (matrix.values >= 0).all()
    assert (matrix.values <= 1).all()

    assert np.allclose(
        matrix.sum(axis=1).values,
        1.0,
    )


def test_multi_year_transition_matrix_is_valid():
    from src.markov_calibration import (
        calculate_multi_year_transition_matrix,
        validate_transition_matrix,
    )

    matrix = calculate_multi_year_transition_matrix()

    assert validate_transition_matrix(matrix) is True
def test_monthly_transition_matrix_uses_month_specific_data():
    from src.markov_calibration import (
        calculate_monthly_transition_matrices_from_training,
        combine_historical_rainfall_data,
        get_monthly_transition_matrix_with_fallback,
    )

    dataframe = combine_historical_rainfall_data()

    monthly_matrices = calculate_monthly_transition_matrices_from_training(
        dataframe
    )

    result = get_monthly_transition_matrix_with_fallback(
        dataframe,
        month=7,
    )

    assert np.allclose(
        result,
        monthly_matrices[7],
    )

    assert np.allclose(
        result.sum(axis=1),
        1.0,
    )


def test_monthly_transition_matrix_falls_back_for_unobserved_state():
    from src.markov_calibration import (
        calculate_transition_matrix,
        combine_historical_rainfall_data,
        get_monthly_transition_matrix_with_fallback,
    )

    dataframe = combine_historical_rainfall_data()

    fallback = calculate_transition_matrix(dataframe)

    result = get_monthly_transition_matrix_with_fallback(
        dataframe,
        month=1,
    )

    rain_index = STATES.index("rain")

    assert np.allclose(
        result[rain_index],
        fallback[rain_index],
    )

    assert np.allclose(
        result.sum(axis=1),
        1.0,
    )


def test_monthly_transition_matrix_rejects_invalid_month():
    from src.markov_calibration import (
        combine_historical_rainfall_data,
        get_monthly_transition_matrix_with_fallback,
    )

    dataframe = combine_historical_rainfall_data()

    with pytest.raises(ValueError):
        get_monthly_transition_matrix_with_fallback(
            dataframe,
            month=0,
        )

    with pytest.raises(ValueError):
        get_monthly_transition_matrix_with_fallback(
            dataframe,
            month=13,
        )