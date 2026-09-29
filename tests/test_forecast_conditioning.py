import math

import pytest

from src.forecast_conditioning import (
    condition_rainfall_state_probabilities,
)


HISTORICAL_PROBABILITIES = {
    "dry": 0.60,
    "drizzle": 0.25,
    "rain": 0.15,
}


def test_zero_forecast_weight_preserves_historical_distribution():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.90,
        forecast_weight=0.0,
    )

    assert result["dry"] == pytest.approx(0.60)
    assert result["drizzle"] == pytest.approx(0.25)
    assert result["rain"] == pytest.approx(0.15)


def test_full_forecast_weight_matches_forecast_wet_probability():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.80,
        forecast_weight=1.0,
    )

    assert result["dry"] == pytest.approx(0.20)

    # Historical wet-day composition:
    # drizzle = 0.25 / 0.40 = 0.625
    # rain = 0.15 / 0.40 = 0.375
    assert result["drizzle"] == pytest.approx(0.80 * 0.625)
    assert result["rain"] == pytest.approx(0.80 * 0.375)


def test_intermediate_weight_blends_wet_probability():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.80,
        forecast_weight=0.50,
    )

    # Historical wet probability = 0.25 + 0.15 = 0.40
    #
    # Conditioned wet probability:
    # 0.5 * 0.40 + 0.5 * 0.80 = 0.60
    assert result["dry"] == pytest.approx(0.40)
    assert result["drizzle"] == pytest.approx(0.375)
    assert result["rain"] == pytest.approx(0.225)


def test_drizzle_rain_ratio_is_preserved():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.70,
        forecast_weight=0.50,
    )

    historical_wet = (
        HISTORICAL_PROBABILITIES["drizzle"]
        + HISTORICAL_PROBABILITIES["rain"]
    )

    historical_drizzle_fraction = (
        HISTORICAL_PROBABILITIES["drizzle"]
        / historical_wet
    )

    historical_rain_fraction = (
        HISTORICAL_PROBABILITIES["rain"]
        / historical_wet
    )

    conditioned_wet = (
        result["drizzle"]
        + result["rain"]
    )

    assert (
        result["drizzle"] / conditioned_wet
        == pytest.approx(historical_drizzle_fraction)
    )

    assert (
        result["rain"] / conditioned_wet
        == pytest.approx(historical_rain_fraction)
    )


def test_conditioned_probabilities_sum_to_one():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.75,
        forecast_weight=0.40,
    )

    assert sum(result.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "forecast_probability",
    [0.0, 1.0],
)
def test_forecast_probability_boundaries(
    forecast_probability,
):
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=forecast_probability,
        forecast_weight=1.0,
    )

    assert sum(result.values()) == pytest.approx(1.0)

    if forecast_probability == 0.0:
        assert result["dry"] == pytest.approx(1.0)
        assert result["drizzle"] == pytest.approx(0.0)
        assert result["rain"] == pytest.approx(0.0)

    else:
        assert result["dry"] == pytest.approx(0.0)
        assert result["drizzle"] == pytest.approx(0.625)
        assert result["rain"] == pytest.approx(0.375)


@pytest.mark.parametrize(
    "historical_probabilities",
    [
        {
            "dry": 0.60,
            "drizzle": 0.25,
        },
        {
            "dry": 0.60,
            "drizzle": 0.25,
            "rain": 0.20,
        },
    ],
)
def test_invalid_historical_probabilities_are_rejected(
    historical_probabilities,
):
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            historical_probabilities,
            forecast_probability=0.50,
            forecast_weight=0.50,
        )


@pytest.mark.parametrize(
    "historical_probabilities",
    [
        {
            "dry": 0.50,
            "drizzle": 0.20,
            "rain": 0.20,
        },
        {
            "dry": 0.70,
            "drizzle": 0.20,
            "rain": 0.20,
        },
    ],
)
def test_historical_probabilities_must_sum_to_one(
    historical_probabilities,
):
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            historical_probabilities,
            forecast_probability=0.50,
            forecast_weight=0.50,
        )


@pytest.mark.parametrize(
    "forecast_probability",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_forecast_probability_is_rejected(
    forecast_probability,
):
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            HISTORICAL_PROBABILITIES,
            forecast_probability=forecast_probability,
            forecast_weight=0.50,
        )


@pytest.mark.parametrize(
    "forecast_weight",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_forecast_weight_is_rejected(
    forecast_weight,
):
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            HISTORICAL_PROBABILITIES,
            forecast_probability=0.50,
            forecast_weight=forecast_weight,
        )


def test_boolean_probability_is_rejected():
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            {
                "dry": True,
                "drizzle": 0.0,
                "rain": 0.0,
            },
            forecast_probability=0.50,
            forecast_weight=0.50,
        )


def test_boolean_forecast_probability_is_rejected():
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            HISTORICAL_PROBABILITIES,
            forecast_probability=True,
            forecast_weight=0.50,
        )


def test_boolean_forecast_weight_is_rejected():
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            HISTORICAL_PROBABILITIES,
            forecast_probability=0.50,
            forecast_weight=True,
        )


def test_missing_historical_state_is_rejected():
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            {
                "dry": 0.60,
                "drizzle": 0.40,
            },
            forecast_probability=0.50,
            forecast_weight=0.50,
        )


def test_non_mapping_historical_probabilities_are_rejected():
    with pytest.raises(TypeError):
        condition_rainfall_state_probabilities(
            ["dry", "drizzle", "rain"],
            forecast_probability=0.50,
            forecast_weight=0.50,
        )


def test_result_contains_only_expected_states():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.60,
        forecast_weight=0.50,
    )

    assert set(result) == {"dry", "drizzle", "rain"}


def test_conditioning_is_deterministic():
    result1 = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.65,
        forecast_weight=0.30,
    )

    result2 = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.65,
        forecast_weight=0.30,
    )

    assert result1 == result2


def test_all_output_probabilities_are_finite_and_bounded():
    result = condition_rainfall_state_probabilities(
        HISTORICAL_PROBABILITIES,
        forecast_probability=0.85,
        forecast_weight=0.75,
    )

    for probability in result.values():
        assert math.isfinite(probability)
        assert 0.0 <= probability <= 1.0

def test_zero_historical_wet_probability_is_rejected():
    with pytest.raises(ValueError):
        condition_rainfall_state_probabilities(
            {
                "dry": 1.0,
                "drizzle": 0.0,
                "rain": 0.0,
            },
            forecast_probability=0.80,
            forecast_weight=0.50,
        )