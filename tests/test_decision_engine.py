import numpy as np
import pandas as pd
import pytest

from datetime import datetime
from datetime import date
from src.decision_engine import (
    calculate_daily_et,
    _get_initial_state,
    _calculate_confidence,
    _calculate_decision_separation,
    _safe_wait_trajectories,
    _scenario_establishment_probability,
    make_decision,
)
from src.forecast import DailyForecast, WeatherForecast
from src.decision_engine import _assess_crop_eligibility


VALID_MATRIX = [
    [0.60, 0.30, 0.10],
    [0.20, 0.60, 0.20],
    [0.10, 0.30, 0.60],
]


def _make_transition_matrix():
    return np.array([
        [0.70, 0.20, 0.10],
        [0.30, 0.50, 0.20],
        [0.10, 0.30, 0.60],
    ])


def _make_decision_forecast(
    start_date="2024-07-01",
    num_days=20,
    precipitation_probability=0.8,
):
    start = pd.Timestamp(start_date)

    daily = tuple(
        DailyForecast(
            forecast_date=(
                start + pd.Timedelta(days=day)
            ).date(),
            rainfall_mm=20.0,
            precipitation_probability=(
                precipitation_probability
            ),
        )
        for day in range(num_days)
    )

    return WeatherForecast(
        source="test-provider",
        issued_at=datetime(2024, 6, 30),
        daily=daily,
    )


def test_calculate_daily_et_returns_one_value_per_day():
    rainfall_scenario = [
        {"rainfall_mm": 0.0, "state": "dry"},
        {"rainfall_mm": 5.0, "state": "drizzle"},
        {"rainfall_mm": 20.0, "state": "rain"},
    ]

    result = calculate_daily_et(rainfall_scenario)

    assert result == [5.0, 5.0, 5.0]
    assert len(result) == len(rainfall_scenario)


def test_initial_rainfall_state_mapping():
    assert _get_initial_state(0) == "dry"
    assert _get_initial_state(0.0) == "dry"
    assert _get_initial_state(5) == "drizzle"
    assert _get_initial_state(9.9) == "drizzle"
    assert _get_initial_state(10) == "rain"
    assert _get_initial_state(25) == "rain"


def test_confidence_is_between_zero_and_one():
    confidence = _calculate_confidence(
        germ_prob_today=0.80,
        germ_prob_wait=0.50,
        germ_prob_soybean=0.40,
    )

    assert 0.0 <= confidence <= 1.0


def test_confidence_increases_with_probability_separation():
    small_gap = _calculate_confidence(
        germ_prob_today=0.60,
        germ_prob_wait=0.55,
        germ_prob_soybean=0.50,
    )

    large_gap = _calculate_confidence(
        germ_prob_today=0.90,
        germ_prob_wait=0.50,
        germ_prob_soybean=0.40,
    )

    assert large_gap > small_gap


def test_make_decision_returns_expected_structure():
    result = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=30,
        days_to_simulate=14,
        random_seed=42,
    )

    assert isinstance(result, dict)

    required_keys = {
        "decision",
        "color",
        "germ_prob_today",
        "germ_prob_wait",
        "germ_prob_soybean",
        "confidence",
        "trajectories",
        "wait_simulations",
        "current_moisture",
        "min_moisture_required",
        "initial_rainfall_state",
        "num_simulations",
        "days_to_simulate",
        "assumptions",
    }

    assert required_keys.issubset(result.keys())


def test_make_decision_returns_valid_recommendation():
    result = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=30,
        days_to_simulate=14,
        random_seed=42,
    )

    assert result["decision"] in {
        "SOW TODAY",
        "WAIT 5 DAYS",
        "SWITCH TO SOYBEAN",
    }


def test_make_decision_probabilities_are_valid():
    result = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=30,
        days_to_simulate=14,
        random_seed=42,
    )

    assert 0.0 <= result["germ_prob_today"] <= 1.0
    assert 0.0 <= result["germ_prob_wait"] <= 1.0
    assert 0.0 <= result["germ_prob_soybean"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_make_decision_is_reproducible_with_same_seed():
    result_1 = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=20,
        days_to_simulate=14,
        random_seed=123,
    )

    result_2 = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=20,
        days_to_simulate=14,
        random_seed=123,
    )

    assert result_1["decision"] == result_2["decision"]
    assert result_1["germ_prob_today"] == result_2["germ_prob_today"]
    assert result_1["germ_prob_wait"] == result_2["germ_prob_wait"]
    assert result_1["germ_prob_soybean"] == result_2["germ_prob_soybean"]


def test_negative_moisture_is_rejected():
    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=-1,
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_moisture_above_field_capacity_is_rejected():
    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=1000,
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_negative_rainfall_is_rejected():
    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=-1,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_invalid_transition_matrix_shape_is_rejected():
    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            num_simulations=20,
            days_to_simulate=14,
        )


def test_transition_matrix_rows_must_sum_to_one():
    invalid_matrix = [
        [0.60, 0.30, 0.30],
        [0.20, 0.60, 0.20],
        [0.10, 0.30, 0.60],
    ]

    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=invalid_matrix,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_zero_simulations_are_rejected():
    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=0,
            days_to_simulate=14,
        )


def test_zero_days_are_rejected():
    with pytest.raises(ValueError):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=0,
        )


def test_make_decision_includes_economic_comparison():
    result = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=20,
        days_to_simulate=14,
        random_seed=42,
    )

    assert "economic_comparison" in result

    economic = result["economic_comparison"]

    assert "sow_today" in economic
    assert "wait" in economic
    assert "switch" in economic
    assert "best_decision" in economic
    assert "best_profit" in economic

    assert isinstance(
        economic["best_profit"],
        (int, float),
    )


def test_final_decision_matches_economic_best_decision():
    result = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=40,
        rainfall_yesterday_mm=10,
        transition_matrix=VALID_MATRIX,
        num_simulations=20,
        days_to_simulate=14,
        random_seed=42,
    )

    economic_best = result["economic_comparison"]["best_decision"]

    mapping = {
        "Sow Today": "SOW TODAY",
        "Wait 5 Days": "WAIT 5 DAYS",
        "Switch to Soybean": "SWITCH TO SOYBEAN",
    }

    assert result["decision"] == mapping[economic_best]


def test_scenario_establishment_probability_all_successful():

    scenarios = [
        [
            {"rainfall_mm": 10.0, "state": "rain"}
            for _ in range(8)
        ],
        [
            {"rainfall_mm": 10.0, "state": "rain"}
            for _ in range(8)
        ],
    ]

    probability, trajectories = _scenario_establishment_probability(
        scenarios=scenarios,
        crop_name="cotton",
        soil_type="medium_black",
        initial_moisture_mm=30.0,
    )

    assert probability == 1.0
    assert trajectories.shape == (2, 9)


def test_scenario_establishment_probability_all_failed():

    scenarios = [
        [
            {"rainfall_mm": 0.0, "state": "dry"}
            for _ in range(8)
        ],
        [
            {"rainfall_mm": 0.0, "state": "dry"}
            for _ in range(8)
        ],
    ]

    probability, trajectories = _scenario_establishment_probability(
        scenarios=scenarios,
        crop_name="cotton",
        soil_type="medium_black",
        initial_moisture_mm=0.0,
    )

    assert probability == 0.0
    assert trajectories.shape == (2, 9)


def test_scenario_establishment_probability_mixed_outcomes():

    scenarios = [
        [
            {"rainfall_mm": 20.0, "state": "rain"}
            for _ in range(8)
        ],
        [
            {"rainfall_mm": 0.0, "state": "dry"}
            for _ in range(8)
        ],
    ]

    probability, trajectories = _scenario_establishment_probability(
        scenarios=scenarios,
        crop_name="cotton",
        soil_type="medium_black",
        initial_moisture_mm=0.0,
    )

    assert probability == 0.5
    assert trajectories.shape == (2, 9)


def test_make_decision_is_reproducible_with_same_seed():

    result_1 = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=30.0,
        rainfall_yesterday_mm=10.0,
        transition_matrix=VALID_MATRIX,
        num_simulations=20,
        days_to_simulate=8,
        random_seed=123,
    )

    result_2 = make_decision(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=30.0,
        rainfall_yesterday_mm=10.0,
        transition_matrix=VALID_MATRIX,
        num_simulations=20,
        days_to_simulate=8,
        random_seed=123,
    )

    assert result_1["decision"] == result_2["decision"]

    assert result_1["germ_prob_today"] == result_2["germ_prob_today"]
    assert result_1["germ_prob_wait"] == result_2["germ_prob_wait"]
    assert result_1["germ_prob_soybean"] == result_2["germ_prob_soybean"]

    np.testing.assert_array_equal(
        result_1["trajectories"],
        result_2["trajectories"],
    )

    np.testing.assert_array_equal(
        result_1["wait_simulations"],
        result_2["wait_simulations"],
    )

    np.testing.assert_array_equal(
        result_1["soybean_trajectories"],
        result_2["soybean_trajectories"],
    )

def test_nan_moisture_is_rejected():
    with pytest.raises(
        ValueError,
        match="current_moisture_mm must be a finite number",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=float("nan"),
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_infinite_moisture_is_rejected():
    with pytest.raises(
        ValueError,
        match="current_moisture_mm must be a finite number",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=float("inf"),
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_nan_rainfall_is_rejected():
    with pytest.raises(
        ValueError,
        match="rainfall_yesterday_mm must be a finite number",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=float("nan"),
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_infinite_rainfall_is_rejected():
    with pytest.raises(
        ValueError,
        match="rainfall_yesterday_mm must be a finite number",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=float("inf"),
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_nan_transition_matrix_is_rejected():
    invalid_matrix = [
        [0.60, 0.30, float("nan")],
        [0.20, 0.60, 0.20],
        [0.10, 0.30, 0.60],
    ]

    with pytest.raises(
        ValueError,
        match="transition probabilities must be finite numbers",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=invalid_matrix,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_infinite_transition_matrix_is_rejected():
    invalid_matrix = [
        [0.60, 0.30, float("inf")],
        [0.20, 0.60, 0.20],
        [0.10, 0.30, 0.60],
    ]

    with pytest.raises(
        ValueError,
        match="transition probabilities must be finite numbers",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=invalid_matrix,
            num_simulations=20,
            days_to_simulate=14,
        )


def test_non_integer_simulations_are_rejected():
    with pytest.raises(
        ValueError,
        match="num_simulations must be an integer",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20.5,
            days_to_simulate=14,
        )


def test_non_integer_days_are_rejected():
    with pytest.raises(
        ValueError,
        match="days_to_simulate must be an integer",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=40,
            rainfall_yesterday_mm=10,
            transition_matrix=VALID_MATRIX,
            num_simulations=20,
            days_to_simulate=14.5,
        )


def test_invalid_initial_state_is_rejected():
    with pytest.raises(
        ValueError,
        match="initial_state must be one of",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="sandy_loam",
            current_moisture_mm=18.0,
            rainfall_yesterday_mm=5.0,
            transition_matrix=VALID_MATRIX,
            num_simulations=10,
            days_to_simulate=7,
            initial_state="invalid",
        )


def test_scenario_establishment_probability_rejects_short_scenario():
    scenarios = [
        [
            {"rainfall_mm": 10.0, "state": "rain"}
            for _ in range(7)
        ]
    ]

    with pytest.raises(
        ValueError,
        match="Scenario is shorter than the crop germination period",
    ):
        _scenario_establishment_probability(
            scenarios=scenarios,
            crop_name="cotton",
            soil_type="medium_black",
            initial_moisture_mm=30.0,
        )


def test_safe_wait_trajectories_rejects_short_scenario():
    wait_scenarios = [
        [
            {"rainfall_mm": 10.0, "state": "rain"}
            for _ in range(12)
        ]
    ]

    with pytest.raises(
        ValueError,
        match=(
            "WAIT scenario is shorter than the required "
            "wait plus crop germination period"
        ),
    ):
        _safe_wait_trajectories(
            wait_scenarios=wait_scenarios,
            wait_days=5,
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=30.0,
        )

def test_make_decision_rejects_forecast_without_start_date():
    forecast = _make_decision_forecast()

    with pytest.raises(ValueError, match="forecast requires start_date"):
        make_decision(
            crop_name="cotton",
            soil_type="sandy_loam",
            current_moisture_mm=20.0,
            rainfall_yesterday_mm=10.0,
            transition_matrix=_make_transition_matrix(),
            num_simulations=10,
            days_to_simulate=7,
            forecast=forecast,
            forecast_weight=0.5,
        )


def test_make_decision_rejects_forecast_with_transition_matrix():
    forecast = _make_decision_forecast()

    with pytest.raises(
        ValueError,
        match="forecast cannot be used with an explicit transition_matrix",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="sandy_loam",
            current_moisture_mm=20.0,
            rainfall_yesterday_mm=10.0,
            transition_matrix=_make_transition_matrix(),
            num_simulations=10,
            days_to_simulate=7,
            start_date="2024-07-01",
            forecast=forecast,
            forecast_weight=0.5,
        )


def test_make_decision_requires_forecast_weight():
    forecast = _make_decision_forecast()

    with pytest.raises(
        ValueError,
        match="forecast_weight must be provided",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="sandy_loam",
            current_moisture_mm=20.0,
            rainfall_yesterday_mm=10.0,
            transition_matrix=None,
            num_simulations=10,
            days_to_simulate=7,
            start_date="2024-07-01",
            forecast=forecast,
        )


def test_make_decision_rejects_forecast_weight_without_forecast():
    with pytest.raises(
        ValueError,
        match="forecast_weight cannot be provided without forecast",
    ):
        make_decision(
            crop_name="cotton",
            soil_type="sandy_loam",
            current_moisture_mm=20.0,
            rainfall_yesterday_mm=10.0,
            transition_matrix=None,
            num_simulations=10,
            days_to_simulate=7,
            start_date="2024-07-01",
            forecast_weight=0.5,
        )

def test_crop_eligibility_within_configured_window():
    result = _assess_crop_eligibility(
        "soybean",
        start_date="2024-07-01",
    )

    assert result["eligible"] is True
    assert result["status"] == "within_configured_window"


def test_crop_eligibility_outside_configured_window():
    result = _assess_crop_eligibility(
        "soybean",
        start_date="2024-08-01",
    )

    assert result["eligible"] is False
    assert result["status"] == "outside_configured_window"


def test_crop_eligibility_without_date_is_unassessed():
    result = _assess_crop_eligibility("soybean")

    assert result["eligible"] is True
    assert result["status"] == "timing_unassessed"
    assert "could not be assessed" in result["reason"]


def test_crop_eligibility_rejects_invalid_date():
    with pytest.raises(ValueError, match="start_date"):
        _assess_crop_eligibility(
            "soybean",
            start_date="not-a-date",
        )


def test_crop_eligibility_supports_year_crossing_window(monkeypatch):
    from src import decision_engine

    monkeypatch.setitem(
        decision_engine.crops["soybean"],
        "optimal_sowing_window",
        "November 15 - January 15",
    )

    for assessment_date in ("2024-11-20", "2025-01-10"):
        result = _assess_crop_eligibility(
            "soybean",
            start_date=assessment_date,
        )
        assert result["eligible"] is True

    result = _assess_crop_eligibility(
        "soybean",
        start_date="2025-02-01",
    )
    assert result["eligible"] is False


def test_crop_eligibility_rejects_date_with_trailing_content():
    with pytest.raises(ValueError, match="start_date"):
        _assess_crop_eligibility(
            "soybean",
            start_date="2024-07-01-invalid",
        )


def test_crop_eligibility_accepts_date_object():
    result = _assess_crop_eligibility(
        "soybean",
        start_date=date(2024, 7, 1),
    )

    assert result["eligible"] is True


def test_decision_separation_uses_top_two_profits():
    decisions = [
        {"expected_profit": 1000.0},
        {"expected_profit": 700.0},
        {"expected_profit": 200.0},
    ]

    assert _calculate_decision_separation(decisions) == 300.0


def test_decision_separation_handles_tie():
    decisions = [
        {"expected_profit": 500.0},
        {"expected_profit": 500.0},
    ]

    assert _calculate_decision_separation(decisions) == 0.0


def test_decision_separation_handles_negative_profits():
    decisions = [
        {"expected_profit": -100.0},
        {"expected_profit": -300.0},
    ]

    assert _calculate_decision_separation(decisions) == 200.0


def test_decision_separation_with_one_option():
    decisions = [
        {"expected_profit": 500.0},
    ]

    assert _calculate_decision_separation(decisions) == 0.0


def test_decision_separation_with_no_options():
    assert _calculate_decision_separation([]) == 0.0