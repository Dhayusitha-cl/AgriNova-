import pytest

from src.crop_data import crops
from src.economic_engine import (
    calculate_economic_outcome,
    calculate_profit,
    compare_all_decisions,
    format_currency,
)


def test_calculate_profit_rejects_nan_probability():
    with pytest.raises(
        ValueError,
        match="success_probability must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=float("nan"),
        )


def test_calculate_profit_rejects_infinite_probability():
    with pytest.raises(
        ValueError,
        match="success_probability must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=float("inf"),
        )


def test_calculate_profit_rejects_negative_infinite_probability():
    with pytest.raises(
        ValueError,
        match="success_probability must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=float("-inf"),
        )


def test_calculate_profit_rejects_negative_probability():
    with pytest.raises(
        ValueError,
        match="success_probability must be between 0 and 1",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=-0.1,
        )


def test_calculate_profit_rejects_probability_above_one():
    with pytest.raises(
        ValueError,
        match="success_probability must be between 0 and 1",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=1.1,
        )


def test_calculate_profit_rejects_nan_yield():
    with pytest.raises(
        ValueError,
        match="yield_per_acre must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=float("nan"),
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=0.5,
        )


def test_calculate_profit_rejects_infinite_yield():
    with pytest.raises(
        ValueError,
        match="yield_per_acre must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=float("inf"),
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=0.5,
        )


def test_calculate_profit_rejects_negative_yield():
    with pytest.raises(
        ValueError,
        match="yield_per_acre cannot be negative",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=-1,
            price_per_quintal=6500,
            seed_cost=3400,
            success_probability=0.5,
        )


def test_calculate_profit_rejects_nan_price():
    with pytest.raises(
        ValueError,
        match="price_per_quintal must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=float("nan"),
            seed_cost=3400,
            success_probability=0.5,
        )


def test_calculate_profit_rejects_infinite_price():
    with pytest.raises(
        ValueError,
        match="price_per_quintal must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=float("inf"),
            seed_cost=3400,
            success_probability=0.5,
        )


def test_calculate_profit_rejects_negative_price():
    with pytest.raises(
        ValueError,
        match="price_per_quintal cannot be negative",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=-1,
            seed_cost=3400,
            success_probability=0.5,
        )


def test_calculate_profit_rejects_nan_seed_cost():
    with pytest.raises(
        ValueError,
        match="seed_cost must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=float("nan"),
            success_probability=0.5,
        )


def test_calculate_profit_rejects_infinite_seed_cost():
    with pytest.raises(
        ValueError,
        match="seed_cost must be a finite number",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=float("inf"),
            success_probability=0.5,
        )


def test_calculate_profit_rejects_negative_seed_cost():
    with pytest.raises(
        ValueError,
        match="seed_cost cannot be negative",
    ):
        calculate_profit(
            crop_name="cotton",
            yield_per_acre=10,
            price_per_quintal=6500,
            seed_cost=-1,
            success_probability=0.5,
        )


def test_wait_rejects_nan_yield_loss_parameter(monkeypatch):
    monkeypatch.setitem(
        crops["cotton"],
        "yield_loss_per_day_pct",
        float("nan"),
    )

    with pytest.raises(
        ValueError,
        match="yield_loss_per_day_pct must be a finite number",
    ):
        calculate_economic_outcome(
            crop_name="cotton",
            soil_type="medium_black",
            decision="wait",
            germination_prob=0.5,
            rainfall_yesterday_mm=10,
            current_moisture_mm=100,
        )


def test_wait_rejects_infinite_yield_loss_parameter(monkeypatch):
    monkeypatch.setitem(
        crops["cotton"],
        "yield_loss_per_day_pct",
        float("inf"),
    )

    with pytest.raises(
        ValueError,
        match="yield_loss_per_day_pct must be a finite number",
    ):
        calculate_economic_outcome(
            crop_name="cotton",
            soil_type="medium_black",
            decision="wait",
            germination_prob=0.5,
            rainfall_yesterday_mm=10,
            current_moisture_mm=100,
        )


def test_wait_rejects_negative_yield_loss_parameter(monkeypatch):
    monkeypatch.setitem(
        crops["cotton"],
        "yield_loss_per_day_pct",
        -1,
    )

    with pytest.raises(
        ValueError,
        match="yield_loss_per_day_pct cannot be negative",
    ):
        calculate_economic_outcome(
            crop_name="cotton",
            soil_type="medium_black",
            decision="wait",
            germination_prob=0.5,
            rainfall_yesterday_mm=10,
            current_moisture_mm=100,
        )

def test_format_currency():
    assert format_currency(45000) == "₹45,000"


def test_sow_today_returns_expected_fields():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="sow_today",
        germination_prob=0.8,
        rainfall_yesterday_mm=10,
        current_moisture_mm=100,
    )

    assert result["decision"] == "Sow Today"
    assert 0 <= result["success_probability"] <= 1
    assert isinstance(result["expected_profit"], (int, float))
    assert isinstance(result["best_case_profit"], (int, float))
    assert isinstance(result["worst_case_profit"], (int, float))
    assert result["risk_level"] in {"Low", "Medium", "High"}


def test_wait_returns_expected_fields():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="wait",
        germination_prob=0.6,
        rainfall_yesterday_mm=5,
        current_moisture_mm=90,
    )

    assert result["decision"] == "Wait 5 Days"
    assert 0 <= result["success_probability"] <= 1
    assert isinstance(result["expected_profit"], (int, float))
    assert result["risk_level"] in {"Low", "Medium", "High"}


def test_switch_returns_expected_fields():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="switch",
        germination_prob=0.4,
        rainfall_yesterday_mm=0,
        current_moisture_mm=60,
    )

    assert result["decision"] == "Switch to Soybean"
    assert 0 <= result["success_probability"] <= 1
    assert isinstance(result["expected_profit"], (int, float))
    assert result["risk_level"] == "High"


def test_unknown_decision_is_handled():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="invalid_decision",
        germination_prob=0.5,
        rainfall_yesterday_mm=10,
        current_moisture_mm=100,
    )

    assert result["decision"] == "Unknown"
    assert result["expected_profit"] == 0
    assert result["success_probability"] == 0


def test_compare_all_decisions_returns_all_options():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.8,
        germ_prob_wait=0.7,
        germ_prob_soybean=0.6,
        rainfall_yesterday_mm=10,
        current_moisture_mm=100,
    )

    assert "sow_today" in result
    assert "wait" in result
    assert "switch" in result

    assert "best_decision" in result
    assert "best_profit" in result
    assert "all_decisions" in result

    assert len(result["all_decisions"]) == 3


def test_compare_all_decisions_selects_highest_expected_profit():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.9,
        germ_prob_wait=0.2,
        germ_prob_soybean=0.2,
        rainfall_yesterday_mm=20,
        current_moisture_mm=150,
    )

    profits = [
        result["sow_today"]["expected_profit"],
        result["wait"]["expected_profit"],
        result["switch"]["expected_profit"],
    ]

    assert result["best_profit"] == max(profits)


def test_expected_profit_uses_probability_once():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="sow_today",
        germination_prob=1.0,
        rainfall_yesterday_mm=10,
        current_moisture_mm=100,
    )

    expected_best_case = (
        10 * 6500
    ) - 3400

    assert result["best_case_profit"] == expected_best_case


def test_switch_uses_supplied_soybean_probability():
    low_probability = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="switch",
        germination_prob=0.20,
        rainfall_yesterday_mm=0,
        current_moisture_mm=20,
    )

    high_probability = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="switch",
        germination_prob=0.90,
        rainfall_yesterday_mm=0,
        current_moisture_mm=20,
    )

    assert (
        high_probability["expected_profit"]
        > low_probability["expected_profit"]
    )


def test_sow_today_does_not_assume_unmodeled_recovery_when_establishment_fails():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="sow_today",
        germination_prob=0.0,
        rainfall_yesterday_mm=0,
        current_moisture_mm=10,
    )

    assert result["success_probability"] == 0.0
    assert result["expected_profit"] == -3400.0
    assert result["worst_case_profit"] == -3400.0

def test_compare_all_decisions_advantage_is_against_best_alternative():
    from src.economic_engine import compare_all_decisions

    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.30,
        germ_prob_wait=0.20,
        germ_prob_soybean=0.40,
        rainfall_yesterday_mm=10.0,
        current_moisture_mm=30.0,
    )

    decisions = result["all_decisions"]

    for decision in decisions:
        other_profits = [
            other["expected_profit"]
            for other in decisions
            if other is not decision
        ]

        expected_advantage = (
            decision["expected_profit"]
            - max(other_profits)
        )

        assert decision["advantage_over_others"] == expected_advantage


def test_compare_all_decisions_uses_each_probability_for_its_decision():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.90,
        germ_prob_wait=0.20,
        germ_prob_soybean=0.40,
        rainfall_yesterday_mm=10,
        current_moisture_mm=100,
    )

    assert result["sow_today"]["success_probability"] == 0.90
    assert result["wait"]["success_probability"] == 0.20
    assert result["switch"]["success_probability"] == 0.40


def test_expected_profit_matches_probability_weighted_outcomes():
    result = calculate_economic_outcome(
        crop_name="cotton",
        soil_type="medium_black",
        decision="sow_today",
        germination_prob=0.50,
        rainfall_yesterday_mm=10,
        current_moisture_mm=100,
    )

    success_profit = (10 * 6500) - 3400
    failure_profit = -3400

    expected_profit = (
        0.50 * success_profit
        + 0.50 * failure_profit
    )

    assert result["expected_profit"] == expected_profit