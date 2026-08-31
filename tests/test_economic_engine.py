from src.economic_engine import (
    calculate_economic_outcome,
    compare_all_decisions,
    format_currency,
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