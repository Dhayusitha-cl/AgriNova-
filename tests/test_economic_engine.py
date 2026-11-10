import pytest

from src.crop_data import crops
from src.economic_engine import (
    calculate_economic_outcome,
    calculate_profit,
    compare_all_decisions,
    format_currency,
)

def test_switch_probability_rejects_boolean():
    with pytest.raises(ValueError):
        compare_all_decisions(
            crop_name="cotton",
            soil_type="sandy_loam",
            germ_prob_today=0.5,
            germ_prob_wait=0.5,
            switch_probabilities={
                "soybean": True
            },
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


def test_multi_crop_switch_returns_all_candidate_options():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.40,
        germ_prob_wait=0.30,
        switch_probabilities={
            "soybean": 0.60,
            "sorghum": 0.70,
            "paddy": 0.50,
        },
    )

    assert set(result["switch_options"]) == {
        "soybean",
        "sorghum",
        "paddy",
    }


def test_multi_crop_switch_preserves_candidate_probabilities():
    probabilities = {
        "soybean": 0.40,
        "sorghum": 0.75,
        "paddy": 0.55,
    }

    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.50,
        germ_prob_wait=0.45,
        switch_probabilities=probabilities,
    )

    for crop_name, probability in probabilities.items():
        assert (
            result["switch_options"][crop_name]["success_probability"]
            == probability
        )


def test_non_soybean_candidate_can_win():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.0,
        germ_prob_wait=0.0,
        switch_probabilities={
            "soybean": 0.0,
            "sorghum": 1.0,
        },
    )

    assert result["best_decision"] == "Switch to Sorghum"


def test_primary_crop_is_excluded_from_switch_options():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.60,
        germ_prob_wait=0.50,
        switch_probabilities={
            "cotton": 0.99,
            "soybean": 0.40,
        },
    )

    assert "cotton" not in result["switch_options"]
    assert "soybean" in result["switch_options"]


def test_empty_switch_options_preserve_sow_and_wait():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.70,
        germ_prob_wait=0.60,
        switch_probabilities={},
    )

    assert result["switch_options"] == {}
    assert result["switch"] is None
    assert len(result["all_decisions"]) == 2
    assert result["best_decision"] in {
        "Sow Today",
        "Wait 5 Days",
    }


def test_multi_crop_switch_rejects_invalid_probabilities():
    invalid_probabilities = [
        float("nan"),
        float("inf"),
        -0.1,
        1.1,
    ]

    for probability in invalid_probabilities:
        with pytest.raises(ValueError):
            compare_all_decisions(
                crop_name="cotton",
                soil_type="medium_black",
                germ_prob_today=0.50,
                germ_prob_wait=0.40,
                switch_probabilities={
                    "sorghum": probability,
                },
            )


def test_multi_crop_switch_legacy_alias_prefers_soybean():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.0,
        germ_prob_wait=0.0,
        switch_probabilities={
            "soybean": 0.40,
            "sorghum": 1.0,
        },
    )

    assert result["best_decision"] == "Switch to Sorghum"
    assert result["switch"] is result["switch_options"]["soybean"]


def test_multi_crop_switch_legacy_alias_uses_best_candidate_without_soybean():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.0,
        germ_prob_wait=0.0,
        switch_probabilities={
            "sorghum": 1.0,
        },
    )

    assert result["switch"] is result["switch_options"]["sorghum"]


def test_legacy_mode_excludes_primary_soybean_from_switch():
    result = compare_all_decisions(
        crop_name="soybean",
        soil_type="medium_black",
        germ_prob_today=0.60,
        germ_prob_wait=0.50,
        germ_prob_soybean=0.80,
    )

    assert result["switch_options"] == {}
    assert result["switch"] is None
    assert len(result["all_decisions"]) == 2
    assert result["best_decision"] in {
        "Sow Today",
        "Wait 5 Days",
    }


def test_compare_all_decisions_handles_one_eligible_option():
    result = compare_all_decisions(
        crop_name="cotton",
        soil_type="medium_black",
        germ_prob_today=0.8,
        germ_prob_wait=0.7,
        switch_probabilities={},
        sow_eligible=True,
        wait_eligible=False,
    )

    assert len(result["all_decisions"]) == 1
    assert result["best_decision"] == "Sow Today"
    assert (
        result["all_decisions"][0]["advantage_over_others"]
        == 0.0
    )


def test_compare_all_decisions_rejects_no_eligible_options():
    with pytest.raises(
        ValueError,
        match="No eligible sowing decision is available",
    ):
        compare_all_decisions(
            crop_name="cotton",
            soil_type="medium_black",
            germ_prob_today=0.8,
            germ_prob_wait=0.7,
            switch_probabilities={},
            sow_eligible=False,
            wait_eligible=False,
        )