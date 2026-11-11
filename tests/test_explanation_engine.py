
from src.explanation_engine import generate_decision_explanation


def sample_result(decision="SOW TODAY"):
    return {
        "decision": decision,
        "economic_comparison": {
            "all_decisions": [
                {
                    "decision": "Sow Today",
                    "expected_profit": 12000.0,
                    "success_probability": 0.80,
                    "risk_level": "Low",
                    "advantage_over_others": 2000.0,
                },
                {
                    "decision": "Wait 5 Days",
                    "expected_profit": 10000.0,
                    "success_probability": 0.70,
                    "risk_level": "Medium",
                    "advantage_over_others": -2000.0,
                },
                {
                    "decision": "Switch to Soybean",
                    "expected_profit": 9000.0,
                    "success_probability": 0.65,
                    "risk_level": "Medium",
                    "advantage_over_others": -3000.0,
                },
            ]
        },
        "current_moisture": 30.0,
        "min_moisture_required": 36.0,
        "initial_rainfall_state": "dry",
        "crop_eligibility": {
            "cotton": {
                "eligible": False,
                "status": "outside_configured_window",
                "reason": "Outside configured sowing window.",
            },
            "soybean": {
                "eligible": True,
                "status": "within_configured_window",
                "reason": "Within configured sowing window.",
            },
        },
        "sowing_eligibility": {
            "sow_today": {
                "eligible": True,
                "status": "within_configured_window",
                "reason": "Within configured sowing window.",
            },
            "wait": {
                "eligible": True,
                "status": "within_configured_window",
                "reason": "Within configured sowing window.",
                "proposed_sowing_date": "2026-06-20",
            },
        },
        "assumptions": {
            "wait_days": 5,
        },
    }


def test_sow_recommendation_is_preserved():
    result = sample_result("SOW TODAY")

    explanation = generate_decision_explanation(
        result,
        primary_crop="cotton",
    )

    assert explanation["recommended_action"] == "SOW TODAY"
    assert explanation["recommended_crop"] == "cotton"
    assert len(explanation["why_recommended"]) > 0


def test_wait_recommendation_is_preserved():
    result = sample_result("WAIT 5 DAYS")

    explanation = generate_decision_explanation(
        result,
        primary_crop="cotton",
    )

    assert explanation["recommended_action"] == "WAIT 5 DAYS"
    assert explanation["recommended_crop"] == "cotton"


def test_switch_recommendation_identifies_alternative_crop():
    result = sample_result("SWITCH TO SOYBEAN")

    explanation = generate_decision_explanation(
        result,
        primary_crop="cotton",
    )

    assert explanation["recommended_action"] == "SWITCH TO SOYBEAN"
    assert explanation["recommended_crop"] == "SOYBEAN"


def test_field_conditions_are_included():
    explanation = generate_decision_explanation(sample_result())

    observations = " ".join(
        explanation["field_observations"]
    )

    assert "30.0 mm" in observations
    assert "36.0 mm" in observations
    assert "below" in observations


def test_ineligible_crop_reason_is_preserved():
    explanation = generate_decision_explanation(sample_result())

    cotton = next(
        item
        for item in explanation["crop_eligibility"]
        if item["crop"] == "cotton"
    )

    assert cotton["eligible"] is False
    assert cotton["status"] == "outside_configured_window"


def test_missing_matching_economic_outcome_does_not_change_decision():
    result = sample_result("WAIT 5 DAYS")
    result["economic_comparison"]["all_decisions"] = []

    explanation = generate_decision_explanation(result)

    assert explanation["recommended_action"] == "WAIT 5 DAYS"
    assert "not found" in explanation["why_recommended"][0].lower()


def test_invalid_result_is_rejected():
    import pytest

    with pytest.raises(TypeError):
        generate_decision_explanation(None)


def test_missing_decision_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        generate_decision_explanation({})



def test_sowing_eligibility_is_separate_from_alternative_crops():
    result = {
        "decision": "SWITCH TO COTTON",
        "economic_comparison": {
            "all_decisions": []
        },
        "crop_eligibility": {
            "paddy": {
                "eligible": True,
                "status": "primary_crop",
                "reason": "Primary crop"
            },
            "cotton": {
                "eligible": True,
                "status": "within_configured_window",
                "reason": "Within configured window"
            }
        },
        "sowing_eligibility": {
            "sow_today": {
                "eligible": True,
                "status": "eligible",
                "reason": "Moisture sufficient"
            },
            "wait": {
                "eligible": True,
                "status": "eligible",
                "reason": "Waiting is permitted"
            }
        }
    }

    explanation = generate_decision_explanation(
        result,
        primary_crop="paddy"
    )

    alternative_names = [
        item["crop"]
        for item in explanation["crop_eligibility"]
    ]

    assert alternative_names == ["cotton"]

    sowing_actions = [
        item["action"]
        for item in explanation["sowing_action_eligibility"]
    ]

    assert sowing_actions == ["sow_today", "wait"]

    assert all(
        item["crop"] == "paddy"
        for item in explanation["sowing_action_eligibility"]
    )