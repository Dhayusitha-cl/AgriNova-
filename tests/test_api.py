from fastapi.testclient import TestClient

from api import app
import pytest
import json

client = TestClient(app)


def valid_payload():
    return {
        "crop_name": "cotton",
        "location_id": "yavatmal",
        "soil_type": "medium_black",
        "current_moisture_mm": 35.0,
        "rainfall_yesterday_mm": 12.0,
        "transition_matrix": [
            [0.75, 0.18, 0.07],
            [0.55, 0.30, 0.15],
            [0.40, 0.35, 0.25]
        ],
        "num_simulations": 10,
        "days_to_simulate": 7
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_decision_success():
    response = client.post(
        "/api/v1/decision",
        json=valid_payload()
    )

    assert response.status_code == 200

    data = response.json()

    assert "decision" in data
    assert "germ_prob_today" in data
    assert "germ_prob_wait" in data
    assert "germ_prob_soybean" in data
    assert "confidence" in data
    assert "economic_comparison" in data

    economic = data["economic_comparison"]

    assert "sow_today" in economic
    assert "wait" in economic
    assert "switch" in economic
    assert "best_decision" in economic
    assert "best_profit" in economic

    assert "initial_rainfall_state" in data
    assert "num_simulations" in data
    assert "days_to_simulate" in data
    assert "assumptions" in data

    assert data["num_simulations"] == 10
    assert data["days_to_simulate"] == 7
    assert isinstance(data["assumptions"], dict)

def test_decision_response_schema():
    response = client.post(
        "/api/v1/decision",
        json=valid_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["decision"], str)
    assert isinstance(data["germ_prob_today"], float)
    assert isinstance(data["germ_prob_wait"], float)
    assert isinstance(data["germ_prob_soybean"], float)
    assert isinstance(data["confidence"], float)

    assert isinstance(data["soil_moisture_today"], dict)
    assert isinstance(data["soil_moisture_wait"], dict)

    assert set(data["soil_moisture_today"]) == {"mean", "min", "max"}
    assert set(data["soil_moisture_wait"]) == {"mean", "min", "max"}


def test_invalid_crop():
    payload = valid_payload()
    payload["crop_name"] = "banana"

    response = client.post(
        "/api/v1/decision",
        json=payload
    )

    assert response.status_code == 400


def test_invalid_soil():
    payload = valid_payload()
    payload["soil_type"] = "unknown_soil"

    response = client.post(
        "/api/v1/decision",
        json=payload
    )

    assert response.status_code == 400


def test_invalid_transition_matrix():
    payload = valid_payload()

    payload["transition_matrix"] = [
        [0.5, 0.2, 0.1],
        [0.55, 0.30, 0.15],
        [0.40, 0.35, 0.25]
    ]

    response = client.post(
        "/api/v1/decision",
        json=payload
    )

    assert response.status_code == 400


def test_negative_moisture():
    payload = valid_payload()
    payload["current_moisture_mm"] = -10

    response = client.post(
        "/api/v1/decision",
        json=payload
    )

    assert response.status_code == 422
def test_rounded_transition_matrix_returns_400():
    payload = valid_payload()

    payload["transition_matrix"] = [
        [0.8758, 0.1086, 0.0156],
        [0.2657, 0.6156, 0.1188],
        [0.1048, 0.5810, 0.3143]
    ]

    response = client.post(
        "/api/v1/decision",
        json=payload
    )

    assert response.status_code == 400
    detail = response.json()["detail"]

    assert detail["code"] == "INVALID_TRANSITION_MATRIX"
    assert "transition" in detail["message"].lower()


def test_unexpected_decision_error_returns_500(monkeypatch):
    def failing_decision(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr("api.make_decision", failing_decision)

    response = client.post(
        "/api/v1/decision",
        json=valid_payload()
    )

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "code": "INTERNAL_ERROR",
        "message": "Internal server error while processing the decision.",
    }

def test_decision_success_with_production_calibration():
    payload = valid_payload()

    payload.pop("transition_matrix")
    payload["start_date"] = "2024-06-15"

    response = client.post(
        "/api/v1/decision",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "decision" in data
    assert "economic_comparison" in data
    assert "germ_prob_today" in data
    assert "germ_prob_wait" in data
    assert "germ_prob_soybean" in data

def test_decision_rejects_unknown_location():
    payload = valid_payload()
    payload.pop("transition_matrix")
    payload["start_date"] = "2024-06-15"
    payload["location_id"] = "unknown_location"

    response = client.post("/api/v1/decision", json=payload)

    assert response.status_code == 400
    detail = response.json()["detail"]

    assert detail["code"] == "CALIBRATION_UNAVAILABLE"
    assert "No calibration artifact is configured" in detail["message"]


def test_decision_requires_location_for_production_calibration():
    payload = valid_payload()
    payload.pop("transition_matrix")
    payload["start_date"] = "2024-06-15"
    payload.pop("location_id")

    response = client.post("/api/v1/decision", json=payload)

    assert response.status_code == 422

def test_decision_economic_comparison_contract():
    response = client.post("/api/v1/decision", json=valid_payload())

    assert response.status_code == 200

    economic = response.json()["economic_comparison"]

    assert set(economic) == {
        "sow_today",
        "wait",
        "switch",
        "best_decision",
        "best_profit",
        "all_decisions",
    }

    for key in ("sow_today", "wait", "switch"):
        outcome = economic[key]

        assert set(outcome) == {
            "decision",
            "expected_profit",
            "success_probability",
            "best_case_profit",
            "worst_case_profit",
            "risk_level",
            "advantage_over_others",
        }

        assert 0.0 <= outcome["success_probability"] <= 1.0


def test_decision_assumptions_contract():
    response = client.post("/api/v1/decision", json=valid_payload())

    assert response.status_code == 200

    assumptions = response.json()["assumptions"]

    assert set(assumptions) == {
        "daily_et_mm",
        "wait_days",
        "economic_decision_policy",
        "confidence_definition",
        "simulation_note",
    }

    assert assumptions["daily_et_mm"] > 0
    assert assumptions["wait_days"] > 0


def test_decision_probability_fields_are_bounded():
    response = client.post("/api/v1/decision", json=valid_payload())

    assert response.status_code == 200

    body = response.json()

    for field in (
        "germ_prob_today",
        "germ_prob_wait",
        "germ_prob_soybean",
        "confidence",
    ):
        assert 0.0 <= body[field] <= 1.0

def test_invalid_crop():
    payload = valid_payload()
    payload["crop_name"] = "banana"

    response = client.post(
        "/api/v1/decision",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "UNKNOWN_CROP",
        "message": "Unknown crop: banana",
    }

def test_invalid_soil():
    payload = valid_payload()
    payload["soil_type"] = "unknown_soil"

    response = client.post(
        "/api/v1/decision",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "UNKNOWN_SOIL",
        "message": "Unknown soil type: unknown_soil",
    }

def test_invalid_transition_matrix():
    payload = valid_payload()

    payload["transition_matrix"] = [
        [0.5, 0.2, 0.1],
        [0.55, 0.30, 0.15],
        [0.40, 0.35, 0.25],
    ]

    response = client.post(
        "/api/v1/decision",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == (
        "INVALID_TRANSITION_MATRIX"
    )

def test_decision_requires_start_date_without_transition_matrix():
    payload = valid_payload()
    payload.pop("transition_matrix")
    payload.pop("start_date", None)

    response = client.post(
        "/api/v1/decision",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "MISSING_START_DATE",
        "message": "Either transition_matrix or start_date must be provided.",
    }

@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_transition_matrix_returns_400(bad_value):
    payload = valid_payload()
    payload["transition_matrix"][0][0] = bad_value

    response = client.post(
        "/api/v1/decision",
        content=json.dumps(payload, allow_nan=True),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400

    body = response.json()["detail"]

    assert body["code"] == "INVALID_TRANSITION_MATRIX"
    assert body["message"] == (
        "Transition probabilities must be finite numbers."
    )