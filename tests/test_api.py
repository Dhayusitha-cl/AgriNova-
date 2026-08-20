from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def valid_payload():
    return {
        "crop_name": "cotton",
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
    assert "transition" in response.json()["detail"].lower()