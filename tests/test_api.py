from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_assess_sowing_risk_endpoint_exists():
    response = client.post(
        "/assess-sowing-risk",
        json={
            # IMPORTANT:
            # Replace these fields with the exact fields
            # required by your current API schema.
        },
    )

    assert response.status_code != 404