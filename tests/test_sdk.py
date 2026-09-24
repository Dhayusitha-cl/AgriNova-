import pytest
from fastapi.testclient import TestClient

from api import app
from croplogic_saathi import CropLogicClient, DecisionResult

def test_sdk_returns_typed_decision_result():
    client = CropLogicClient()

    result = client.assess_sowing(
        location_id="yavatmal",
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=35.0,
        rainfall_yesterday_mm=12.0,
        start_date="2024-06-15",
        num_simulations=10,
        days_to_simulate=7,
    )

    assert isinstance(result, DecisionResult)
    assert result.decision in {"SOW TODAY", "WAIT", "SWITCH CROP"}


def test_sdk_probability_fields_are_bounded():
    result = CropLogicClient().assess_sowing(
        location_id="yavatmal",
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=35.0,
        rainfall_yesterday_mm=12.0,
        start_date="2024-06-15",
        num_simulations=10,
        days_to_simulate=7,
    )

    for probability in (
        result.germ_prob_today,
        result.germ_prob_wait,
        result.germ_prob_soybean,
        result.confidence,
    ):
        assert 0.0 <= probability <= 1.0


def test_sdk_economic_comparison_is_typed():
    result = CropLogicClient().assess_sowing(
        location_id="yavatmal",
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=35.0,
        rainfall_yesterday_mm=12.0,
        start_date="2024-06-15",
        num_simulations=10,
        days_to_simulate=7,
    )

    assert result.economic_comparison.sow_today.decision == "Sow Today"
    assert result.economic_comparison.wait.decision == "Wait 5 Days"
    assert result.economic_comparison.switch.decision == "Switch to Soybean"


def test_sdk_rejects_unknown_location():
    with pytest.raises(ValueError, match="No calibration artifact"):
        CropLogicClient().assess_sowing(
            location_id="unknown_location",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
            start_date="2024-06-15",
            num_simulations=10,
            days_to_simulate=7,
        )


def test_sdk_rejects_empty_location():
    with pytest.raises(ValueError, match="Calibration location must not be empty"):
        CropLogicClient().assess_sowing(
            location_id="   ",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
        )




def test_sdk_and_api_return_same_decision():
    payload = {
        "location_id": "yavatmal",
        "crop_name": "cotton",
        "soil_type": "medium_black",
        "current_moisture_mm": 35.0,
        "rainfall_yesterday_mm": 12.0,
        "start_date": "2024-06-15",
        "num_simulations": 10,
        "days_to_simulate": 7,
    }

    sdk_result = CropLogicClient().assess_sowing(**payload)

    api_response = TestClient(app).post(
        "/api/v1/decision",
        json=payload,
    )

    assert api_response.status_code == 200
    api_result = api_response.json()

    assert sdk_result.decision == api_result["decision"]
    assert sdk_result.germ_prob_today == api_result["germ_prob_today"]
    assert sdk_result.germ_prob_wait == api_result["germ_prob_wait"]
    assert sdk_result.germ_prob_soybean == api_result["germ_prob_soybean"]
    assert sdk_result.confidence == api_result["confidence"]

def test_sdk_rejects_unknown_crop():
    with pytest.raises(ValueError, match="Unknown crop"):
        CropLogicClient().assess_sowing(
            location_id="yavatmal",
            crop_name="unknown_crop",
            soil_type="medium_black",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
            start_date="2024-06-15",
            num_simulations=10,
            days_to_simulate=7,
        )


def test_sdk_rejects_unknown_soil():
    with pytest.raises(ValueError, match="Unknown soil"):
        CropLogicClient().assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="unknown_soil",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
            start_date="2024-06-15",
            num_simulations=10,
            days_to_simulate=7,
        )


def test_sdk_rejects_invalid_simulation_count():
    with pytest.raises(ValueError, match="num_simulations"):
        CropLogicClient().assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
            start_date="2024-06-15",
            num_simulations=0,
            days_to_simulate=7,
        )


def test_sdk_rejects_invalid_simulation_horizon():
    with pytest.raises(ValueError, match="days_to_simulate"):
        CropLogicClient().assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
            start_date="2024-06-15",
            num_simulations=10,
            days_to_simulate=0,
        )

def test_rejects_negative_moisture():
    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=-1,
            rainfall_yesterday_mm=10,
            start_date="2024-06-15",
        )


def test_rejects_excessive_moisture():
    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=501,
            rainfall_yesterday_mm=10,
            start_date="2024-06-15",
        )


def test_rejects_negative_rainfall():
    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=20,
            rainfall_yesterday_mm=-1,
            start_date="2024-06-15",
        )


def test_rejects_excessive_rainfall():
    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=20,
            rainfall_yesterday_mm=1001,
            start_date="2024-06-15",
        )


def test_rejects_excessive_simulations():
    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=20,
            rainfall_yesterday_mm=10,
            start_date="2024-06-15",
            num_simulations=10001,
        )


def test_rejects_excessive_simulation_horizon():
    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=20,
            rainfall_yesterday_mm=10,
            start_date="2024-06-15",
            days_to_simulate=31,
        )
