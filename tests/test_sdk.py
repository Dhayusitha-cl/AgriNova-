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

def test_sdk_is_reproducible_with_same_request():
    payload = {
        "location_id": "yavatmal",
        "crop_name": "cotton",
        "soil_type": "medium_black",
        "current_moisture_mm": 35.0,
        "rainfall_yesterday_mm": 12.0,
        "start_date": "2024-06-15",
        "num_simulations": 10,
        "days_to_simulate": 7,
        "random_seed": 123,
    }

    client = CropLogicClient()

    first = client.assess_sowing(**payload)
    second = client.assess_sowing(**payload)

    assert first.model_dump() == second.model_dump()

def test_sdk_trace_contains_request_and_calibration_provenance():
    result = CropLogicClient().assess_sowing(
        location_id="yavatmal",
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture_mm=35.0,
        rainfall_yesterday_mm=12.0,
        start_date="2024-06-15",
        num_simulations=10,
        days_to_simulate=7,
        random_seed=123,
    )

    trace = result.trace

    assert trace.location_id == "yavatmal"
    assert trace.calibration_schema_version == "1.0"
    assert trace.calibration_artifact_type == "rainfall_calibration"
    assert len(trace.calibration_artifact_id) == 64
    assert trace.start_date.isoformat() == "2024-06-15"
    assert trace.random_seed == 123
    assert trace.num_simulations == 10
    assert trace.days_to_simulate == 7
    assert trace.crop_name == "cotton"
    assert trace.soil_type == "medium_black"
    assert trace.initial_rainfall_state in {"dry", "drizzle", "rain"}

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
            start_date="2024-06-15",
            rainfall_yesterday_mm=12.0,
        )




def test_sdk_and_api_return_same_public_decision_contract():
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
    assert sdk_result.economic_comparison.model_dump() == (
        api_result["economic_comparison"]
    )

    assert sdk_result.germ_prob_today == api_result["germ_prob_today"]
    assert sdk_result.germ_prob_wait == api_result["germ_prob_wait"]
    assert sdk_result.germ_prob_soybean == api_result["germ_prob_soybean"]
    assert sdk_result.confidence == api_result["confidence"]

    assert sdk_result.current_moisture == api_result["current_moisture"]
    assert (
        sdk_result.min_moisture_required
        == api_result["min_moisture_required"]
    )
    assert (
        sdk_result.initial_rainfall_state
        == api_result["initial_rainfall_state"]
    )
    assert sdk_result.num_simulations == api_result["num_simulations"]
    assert sdk_result.days_to_simulate == api_result["days_to_simulate"]

    assert sdk_result.assumptions.model_dump() == api_result["assumptions"]
    assert sdk_result.trace.model_dump(mode="json") == api_result["trace"]

    # These are API diagnostics, not part of the SDK DecisionResult contract.
    assert "soil_moisture_today" in api_result
    assert "soil_moisture_wait" in api_result

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

def test_sdk_rejects_missing_start_date():
    with pytest.raises(TypeError, match="start_date"):
        CropLogicClient().assess_sowing(
            location_id="yavatmal",
            crop_name="cotton",
            soil_type="medium_black",
            current_moisture_mm=35.0,
            rainfall_yesterday_mm=12.0,
        )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_moisture_mm", float("nan")),
        ("current_moisture_mm", float("inf")),
        ("current_moisture_mm", float("-inf")),
        ("rainfall_yesterday_mm", float("nan")),
        ("rainfall_yesterday_mm", float("inf")),
        ("rainfall_yesterday_mm", float("-inf")),
    ],
)
def test_sdk_rejects_non_finite_numeric_inputs(field, value):
    client = CropLogicClient()

    kwargs = {
        "location_id": "yavatmal",
        "crop_name": "cotton",
        "soil_type": "medium_black",
        "current_moisture_mm": 35,
        "rainfall_yesterday_mm": 12,
        "start_date": "2024-06-15",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        client.assess_sowing(**kwargs)