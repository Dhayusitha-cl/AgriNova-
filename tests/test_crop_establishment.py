from src.crop_establishment import (
    calculate_moisture_percentage,
    evaluate_establishment,
)


def test_calculate_moisture_percentage():

    result = calculate_moisture_percentage(
        soil_water_mm=30,
        soil_type="medium_black",
    )

    assert result == 50.0


def test_moisture_percentage_at_field_capacity():

    result = calculate_moisture_percentage(
        soil_water_mm=60,
        soil_type="medium_black",
    )

    assert result == 100.0


def test_moisture_percentage_rejects_negative():

    try:
        calculate_moisture_percentage(
            -1,
            "medium_black",
        )
        assert False
    except ValueError:
        assert True


def test_establishment_succeeds_when_threshold_is_met():

    soil_results = [
        {
            "day": day,
            "final_water_mm": 30,
        }
        for day in range(1, 9)
    ]

    result = evaluate_establishment(
        soil_water_results=soil_results,
        crop="cotton",
        soil_type="medium_black",
    )

    assert result["establishment_success"] is True
    assert result["successful_days"] == 8


def test_establishment_fails_when_threshold_is_not_met():

    soil_results = [
        {
            "day": day,
            "final_water_mm": 10,
        }
        for day in range(1, 9)
    ]

    result = evaluate_establishment(
        soil_water_results=soil_results,
        crop="cotton",
        soil_type="medium_black",
    )

    assert result["establishment_success"] is False
    assert result["successful_days"] == 0


def test_establishment_requires_germination_period():

    soil_results = [
        {
            "day": day,
            "final_water_mm": 30,
        }
        for day in range(1, 5)
    ]

    try:
        evaluate_establishment(
            soil_water_results=soil_results,
            crop="cotton",
            soil_type="medium_black",
        )
        assert False
    except ValueError:
        assert True