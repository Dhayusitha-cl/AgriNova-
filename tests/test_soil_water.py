from src.soil_water import (
    calculate_effective_rainfall,
    simulate_soil_water,
    update_soil_water,
)


def test_effective_rainfall_rejects_negative_values():

    try:
        calculate_effective_rainfall(
            -1,
            "medium_black",
        )
        assert False
    except ValueError:
        assert True


def test_update_soil_water_increases_after_rain():

    result = update_soil_water(
        initial_water_mm=20,
        rainfall_mm=10,
        et_mm=2,
        soil_type="medium_black",
    )

    assert result["final_water_mm"] == 28


def test_soil_water_cannot_exceed_field_capacity():

    result = update_soil_water(
        initial_water_mm=55,
        rainfall_mm=20,
        et_mm=2,
        soil_type="medium_black",
    )

    assert result["final_water_mm"] == 58
    assert result["final_water_mm"] <= 60


def test_soil_water_cannot_be_negative():

    result = update_soil_water(
        initial_water_mm=5,
        rainfall_mm=0,
        et_mm=20,
        soil_type="medium_black",
    )

    assert result["final_water_mm"] == 0


def test_simulate_soil_water_length():

    rainfall = [10, 0, 5, 20]
    et = [4, 4, 4, 4]

    results = simulate_soil_water(
        rainfall_series=rainfall,
        et_series=et,
        soil_type="medium_black",
        initial_water_mm=30,
    )

    assert len(results) == 4


def test_simulate_soil_water_respects_field_capacity():

    rainfall = [100, 100, 100]
    et = [1, 1, 1]

    results = simulate_soil_water(
        rainfall_series=rainfall,
        et_series=et,
        soil_type="medium_black",
        initial_water_mm=30,
    )

    for result in results:
        assert result["final_water_mm"] <= 60
        assert result["final_water_mm"] >= 0