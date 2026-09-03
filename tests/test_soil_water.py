import pytest

from src.soil_water import (
    simulate_soil_water,
    update_soil_water,
)


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



def test_update_soil_water_rejects_initial_water_above_field_capacity():

    with pytest.raises(ValueError):
        update_soil_water(
            initial_water_mm=61,
            rainfall_mm=0,
            et_mm=0,
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_negative_et():

    with pytest.raises(ValueError):
        update_soil_water(
            initial_water_mm=30,
            rainfall_mm=0,
            et_mm=-1,
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_negative_initial_water():

    with pytest.raises(ValueError):
        update_soil_water(
            initial_water_mm=-1,
            rainfall_mm=0,
            et_mm=0,
            soil_type="medium_black",
        )


def test_simulate_soil_water_rejects_empty_series():

    with pytest.raises(ValueError):
        simulate_soil_water(
            rainfall_series=[],
            et_series=[],
            soil_type="medium_black",
            initial_water_mm=30,
        )

def test_water_balance_is_conserved():

    result = update_soil_water(
        initial_water_mm=50,
        rainfall_mm=20,
        et_mm=5,
        soil_type="medium_black",
    )

    assert (
        result["initial_water_mm"]
        + result["rainfall_mm"]
        - result["excess_water_mm"]
        - result["et_mm"]
        - result["final_water_mm"]
    ) == 0

def test_water_balance_when_et_exceeds_available_water():

    result = update_soil_water(
        initial_water_mm=5,
        rainfall_mm=0,
        et_mm=20,
        soil_type="medium_black",
    )

    assert result["final_water_mm"] == 0
    assert result["excess_water_mm"] == 0
    assert result["initial_water_mm"] == 5
    assert result["rainfall_mm"] == 0

def test_zero_rainfall_zero_et_preserves_soil_water():

    result = update_soil_water(
        initial_water_mm=30,
        rainfall_mm=0,
        et_mm=0,
        soil_type="medium_black",
    )

    assert result["final_water_mm"] == 30
    assert result["excess_water_mm"] == 0

def test_soil_types_use_their_own_field_capacity():

    sandy = update_soil_water(
        initial_water_mm=10,
        rainfall_mm=20,
        et_mm=2,
        soil_type="sandy_loam",
    )

    medium = update_soil_water(
        initial_water_mm=10,
        rainfall_mm=20,
        et_mm=2,
        soil_type="medium_black",
    )

    deep = update_soil_water(
        initial_water_mm=10,
        rainfall_mm=20,
        et_mm=2,
        soil_type="deep_black",
    )

    assert sandy["field_capacity_mm"] == 36
    assert medium["field_capacity_mm"] == 60
    assert deep["field_capacity_mm"] == 72

    assert sandy["final_water_mm"] == 28
    assert medium["final_water_mm"] == 28
    assert deep["final_water_mm"] == 28

def test_soil_types_have_different_storage_limits():

    sandy = update_soil_water(
        initial_water_mm=30,
        rainfall_mm=20,
        et_mm=0,
        soil_type="sandy_loam",
    )

    medium = update_soil_water(
        initial_water_mm=30,
        rainfall_mm=20,
        et_mm=0,
        soil_type="medium_black",
    )

    deep = update_soil_water(
        initial_water_mm=30,
        rainfall_mm=20,
        et_mm=0,
        soil_type="deep_black",
    )

    assert sandy["final_water_mm"] == 36
    assert medium["final_water_mm"] == 50
    assert deep["final_water_mm"] == 50

def test_update_soil_water_rejects_nan_initial_water():

    with pytest.raises(ValueError, match="initial_water_mm must be a finite number"):
        update_soil_water(
            initial_water_mm=float("nan"),
            rainfall_mm=0,
            et_mm=0,
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_infinite_initial_water():

    with pytest.raises(ValueError, match="initial_water_mm must be a finite number"):
        update_soil_water(
            initial_water_mm=float("inf"),
            rainfall_mm=0,
            et_mm=0,
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_nan_rainfall():

    with pytest.raises(ValueError, match="rainfall_mm must be a finite number"):
        update_soil_water(
            initial_water_mm=30,
            rainfall_mm=float("nan"),
            et_mm=0,
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_infinite_rainfall():

    with pytest.raises(ValueError, match="rainfall_mm must be a finite number"):
        update_soil_water(
            initial_water_mm=30,
            rainfall_mm=float("inf"),
            et_mm=0,
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_nan_et():

    with pytest.raises(ValueError, match="et_mm must be a finite number"):
        update_soil_water(
            initial_water_mm=30,
            rainfall_mm=0,
            et_mm=float("nan"),
            soil_type="medium_black",
        )


def test_update_soil_water_rejects_infinite_et():

    with pytest.raises(ValueError, match="et_mm must be a finite number"):
        update_soil_water(
            initial_water_mm=30,
            rainfall_mm=0,
            et_mm=float("inf"),
            soil_type="medium_black",
        )

def test_simulate_soil_water_defaults_to_half_field_capacity():

    results = simulate_soil_water(
        rainfall_series=[0],
        et_series=[0],
        soil_type="medium_black",
    )

    assert results[0]["initial_water_mm"] == 30
    assert results[0]["final_water_mm"] == 30