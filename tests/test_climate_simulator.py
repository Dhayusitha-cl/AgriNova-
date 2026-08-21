import numpy as np
import pytest

from src.climate_simulator import (
    calculate_et_with_warming,
    get_climate_impact_summary,
    run_climate_projection,
    simulate_climate_change,
)


def transition_matrix():
    return np.array([
        [0.75, 0.18, 0.07],
        [0.55, 0.30, 0.15],
        [0.40, 0.35, 0.25],
    ])


def test_climate_change_matrix_is_normalized():
    adjusted = simulate_climate_change(
        transition_matrix()
    )

    assert adjusted.shape == (3, 3)
    assert np.all(adjusted >= 0)
    assert np.allclose(
        adjusted.sum(axis=1),
        1.0,
    )


def test_warming_increases_et():
    current = calculate_et_with_warming(
        32,
        24,
        0,
    )

    warmed = calculate_et_with_warming(
        32,
        24,
        2,
    )

    assert warmed > current


def test_climate_projection_returns_scenarios():
    results = run_climate_projection(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture=35,
        rainfall_yesterday=10,
        transition_matrix=transition_matrix(),
        num_simulations=20,
        random_seed=42,
    )

    assert set(results) == {
        "Current",
        "2030",
        "2050",
    }


def test_climate_probabilities_are_valid():
    results = run_climate_projection(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture=35,
        rainfall_yesterday=10,
        transition_matrix=transition_matrix(),
        num_simulations=20,
        random_seed=42,
    )

    for result in results.values():
        probability = result[
            "germination_probability"
        ]

        assert 0 <= probability <= 1


def test_climate_summary():
    results = run_climate_projection(
        crop_name="cotton",
        soil_type="medium_black",
        current_moisture=35,
        rainfall_yesterday=10,
        transition_matrix=transition_matrix(),
        num_simulations=20,
        random_seed=42,
    )

    summary = get_climate_impact_summary(
        results
    )

    assert "Current establishment probability" in summary
    assert "2030 scenario" in summary
    assert "2050 scenario" in summary


def test_invalid_crop():
    with pytest.raises(ValueError):
        run_climate_projection(
            crop_name="banana",
            soil_type="medium_black",
            current_moisture=35,
            rainfall_yesterday=10,
            transition_matrix=transition_matrix(),
            num_simulations=10,
        )


def test_invalid_soil():
    with pytest.raises(ValueError):
        run_climate_projection(
            crop_name="cotton",
            soil_type="unknown_soil",
            current_moisture=35,
            rainfall_yesterday=10,
            transition_matrix=transition_matrix(),
            num_simulations=10,
        )