import numpy as np
import pandas as pd

from src.monte_carlo_weather import (
    generate_monte_carlo_scenarios,
    summarize_monte_carlo_scenarios,
)


TRANSITION_MATRIX = np.array([
    [0.871685, 0.113262, 0.015054],
    [0.261563, 0.604466, 0.133971],
    [0.097561, 0.542683, 0.359756],
])


def test_generate_monte_carlo_scenarios():

    rainfall_data = pd.DataFrame({
        "rainfall_state": [
            "dry",
            "dry",
            "drizzle",
            "drizzle",
            "rain",
            "rain",
        ],
        "rainfall_mm": [
            0.0,
            0.0,
            1.0,
            5.0,
            15.0,
            30.0,
        ],
    })

    scenarios = generate_monte_carlo_scenarios(
        transition_matrix=TRANSITION_MATRIX,
        num_days=14,
        num_simulations=100,
        initial_state="dry",
        rainfall_data=rainfall_data,
        random_seed=42,
    )

    assert len(scenarios) == 100

    for scenario in scenarios:
        assert len(scenario) == 14

        for day in scenario:
            assert day["rainfall_state"] in [
                "dry",
                "drizzle",
                "rain",
            ]

            assert day["rainfall_mm"] >= 0


def test_monte_carlo_is_reproducible():

    rainfall_data = pd.DataFrame({
        "rainfall_state": [
            "dry",
            "drizzle",
            "rain",
        ],
        "rainfall_mm": [
            0.0,
            5.0,
            20.0,
        ],
    })

    scenarios_1 = generate_monte_carlo_scenarios(
        TRANSITION_MATRIX,
        num_days=10,
        num_simulations=20,
        rainfall_data=rainfall_data,
        random_seed=123,
    )

    scenarios_2 = generate_monte_carlo_scenarios(
        TRANSITION_MATRIX,
        num_days=10,
        num_simulations=20,
        rainfall_data=rainfall_data,
        random_seed=123,
    )

    assert scenarios_1 == scenarios_2


def test_summarize_monte_carlo_scenarios():

    scenarios = [
        [
            {"rainfall_mm": 10.0},
            {"rainfall_mm": 20.0},
        ],
        [
            {"rainfall_mm": 5.0},
            {"rainfall_mm": 15.0},
        ],
    ]

    summary = summarize_monte_carlo_scenarios(scenarios)

    assert summary["num_simulations"] == 2
    assert summary["min_total_mm"] == 20.0
    assert summary["max_total_mm"] == 30.0
    assert summary["mean_total_mm"] == 25.0
def test_calibrated_rainfall_scenario_structure():
    from src.weather_simulator import (
        generate_calibrated_rainfall_scenario,
    )

    scenario = generate_calibrated_rainfall_scenario(
        start_date="2024-07-01",
        num_days=14,
        initial_state="drizzle",
        random_seed=42,
    )

    assert len(scenario) == 14

    dates = [
        pd.Timestamp(day["date"])
        for day in scenario
    ]

    assert dates[0] == pd.Timestamp("2024-07-01")

    assert dates[-1] == pd.Timestamp("2024-07-14")

    for day in scenario:
        assert day["rainfall_state"] in [
            "dry",
            "drizzle",
            "rain",
        ]

        assert day["rainfall_mm"] >= 0


def test_calibrated_rainfall_scenario_is_reproducible():
    from src.weather_simulator import (
        generate_calibrated_rainfall_scenario,
    )

    scenario_1 = generate_calibrated_rainfall_scenario(
        start_date="2024-07-01",
        num_days=14,
        initial_state="drizzle",
        random_seed=42,
    )

    scenario_2 = generate_calibrated_rainfall_scenario(
        start_date="2024-07-01",
        num_days=14,
        initial_state="drizzle",
        random_seed=42,
    )

    assert scenario_1 == scenario_2