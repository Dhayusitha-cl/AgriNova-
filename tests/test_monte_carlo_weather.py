import numpy as np
import pandas as pd
import pytest

from src.forecast import DailyForecast, WeatherForecast
from src.monte_carlo_weather import (
    generate_calibrated_monte_carlo_scenarios,
    generate_monte_carlo_scenarios,
    summarize_monte_carlo_scenarios,
)


TRANSITION_MATRIX = np.array([
    [0.871685, 0.113262, 0.015054],
    [0.261563, 0.604466, 0.133971],
    [0.097561, 0.542683, 0.359756],
])


def _make_forecast(
    start_date="2024-07-01",
    num_days=5,
    precipitation_probability=0.8,
    rainfall_mm=20.0,
):
    start = pd.Timestamp(start_date)

    daily = tuple(
        DailyForecast(
            forecast_date=(start + pd.Timedelta(days=day)).date(),
            rainfall_mm=rainfall_mm,
            precipitation_probability=precipitation_probability,
        )
        for day in range(num_days)
    )

    return WeatherForecast(
        source="test-provider",
        issued_at=pd.Timestamp(
            "2024-06-30"
        ).to_pydatetime(),
        daily=daily,
    )


def _make_rainfall_data():
    dates = pd.date_range(
        "2024-06-01",
        "2024-08-31",
        freq="D",
    )

    states = [
        "dry",
        "drizzle",
        "rain",
        "dry",
        "rain",
        "drizzle",
    ]

    rainfall_by_state = {
        "dry": 0.0,
        "drizzle": 5.0,
        "rain": 20.0,
    }

    return pd.DataFrame(
        {
            "date": dates,
            "rainfall_state": [
                states[index % len(states)]
                for index in range(len(dates))
            ],
            "rainfall_mm": [
                rainfall_by_state[
                    states[index % len(states)]
                ]
                for index in range(len(dates))
            ],
        }
    )
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

def test_calibrated_monte_carlo_forecast_is_reproducible():
    rainfall_data = _make_rainfall_data()
    forecast = _make_forecast()

    scenarios_1 = generate_calibrated_monte_carlo_scenarios(
        start_date="2024-07-01",
        num_days=5,
        num_simulations=50,
        initial_state="dry",
        random_seed=42,
        rainfall_data=rainfall_data,
        forecast=forecast,
        forecast_weight=0.5,
    )

    scenarios_2 = generate_calibrated_monte_carlo_scenarios(
        start_date="2024-07-01",
        num_days=5,
        num_simulations=50,
        initial_state="dry",
        random_seed=42,
        rainfall_data=rainfall_data,
        forecast=forecast,
        forecast_weight=0.5,
    )

    assert scenarios_1 == scenarios_2


def test_calibrated_monte_carlo_forecast_does_not_change_day_zero():
    rainfall_data = _make_rainfall_data()
    forecast = _make_forecast()

    scenarios = generate_calibrated_monte_carlo_scenarios(
        start_date="2024-07-01",
        num_days=5,
        num_simulations=50,
        initial_state="drizzle",
        random_seed=42,
        rainfall_data=rainfall_data,
        forecast=forecast,
        forecast_weight=1.0,
    )

    for scenario in scenarios:
        assert scenario[0]["rainfall_state"] == "drizzle"


def test_calibrated_monte_carlo_requires_forecast_weight():
    rainfall_data = _make_rainfall_data()
    forecast = _make_forecast()

    with pytest.raises(ValueError):
        generate_calibrated_monte_carlo_scenarios(
            start_date="2024-07-01",
            num_days=5,
            num_simulations=10,
            initial_state="dry",
            random_seed=42,
            rainfall_data=rainfall_data,
            forecast=forecast,
        )


def test_calibrated_monte_carlo_rejects_weight_without_forecast():
    rainfall_data = _make_rainfall_data()

    with pytest.raises(ValueError):
        generate_calibrated_monte_carlo_scenarios(
            start_date="2024-07-01",
            num_days=5,
            num_simulations=10,
            initial_state="dry",
            random_seed=42,
            rainfall_data=rainfall_data,
            forecast_weight=0.5,
        )


def test_calibrated_monte_carlo_rejects_invalid_forecast_type():
    rainfall_data = _make_rainfall_data()

    with pytest.raises(TypeError):
        generate_calibrated_monte_carlo_scenarios(
            start_date="2024-07-01",
            num_days=5,
            num_simulations=10,
            initial_state="dry",
            random_seed=42,
            rainfall_data=rainfall_data,
            forecast="not-a-weather-forecast",
            forecast_weight=0.5,
        )


def test_calibrated_monte_carlo_rejects_missing_forecast_date():
    rainfall_data = _make_rainfall_data()

    forecast = _make_forecast(
        start_date="2024-07-01",
        num_days=4,
    )

    with pytest.raises(ValueError):
        generate_calibrated_monte_carlo_scenarios(
            start_date="2024-07-01",
            num_days=5,
            num_simulations=10,
            initial_state="dry",
            random_seed=42,
            rainfall_data=rainfall_data,
            forecast=forecast,
            forecast_weight=0.5,
        )

def test_forecast_probability_changes_monte_carlo_distribution():
    rainfall_data = _make_rainfall_data()

    dry_forecast = _make_forecast(
        precipitation_probability=0.0,
    )

    wet_forecast = _make_forecast(
        precipitation_probability=1.0,
    )

    dry_scenarios = generate_calibrated_monte_carlo_scenarios(
        start_date="2024-07-01",
        num_days=5,
        num_simulations=500,
        initial_state="drizzle",
        random_seed=42,
        rainfall_data=rainfall_data,
        forecast=dry_forecast,
        forecast_weight=1.0,
    )

    wet_scenarios = generate_calibrated_monte_carlo_scenarios(
        start_date="2024-07-01",
        num_days=5,
        num_simulations=500,
        initial_state="drizzle",
        random_seed=42,
        rainfall_data=rainfall_data,
        forecast=wet_forecast,
        forecast_weight=1.0,
    )

    def wet_day_count(scenarios):
        return sum(
            day["rainfall_state"] in {"drizzle", "rain"}
            for scenario in scenarios
            for day in scenario[1:]
        )

    dry_wet_days = wet_day_count(dry_scenarios)
    wet_wet_days = wet_day_count(wet_scenarios)

    assert wet_wet_days > dry_wet_days