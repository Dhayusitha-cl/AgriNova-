from datetime import date, datetime

import pytest

from croplogic_saathi.client import CropLogicClient
from src.forecast import DailyForecast, WeatherForecast


def valid_client_kwargs():
    return {
        "location_id": "yavatmal",
        "crop_name": "cotton",
        "soil_type": "medium_black",
        "current_moisture_mm": 35.0,
        "rainfall_yesterday_mm": 12.0,
        "start_date": date(2024, 6, 15),
        "num_simulations": 10,
        "days_to_simulate": 7,
        "random_seed": 42,
    }


def test_client_assess_sowing_success():
    client = CropLogicClient()

    result = client.assess_sowing(
        **valid_client_kwargs()
    )

    assert result.decision
    assert 0.0 <= result.germ_prob_today <= 1.0
    assert 0.0 <= result.germ_prob_wait <= 1.0
    assert 0.0 <= result.germ_prob_soybean <= 1.0

    assert result.trace.location_id == "yavatmal"
    assert result.trace.calibration_artifact_id
    assert result.trace.calibration_artifact_type == "rainfall_calibration"

    assert result.trace.current_moisture_mm == 35.0
    assert result.trace.rainfall_yesterday_mm == 12.0
    assert result.trace.forecast_used is False
    assert result.trace.forecast_source is None
    assert result.trace.forecast_issued_at is None
    assert result.trace.forecast_weight is None


def test_client_accepts_coordinate_location(monkeypatch):
    monkeypatch.setenv(
        "CROPLOGIC_IMD_RAINFALL_DATASET",
        "data/raw/RF25_ind2024_rfp25.nc",
    )

    kwargs = valid_client_kwargs()
    kwargs.pop("location_id")

    kwargs["latitude"] = 20.50
    kwargs["longitude"] = 78.25

    client = CropLogicClient()

    result = client.assess_sowing(**kwargs)

    assert result.trace.location_id is None
    assert result.trace.latitude == 20.50
    assert result.trace.longitude == 78.25

    assert result.trace.climate_source == "imd_gridded_rainfall"
    assert result.trace.climate_grid_key == (
        "imd_gridded_rainfall:20.50:78.25"
    )


def test_client_rejects_mixed_location_sources():
    kwargs = valid_client_kwargs()

    kwargs["latitude"] = 20.50
    kwargs["longitude"] = 78.25

    client = CropLogicClient()

    with pytest.raises(ValueError, match="either location_id"):
        client.assess_sowing(**kwargs)


def test_client_requires_location():
    kwargs = valid_client_kwargs()

    kwargs.pop("location_id")

    client = CropLogicClient()

    with pytest.raises(ValueError, match="location_id"):
        client.assess_sowing(**kwargs)


def test_client_rejects_unknown_location():
    kwargs = valid_client_kwargs()
    kwargs["location_id"] = "unknown_location"

    client = CropLogicClient()

    with pytest.raises(ValueError):
        client.assess_sowing(**kwargs)


def test_client_accepts_forecast():
    kwargs = valid_client_kwargs()

    kwargs["start_date"] = date(2024, 7, 1)

    kwargs["forecast"] = WeatherForecast(
        source="test-provider",
        issued_at=datetime.fromisoformat(
            "2024-07-01T06:00:00+00:00"
        ),
        daily=tuple(
            DailyForecast(
                forecast_date=date(2024, 7, day),
                rainfall_mm=12.0,
                precipitation_probability=0.8,
            )
            for day in range(2, 22)
        ),
    )

    kwargs["forecast_weight"] = 0.5

    client = CropLogicClient()

    result = client.assess_sowing(**kwargs)

    assert result.decision
    assert result.trace.forecast_used is True
    assert result.trace.forecast_source == "test-provider"
    assert result.trace.forecast_issued_at == datetime.fromisoformat(
        "2024-07-01T06:00:00+00:00"
    )
    assert result.trace.forecast_weight == 0.5

def test_client_rejects_forecast_without_weight():
    kwargs = valid_client_kwargs()

    kwargs["start_date"] = date(2024, 7, 1)

    kwargs["forecast"] = WeatherForecast(
        source="test-provider",
        issued_at=datetime.fromisoformat(
            "2024-07-01T06:00:00+00:00"
        ),
        daily=tuple(
            DailyForecast(
                forecast_date=date(2024, 7, day),
                rainfall_mm=12.0,
                precipitation_probability=0.8,
            )
            for day in range(2, 22)
        ),
    )

    client = CropLogicClient()

    with pytest.raises(ValueError, match="forecast_weight"):
        client.assess_sowing(**kwargs)


def test_client_rejects_weight_without_forecast():
    kwargs = valid_client_kwargs()

    kwargs["forecast_weight"] = 0.5

    client = CropLogicClient()

    with pytest.raises(ValueError, match="forecast"):
        client.assess_sowing(**kwargs)