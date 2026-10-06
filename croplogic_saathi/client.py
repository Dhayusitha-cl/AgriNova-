from datetime import date

from pydantic import BaseModel, Field, model_validator

from src.calibration_artifact import ARTIFACT_SCHEMA_VERSION
from src.calibration_registry import get_calibration_artifact
from src.climate_data_provider import create_configured_climate_data_provider
from src.decision_engine import make_decision
from src.forecast import WeatherForecast
from src.location import GeographicLocation
from src.runtime_calibration import (
    resolve_calibration_resolution_from_provider,
)

from .models import DecisionResult, DecisionTrace

class _DecisionRequest(BaseModel):
    location_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        allow_inf_nan=False,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        allow_inf_nan=False,
    )

    crop_name: str = Field(
        min_length=1,
        max_length=50,
    )

    soil_type: str = Field(
        min_length=1,
        max_length=50,
    )

    current_moisture_mm: float = Field(
        ge=0,
        le=500,
        allow_inf_nan=False,
    )

    rainfall_yesterday_mm: float = Field(
        ge=0,
        le=1000,
        allow_inf_nan=False,
    )

    start_date: date

    num_simulations: int = Field(
        default=500,
        ge=1,
        le=10000,
    )

    days_to_simulate: int = Field(
        default=7,
        ge=1,
        le=30,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**31 - 1,
    )

    forecast: WeatherForecast | None = None

    forecast_weight: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_location_source(self):
        has_location_id = self.location_id is not None
        has_latitude = self.latitude is not None
        has_longitude = self.longitude is not None

        if has_location_id and (has_latitude or has_longitude):
            raise ValueError(
                "Provide either location_id or latitude and longitude, "
                "not both."
            )

        if not has_location_id and not (has_latitude and has_longitude):
            raise ValueError(
                "Provide either location_id or both latitude and longitude."
            )

        return self

class CropLogicClient:
    """In-process Python interface to CropLogic-Saathi."""

    def assess_sowing(
        self,
        *,
        crop_name: str,
        soil_type: str,
        current_moisture_mm: float,
        rainfall_yesterday_mm: float,
        start_date: date,
        location_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        num_simulations: int = 500,
        random_seed: int = 42,
        days_to_simulate: int = 7,
        forecast: WeatherForecast | None = None,
        forecast_weight: float | None = None,
    ) -> DecisionResult:

        request = _DecisionRequest(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            crop_name=crop_name,
            soil_type=soil_type,
            current_moisture_mm=current_moisture_mm,
            rainfall_yesterday_mm=rainfall_yesterday_mm,
            start_date=start_date,
            num_simulations=num_simulations,
            random_seed=random_seed,
            days_to_simulate=days_to_simulate,
            forecast=forecast,
            forecast_weight=forecast_weight,
        )

        calibration_artifact = None
        climate_source = None
        climate_grid_key = None

        if request.location_id is not None:
            calibration_artifact = get_calibration_artifact(
                request.location_id
            )

        else:
            location = GeographicLocation(
                latitude=request.latitude,
                longitude=request.longitude,
            )

            climate_provider = create_configured_climate_data_provider()

            resolution = resolve_calibration_resolution_from_provider(
                location=location,
                climate_provider=climate_provider,
            )

            calibration_artifact = resolution.artifact
            climate_source = resolution.climate_grid.source
            climate_grid_key = resolution.climate_grid.key

        result = make_decision(
            crop_name=request.crop_name,
            soil_type=request.soil_type,
            current_moisture_mm=request.current_moisture_mm,
            rainfall_yesterday_mm=request.rainfall_yesterday_mm,
            transition_matrix=None,
            num_simulations=request.num_simulations,
            random_seed=request.random_seed,
            days_to_simulate=request.days_to_simulate,
            start_date=request.start_date.isoformat(),
            calibration_artifact=calibration_artifact,
            forecast=request.forecast,
            forecast_weight=request.forecast_weight,
        )

        trace = DecisionTrace(
            location_id=request.location_id,
            latitude=request.latitude,
            longitude=request.longitude,
            climate_source=climate_source,
            climate_grid_key=climate_grid_key,
            calibration_schema_version=ARTIFACT_SCHEMA_VERSION,
            calibration_artifact_type="rainfall_calibration",
            calibration_artifact_id=calibration_artifact.content_hash(),
            start_date=request.start_date,
            random_seed=request.random_seed,
            num_simulations=request.num_simulations,
            days_to_simulate=request.days_to_simulate,
            initial_rainfall_state=result["initial_rainfall_state"],
            crop_name=request.crop_name,
            soil_type=request.soil_type,
        )

        result["trace"] = trace

        return DecisionResult.model_validate(
            {
                key: result[key]
                for key in DecisionResult.model_fields
            }
        )