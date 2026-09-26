from datetime import date

from pydantic import BaseModel, Field

from src.calibration_registry import get_calibration_artifact
from src.decision_engine import make_decision

from src.calibration_artifact import ARTIFACT_SCHEMA_VERSION
from .models import DecisionResult, DecisionTrace


class _DecisionRequest(BaseModel):
    location_id: str = Field(min_length=1, max_length=100)
    crop_name: str = Field(min_length=1, max_length=50)
    soil_type: str = Field(min_length=1, max_length=50)
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
    num_simulations: int = Field(default=500, ge=1, le=10000)
    days_to_simulate: int = Field(default=7, ge=1, le=30)
    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**31 - 1,
    )


class CropLogicClient:
    """In-process Python interface to CropLogic-Saathi."""

    def assess_sowing(
        self,
        *,
        location_id: str,
        crop_name: str,
        soil_type: str,
        current_moisture_mm: float,
        rainfall_yesterday_mm: float,
        start_date: str,
        num_simulations: int = 500,
        random_seed: int = 42,
        days_to_simulate: int = 7,
    ) -> DecisionResult:
        request = _DecisionRequest(
            location_id=location_id,
            crop_name=crop_name,
            soil_type=soil_type,
            current_moisture_mm=current_moisture_mm,
            rainfall_yesterday_mm=rainfall_yesterday_mm,
            start_date=start_date,
            num_simulations=num_simulations,
            days_to_simulate=days_to_simulate,
            random_seed=random_seed,
        )

        calibration_artifact = get_calibration_artifact(
            request.location_id
        )

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
        )

        trace = DecisionTrace(
            location_id=request.location_id,
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