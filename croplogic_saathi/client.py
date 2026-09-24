from datetime import date

from pydantic import BaseModel, Field

from src.calibration_registry import get_calibration_artifact
from src.decision_engine import make_decision

from .models import DecisionResult


class _DecisionRequest(BaseModel):
    location_id: str = Field(min_length=1, max_length=100)
    crop_name: str = Field(min_length=1, max_length=50)
    soil_type: str = Field(min_length=1, max_length=50)
    current_moisture_mm: float = Field(ge=0, le=500)
    rainfall_yesterday_mm: float = Field(ge=0, le=1000)
    start_date: date | None = None
    num_simulations: int = Field(default=500, ge=1, le=10000)
    days_to_simulate: int = Field(default=7, ge=1, le=30)


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
        start_date: str | None = None,
        num_simulations: int = 500,
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
            days_to_simulate=request.days_to_simulate,
            start_date=(
                request.start_date.isoformat()
                if request.start_date is not None
                else None
            ),
            calibration_artifact=calibration_artifact,
        )

        return DecisionResult.model_validate(
            {
                key: result[key]
                for key in DecisionResult.model_fields
            }
        )