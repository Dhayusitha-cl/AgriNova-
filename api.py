import logging
import math
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.calibration_artifact import ARTIFACT_SCHEMA_VERSION
from croplogic_saathi.models import (
    Assumptions,
    EconomicComparison,
    EconomicOutcome,
    DecisionTrace,
)

logger = logging.getLogger("croplogic_saathi")

from src.crop_data import crops
from src.soil_data import soils
from src.decision_engine import make_decision
from src.calibration_registry import get_calibration_artifact

app = FastAPI(
    title="CropLogic-Saathi API",
    version="1.0.0",
    description="API for the CropLogic-Saathi pre-sowing decision engine",
)

def _api_error(
    status_code: int,
    code: str,
    message: str,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
    )

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = []

    for error in exc.errors():
        errors.append(
            {
                "loc": list(error.get("loc", [])),
                "type": error.get("type", "validation_error"),
                "message": error.get("msg", "Invalid request."),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "Request validation failed.",
                "errors": errors,
            }
        },
    )

# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class DecisionRequest(BaseModel):
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
    transition_matrix: list[list[float]] | None = None
    start_date: date | None = None
    num_simulations: int = Field(default=500, ge=1, le=10000)
    days_to_simulate: int = Field(default=7, ge=1, le=30)
    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**31 - 1,
    )

class MoistureSummary(BaseModel):
    mean: list[float]
    min: list[float]
    max: list[float]

class DecisionResponse(BaseModel):
    decision: str
    economic_comparison: EconomicComparison
    germ_prob_today: float
    germ_prob_wait: float
    germ_prob_soybean: float
    confidence: float
    current_moisture: float
    min_moisture_required: float
    initial_rainfall_state: str
    num_simulations: int
    days_to_simulate: int
    assumptions: Assumptions
    soil_moisture_today: MoistureSummary
    soil_moisture_wait: MoistureSummary
    trace: DecisionTrace

# ---------------------------------------------------------
# API 1 — HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "CropLogic-Saathi API",
    }


# ---------------------------------------------------------
# API 2 — LIST CROPS
# ---------------------------------------------------------

@app.get("/api/v1/crops")
def get_crops():
    return {
        "crops": list(crops.keys()),
    }


# ---------------------------------------------------------
# API 3 — LIST SOIL TYPES
# ---------------------------------------------------------

@app.get("/api/v1/soils")
def get_soils():
    return {
        "soils": list(soils.keys()),
    }


# ---------------------------------------------------------
# API 4 — GET CROP DETAILS
# ---------------------------------------------------------

@app.get("/api/v1/crops/{crop_name}")
def get_crop(crop_name: str):

    if crop_name not in crops:
        raise _api_error(
            404,
            "UNKNOWN_CROP",
            f"Unknown crop: {crop_name}",
        )

    return {
        "crop_name": crop_name,
        "data": crops[crop_name],
    }


# ---------------------------------------------------------
# API 5 — GET SOIL DETAILS
# ---------------------------------------------------------

@app.get("/api/v1/soils/{soil_type}")
def get_soil(soil_type: str):

    if soil_type not in soils:
        raise _api_error(
            404,
            "UNKNOWN_SOIL",
            f"Unknown soil type: {soil_type}",
        )

    return {
        "soil_type": soil_type,
        "data": soils[soil_type],
    }


# ---------------------------------------------------------
# API 6 — SOWING DECISION
# ---------------------------------------------------------

@app.post(
    "/api/v1/decision",
    response_model=DecisionResponse,
)
def decision(request: DecisionRequest):

    # -----------------------------------------------------
    # Basic input validation
    # -----------------------------------------------------

    if request.crop_name not in crops:
        raise _api_error(
            400,
            "UNKNOWN_CROP",
            f"Unknown crop: {request.crop_name}",
        )

    if request.soil_type not in soils:
        raise _api_error(
            400,
            "UNKNOWN_SOIL",
            f"Unknown soil type: {request.soil_type}",
        )

    if request.transition_matrix is not None:

        if len(request.transition_matrix) != 3:
            raise _api_error(
                400,
                "INVALID_TRANSITION_MATRIX",
                "Transition matrix must contain 3 rows.",
            )

        for row in request.transition_matrix:

            if len(row) != 3:
                raise _api_error(
                    400,
                    "INVALID_TRANSITION_MATRIX",
                    "Each transition matrix row must contain 3 values."
                )

            if any(not math.isfinite(value) for value in row):
                raise _api_error(
                    400,
                    "INVALID_TRANSITION_MATRIX",
                    "Transition probabilities must be finite numbers.",
                )

            if any(value < 0 or value > 1 for value in row):
                raise _api_error(
                    400,
                    "INVALID_TRANSITION_MATRIX",
                    "Transition probabilities must be between 0 and 1."
                )

            if abs(sum(row) - 1.0) > 1e-5:
                raise _api_error(
                    400,
                    "INVALID_TRANSITION_MATRIX",
                    "Each transition matrix row must sum approximately to 1.",
                )

    elif request.start_date is None:

        raise _api_error(
            400,
            "MISSING_START_DATE",
            "Either transition_matrix or start_date must be provided.",
        )

    # -----------------------------------------------------
    # Run decision engine
    # -----------------------------------------------------

    try:
        logger.info(
            "Decision request: location=%s crop=%s soil=%s simulations=%d days=%d seed=%d",
            request.location_id,
            request.crop_name,
            request.soil_type,
            request.num_simulations,
            request.days_to_simulate,
            request.random_seed,
        )

        calibration_artifact = None

        if request.transition_matrix is None:
            try:
                calibration_artifact = get_calibration_artifact(
                    request.location_id
                )
            except (ValueError, FileNotFoundError) as exc:
                raise _api_error(
                    400,
                    "CALIBRATION_UNAVAILABLE",
                    str(exc),
                ) from exc

        result = make_decision(
            crop_name=request.crop_name,
            soil_type=request.soil_type,
            current_moisture_mm=request.current_moisture_mm,
            rainfall_yesterday_mm=request.rainfall_yesterday_mm,
            transition_matrix=request.transition_matrix,
            num_simulations=request.num_simulations,
            days_to_simulate=request.days_to_simulate,
            random_seed=request.random_seed,
            start_date=(
                request.start_date.isoformat()
                if request.start_date is not None
                else None
            ),
            calibration_artifact=calibration_artifact,
        )

        trace = DecisionTrace(
            location_id=request.location_id,
            calibration_schema_version=(
                ARTIFACT_SCHEMA_VERSION
                if calibration_artifact is not None
                else None
            ),
            calibration_artifact_type=(
                "rainfall_calibration"
                if calibration_artifact is not None
                else None
            ),
            calibration_artifact_id=(
                calibration_artifact.content_hash()
                if calibration_artifact is not None
                else None
            ),
            start_date=request.start_date,
            random_seed=request.random_seed,
            num_simulations=request.num_simulations,
            days_to_simulate=request.days_to_simulate,
            initial_rainfall_state=result["initial_rainfall_state"],
            crop_name=request.crop_name,
            soil_type=request.soil_type,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning("Decision validation error: %s", exc)
        raise _api_error(
            400,
            "DECISION_VALIDATION_ERROR",
            str(exc),
        ) from exc
    except Exception:
        logger.exception("Unexpected error while processing decision request")
        raise _api_error(
            500,
            "INTERNAL_ERROR",
            "Internal server error while processing the decision.",
        )

    # -----------------------------------------------------
    # Extract simulation results
    # -----------------------------------------------------

    trajectories = result["trajectories"]
    wait_simulations = result["wait_simulations"]

    # -----------------------------------------------------
    # API response
    # -----------------------------------------------------

    return {
        "decision": result["decision"],
        "economic_comparison": result["economic_comparison"],
        "germ_prob_today": result["germ_prob_today"],
        "germ_prob_wait": result["germ_prob_wait"],
        "germ_prob_soybean": result["germ_prob_soybean"],
        "confidence": result["confidence"],
        "current_moisture": result["current_moisture"],
        "min_moisture_required": result["min_moisture_required"],
        "initial_rainfall_state": result["initial_rainfall_state"],
        "num_simulations": result["num_simulations"],
        "days_to_simulate": result["days_to_simulate"],
        "assumptions": result["assumptions"],
        "trace": trace,

        "soil_moisture_today": {
            "mean": trajectories.mean(axis=0).tolist(),
            "min": trajectories.min(axis=0).tolist(),
            "max": trajectories.max(axis=0).tolist(),
        },

        "soil_moisture_wait": {
            "mean": wait_simulations.mean(axis=0).tolist(),
            "min": wait_simulations.min(axis=0).tolist(),
            "max": wait_simulations.max(axis=0).tolist(),
        },
    }
