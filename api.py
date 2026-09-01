import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date

logger = logging.getLogger("croplogic_saathi")

from src.crop_data import crops
from src.soil_data import soils
from src.decision_engine import make_decision


app = FastAPI(
    title="CropLogic-Saathi API",
    version="1.0.0",
    description="API for the CropLogic-Saathi pre-sowing decision engine",
)


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class DecisionRequest(BaseModel):
    crop_name: str = Field(min_length=1, max_length=50)
    soil_type: str = Field(min_length=1, max_length=50)
    current_moisture_mm: float = Field(ge=0, le=500)
    rainfall_yesterday_mm: float = Field(ge=0, le=1000)
    transition_matrix: list[list[float]] | None = None
    start_date: date | None = None
    num_simulations: int = Field(default=500, ge=0, le=10000)
    days_to_simulate: int = Field(default=7, ge=1, le=30)


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
        raise HTTPException(
            status_code=404,
            detail=f"Unknown crop: {crop_name}",
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
        raise HTTPException(
            status_code=404,
            detail=f"Unknown soil type: {soil_type}",
        )

    return {
        "soil_type": soil_type,
        "data": soils[soil_type],
    }


# ---------------------------------------------------------
# API 6 — SOWING DECISION
# ---------------------------------------------------------

@app.post("/api/v1/decision")
def decision(request: DecisionRequest):

    # -----------------------------------------------------
    # Basic input validation
    # -----------------------------------------------------

    if request.crop_name not in crops:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop: {request.crop_name}",
        )

    if request.soil_type not in soils:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown soil type: {request.soil_type}",
        )

    if request.transition_matrix is not None:

        if len(request.transition_matrix) != 3:
            raise HTTPException(
                status_code=400,
                detail="Transition matrix must contain 3 rows.",
            )

        for row in request.transition_matrix:

            if len(row) != 3:
                raise HTTPException(
                    status_code=400,
                    detail="Each transition matrix row must contain 3 values.",
                )

            if any(value < 0 or value > 1 for value in row):
                raise HTTPException(
                    status_code=400,
                    detail="Transition probabilities must be between 0 and 1.",
                )

            if abs(sum(row) - 1.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail="Each transition matrix row must sum approximately to 1.",
                )

    elif request.start_date is None:

        raise HTTPException(
            status_code=400,
            detail="Either transition_matrix or start_date must be provided.",
        )

    # -----------------------------------------------------
    # Run decision engine
    # -----------------------------------------------------

    try:
        logger.info(
            "Decision request: crop=%s soil=%s simulations=%d days=%d",
            request.crop_name,
            request.soil_type,
            request.num_simulations,
            request.days_to_simulate,
        )

        result = make_decision(
            crop_name=request.crop_name,
            soil_type=request.soil_type,
            current_moisture_mm=request.current_moisture_mm,
            rainfall_yesterday_mm=request.rainfall_yesterday_mm,
            transition_matrix=request.transition_matrix,
            num_simulations=request.num_simulations,
            days_to_simulate=request.days_to_simulate,
            start_date=(
                request.start_date.isoformat()
                if request.start_date is not None
                else None
            ),
        )

    except ValueError as exc:
        logger.warning("Decision validation error: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Unexpected error while processing decision request")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing the decision.",
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
