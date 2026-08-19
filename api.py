from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.decision_engine import make_decision
from src.crop_data import crops
from src.soil_data import soils


app = FastAPI(
    title="CropLogic-Saathi API",
    version="1.0.0",
    description="API for the CropLogic-Saathi pre-sowing decision engine"
)


class DecisionRequest(BaseModel):
    crop_name: str
    soil_type: str
    current_moisture_mm: float = Field(ge=0)
    rainfall_yesterday_mm: float = Field(ge=0)
    transition_matrix: list[list[float]]
    num_simulations: int = Field(default=500, gt=0)
    days_to_simulate: int = Field(default=7, gt=0)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "CropLogic-Saathi API"
    }


@app.post("/api/v1/decision")
def decision(request: DecisionRequest):

    if request.crop_name not in crops:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop: {request.crop_name}"
        )

    if request.soil_type not in soils:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown soil type: {request.soil_type}"
        )

    if len(request.transition_matrix) != 3:
        raise HTTPException(
            status_code=400,
            detail="Transition matrix must contain 3 rows."
        )

    for row in request.transition_matrix:
        if len(row) != 3:
            raise HTTPException(
                status_code=400,
                detail="Each transition matrix row must contain 3 values."
            )

        if any(value < 0 or value > 1 for value in row):
            raise HTTPException(
                status_code=400,
                detail="Transition probabilities must be between 0 and 1."
            )

        if abs(sum(row) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="Each transition matrix row must sum approximately to 1."
            )

    result = make_decision(
        crop_name=request.crop_name,
        soil_type=request.soil_type,
        current_moisture_mm=request.current_moisture_mm,
        rainfall_yesterday_mm=request.rainfall_yesterday_mm,
        transition_matrix=request.transition_matrix,
        num_simulations=request.num_simulations,
        days_to_simulate=request.days_to_simulate
    )

    return {
        "decision": result["decision"],
        "germ_prob_today": result["germ_prob_today"],
        "germ_prob_wait": result["germ_prob_wait"],
        "germ_prob_soybean": result["germ_prob_soybean"],
        "confidence": result["confidence"],
        "current_moisture": result["current_moisture"],
        "min_moisture_required": result["min_moisture_required"]
    }