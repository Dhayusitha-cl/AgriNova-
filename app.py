from fastapi import FastAPI
from pydantic import BaseModel

from src.decision_engine import make_decision
from src.weather_simulator import create_transition_matrix


app = FastAPI(
    title="AgriNova - CropLogic-Saathi",
    description="Probabilistic pre-sowing decision support API",
    version="1.0.0",
)


class SowingRiskRequest(BaseModel):
    crop: str
    soil_type: str
    current_moisture_mm: float
    rainfall_yesterday_mm: float
    week_number: int = 1


@app.get("/")
def root():
    return {
        "message": "AgriNova CropLogic-Saathi API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/assess-sowing-risk")
def assess_sowing_risk(request: SowingRiskRequest):

    transition_matrix = create_transition_matrix(
        week_number=request.week_number
    )

    result = make_decision(
        crop_name=request.crop,
        soil_type=request.soil_type,
        current_moisture_mm=request.current_moisture_mm,
        rainfall_yesterday_mm=request.rainfall_yesterday_mm,
        transition_matrix=transition_matrix,
    )

    # NumPy arrays must be converted before returning JSON
    result["trajectories"] = result["trajectories"].tolist()
    result["wait_simulations"] = result["wait_simulations"].tolist()

    return result