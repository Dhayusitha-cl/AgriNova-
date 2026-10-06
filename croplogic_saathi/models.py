from datetime import date, datetime

from pydantic import BaseModel


class EconomicOutcome(BaseModel):
    decision: str
    expected_profit: float
    success_probability: float
    best_case_profit: float
    worst_case_profit: float
    risk_level: str
    advantage_over_others: float


class EconomicComparison(BaseModel):
    sow_today: EconomicOutcome
    wait: EconomicOutcome
    switch: EconomicOutcome
    best_decision: str
    best_profit: float
    all_decisions: list[EconomicOutcome]


class Assumptions(BaseModel):
    daily_et_mm: float
    wait_days: int
    economic_decision_policy: str
    confidence_definition: str
    simulation_note: str


class DecisionTrace(BaseModel):
    location_id: str | None = None

    latitude: float | None = None
    longitude: float | None = None

    climate_source: str | None = None
    climate_grid_key: str | None = None

    calibration_schema_version: str | None = None
    calibration_artifact_type: str | None = None
    calibration_artifact_id: str | None = None
    start_date: date | None = None

    random_seed: int
    num_simulations: int
    days_to_simulate: int
    initial_rainfall_state: str
    crop_name: str
    soil_type: str

    current_moisture_mm: float
    rainfall_yesterday_mm: float

    forecast_used: bool = False
    forecast_source: str | None = None
    forecast_issued_at: datetime | None = None
    forecast_weight: float | None = None


class DecisionResult(BaseModel):
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
    trace: DecisionTrace
    assumptions: Assumptions