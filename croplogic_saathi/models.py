from datetime import date

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
    location_id: str
    calibration_schema_version: str | None
    calibration_artifact_type: str | None
    calibration_artifact_id: str | None
    start_date: date | None
    random_seed: int
    num_simulations: int
    days_to_simulate: int
    initial_rainfall_state: str
    crop_name: str
    soil_type: str


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