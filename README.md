# AgriNova - CropLogic-Saathi

## Overview

CropLogic-Saathi is a probabilistic pre-sowing decision engine for rainfed agriculture.
It estimates crop-establishment risk under uncertain rainfall and compares SOW TODAY, WAIT, and SWITCH CROP.

The system is designed as a backend decision engine and API/SDK component that existing agricultural applications can call.

## Problem

Rainfed farmers must make pre-sowing decisions while future rainfall remains uncertain.
Weather information alone does not directly provide a crop-specific, risk- and economically-aware sowing decision.

CropLogic-Saathi addresses this decision gap by combining field observations, historical climate behaviour, current/recent weather, rainfall uncertainty, soil-water conditions, crop establishment requirements, and economic risk.

## Core workflow

UNCERTAINTY -> SCENARIOS -> PROBABILITY -> ECONOMIC RISK -> DECISION

1. Validate input
2. Establish current field and soil state
3. Use historical rainfall behaviour and current/recent weather information
4. Generate plausible rainfall scenarios
5. Simulate soil-water balance
6. Evaluate crop establishment conditions
7. Estimate outcomes with Monte Carlo simulation
8. Compare SOW, WAIT, and SWITCH economically
9. Return an explainable decision with assumptions and uncertainty

## Technical components

- Historical climate calibration
- Seasonal rainfall transition modelling
- Rainfall amount calibration
- Monte Carlo weather scenarios
- Soil-water balance
- Crop establishment evaluation
- Economic risk comparison
- Leakage-safe historical backtesting
- FastAPI decision API
- Versioned calibration artifacts

Probabilities produced by simulation are estimates, not guarantees.

## Repository structure

```text
AgriNova/
|-- api.py
|-- src/
|   |-- crop_data.py
|   |-- soil_data.py
|   |-- weather_simulator.py
|   |-- climate_simulator.py
|   |-- monte_carlo_weather.py
|   |-- soil_water.py
|   |-- crop_establishment.py
|   |-- economic_engine.py
|   |-- decision_engine.py
|   |-- backtesting.py
|   |-- rainfall_states.py
|   |-- rainfall_preprocessing.py
|   |-- rainfall_amount_calibration.py
|   |-- markov_calibration.py
|   |-- calibration_artifact.py
|   |-- calibration_registry.py
|   `-- build_calibration_artifact.py
|-- data/
|   `-- calibration/
|-- tests/
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## Calibration and location

Production calibration is selected using an explicit location_id.

location_id identifies a validated calibration artifact. It does not inherently mean a district, village, weather station, coordinate, or grid.

The calibration registry is the authoritative mapping between location_id values and versioned calibration artifacts.

Current configured production calibration:

- location_id: yavatmal
- artifact: data/calibration/yavatmal_rainfall_calibration_v1.json

The API does not silently fall back to Yavatmal when a production location is missing or unknown.

## API

The current FastAPI service exposes:

- GET /health
- GET /api/v1/crops
- GET /api/v1/soils
- GET /api/v1/crops/{crop_name}
- GET /api/v1/soils/{soil_type}
- POST /api/v1/decision

The decision endpoint accepts crop, soil, current moisture, recent rainfall, location_id, simulation settings, and either a start date or an explicit transition matrix.

When start_date is provided without a transition matrix, the API resolves the configured calibration artifact through the location registry.

## Installation

Create or activate the project environment and install production dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development and testing:

```powershell
python -m pip install -r requirements-dev.txt
```

Start the API locally with:

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## Example decision request

```json
{
  "location_id": "yavatmal",
  "crop_name": "cotton",
  "soil_type": "sandy_loam",
  "current_moisture_mm": 30,
  "rainfall_yesterday_mm": 12,
  "start_date": "2024-06-15",
  "num_simulations": 500,
  "days_to_simulate": 7
}
```

The response includes the selected decision, simulated establishment probabilities, economic comparison, moisture summaries, initial rainfall state, simulation settings, and modelling assumptions.

## Validation

The project includes leakage-safe historical backtesting and baseline comparison.

Historical validation currently covers 28 eligible decision dates from 2019-2024 for cotton within the evaluated sowing window.

Using a 14-day held-out future and 1000 simulations, the historical model-based binary outcome proxy produced:

- Weather-only baseline: 0.679 binary held-out score
- Rule-based baseline: 0.750 binary held-out score
- CropLogic-Saathi: 0.786 binary held-out score

For best-action agreement on the evaluated historical cases:

- Weather-only: 0.893
- Rule-based: 0.964
- CropLogic-Saathi: 1.000

These results are historical backtesting results using a model-based outcome proxy. They are not field validation, production impact measurements, or guarantees of future performance.

## Backtesting and leakage protection

Historical evaluation recreates the information that would have been available on each historical decision date.

Training data is restricted to information available before the decision date. Subsequent rainfall is held out and used only for evaluation.

Future observations are therefore not used to construct the historical decision-time calibration or initial state.

## Production architecture

The intended production flow is:

Existing agricultural application
  -> CropLogic-Saathi API
  -> Input validation
  -> location_id
  -> Calibration registry
  -> Versioned calibration artifact
  -> Decision engine
  -> Probabilistic and economic analysis
  -> Typed API response

CropLogic-Saathi is intended to provide the decision capability as an SDK/API component rather than requiring a separate farmer-facing application.

## Current production-readiness status

Implemented:

- FastAPI service and typed decision response
- Production dependency separation
- Versioned rainfall calibration artifact
- Explicit location_id calibration selection
- Calibrated Monte Carlo simulation path
- Leakage-safe historical backtesting
- Automated test suite
- API health endpoint

Remaining production work includes deployment infrastructure, production data refresh pipelines, authentication and rate limiting where required, observability, calibration expansion to additional locations, and pilot validation.

## Limitations

- Current production calibration is configured for Yavatmal.
- Soil condition inputs can contain subjective field observations.
- Soil and crop parameters are simplified representations of agricultural conditions.
- Rainfall behaviour can vary spatially and may change over time.
- Economic outcomes depend on uncertain input costs, yields, and prices.
- Farmer-specific financial constraints and risk preferences are not fully modelled.
- Live weather integration is not assumed unless explicitly configured and implemented.
- No field trials have yet established real-world impact.
- Extreme weather events can still cause establishment failure.

The system is decision support, not a guarantee of crop success.

## Development principles

- Prefer reproducible and explainable methods.
- Keep historical calibration separate from future evaluation data.
- Treat literature values, dataset values, assumptions, simulations, backtesting, and field validation as distinct evidence types.
- Avoid unnecessary UI and language-layer complexity in the core engine.
- Extend the existing architecture rather than rebuilding working components.
- Validate changes with targeted tests and the full test suite.
- Do not claim integrations, field impact, or capabilities that have not been implemented.

## Current status

CropLogic-Saathi is being developed as a real-time agricultural decision-support engine and API/SDK component for integration into existing agricultural platforms.
