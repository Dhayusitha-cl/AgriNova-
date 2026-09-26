# AgriNova --- CropLogic-Saathi Project Context

> **Purpose:** Durable project-level source of truth for future
> development chats.
>
> **Current status:** Real-time agricultural decision-support SDK/API.
> This is no longer treated as a hackathon project or as a farmer-facing
> UI product.

------------------------------------------------------------------------

## 1. Project Identity

**Repository:** AgriNova\
**Core engine:** CropLogic-Saathi

CropLogic-Saathi is a **location-general, probabilistic pre-sowing
decision engine** designed to support rainfed agricultural decisions
under uncertain rainfall and current field conditions.

The engine compares:

-   **SOW NOW**
-   **WAIT**
-   **SWITCH CROP**, only when sufficiently justified by available crop,
    climate, soil and economic data.

The project is now being developed as a **reusable SDK/API component**
that can be integrated into existing agricultural applications/platforms
rather than as a standalone farmer-facing application.

The project owner is currently handling the complete project
development.

------------------------------------------------------------------------

## 2. Problem Definition

The core problem is a **decision gap**, not a lack of weather
information.

Farmers and agricultural platforms may already have forecasts,
advisories, historical climate information and local observations. The
missing layer is converting uncertain information plus current field
state into a **crop-specific, risk-aware and economically-aware
pre-sowing decision**.

The engine therefore does not attempt to replace:

-   weather services,
-   agricultural extension,
-   farmer knowledge,
-   existing agricultural applications,
-   or field measurements.

It combines available information into a structured decision-support
process.

------------------------------------------------------------------------

## 3. Core Solution

The central concept is:

**UNCERTAINTY → CLIMATE CONTEXT → CURRENT FIELD STATE → SCENARIOS →
PROBABILITY → ECONOMIC RISK → DECISION → OBSERVED OUTCOME → BACKTESTING
→ RECALIBRATION**

The decision is based on multiple information layers rather than a
single rainfall dataset.

### Regional / historical layer

Answers:

> What rainfall/climate behaviour is historically plausible for this
> geographic context?

Uses:

-   historical rainfall,
-   climate behaviour,
-   transition probabilities,
-   rainfall amount distributions,
-   calibration artifacts.

### Current / local layer

Answers:

> What is happening around this field now?

Uses:

-   recent rainfall,
-   current moisture estimate/state,
-   soil type,
-   farmer-observed field condition,
-   ball-test information.

### Near-term layer

Answers:

> What may happen soon?

Uses:

-   current/recent weather,
-   available forecast information.

The system therefore does **not** depend entirely on spatial rainfall
resolution.

------------------------------------------------------------------------

## 4. Important Spatial-Resolution Principle

Spatial rainfall data is a representation of **regional climate
behaviour**, not exact field truth.

Spatial resolution is therefore a **bounded uncertainty/representation
constraint**, not by itself a reason to make the project infeasible.

A farmer-specific decision can still incorporate more local information
through:

-   recent rainfall,
-   ball-test observation,
-   soil type,
-   current moisture/state,
-   and other validated field inputs.

The architecture should never claim that a coarse rainfall grid gives
exact rainfall or exact soil moisture for an individual field.

Correct framing:

> Regional climate data provides climate context; current local
> observations provide information about the field's present state.

Do not claim:

> Exact field-level rainfall or exact field-level soil moisture is
> known.

If ball-test observations are converted into moisture estimates, avoid
false precision. Prefer categorical or bounded uncertainty where
scientifically appropriate.

------------------------------------------------------------------------

## 5. Current High-Level Architecture

``` text
                    LOCATION
                       │
             lat/lon / geographic unit
                       │
                       ▼
              Climate-data layer
                       │
             historical rainfall
                       │
                       ▼
                 Calibration
                       │
              versioned artifact
                       │
                 persistent cache
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Historical climate          Current request
      behaviour                     │
                                    ├── recent rainfall
                                    ├── soil type
                                    ├── ball-test condition
                                    ├── current moisture/state
                                    ├── crop
                                    └── forecast
                                    │
                                    ▼
                           Current field state
                                    │
                                    ▼
                         Weather scenarios
                                    │
                                    ▼
                          Soil-water balance
                                    │
                                    ▼
                       Crop establishment model
                                    │
                                    ▼
                           Monte Carlo
                                    │
                                    ▼
                    Probability / risk distribution
                                    │
                                    ▼
                         Economic comparison
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                       SOW        WAIT       SWITCH
```

------------------------------------------------------------------------

## 6. Separation of Data/Calibration and Decision Execution

This separation is a major architectural principle.

### Data / calibration pipeline

``` text
Approved source
      ↓
Historical data
      ↓
Preprocessing
      ↓
Calibration
      ↓
Versioned artifact
      ↓
Persistent cache
      ↓
Periodic recalibration
```

### Decision pipeline

``` text
Cached calibration
      +
Recent rainfall
      +
Field observations
      +
Soil state
      +
Forecast
      ↓
Monte Carlo
      ↓
Soil-water simulation
      ↓
Crop establishment
      ↓
Economic comparison
      ↓
SOW / WAIT / SWITCH
```

External data-source downtime should **not normally break decisions for
locations that already have a valid cached calibration artifact**.

Bad architecture:

``` text
Farmer request → external data source → decision
```

Preferred architecture:

``` text
Data source → ingestion/calibration → persistent artifact

Farmer request → existing artifact + current inputs → decision
```

A new location without a valid calibration may require background
calibration before a defensible decision can be produced.

Never fabricate a calibration by silently using another location's
artifact.

------------------------------------------------------------------------

## 7. Location-General Design

Yavatmal is the **first validated/calibrated reference location**, not
the product boundary.

The intended architecture is:

``` text
Location
   ↓
Canonical geographic/climate unit
   ↓
Existing calibration?
   ├── YES → reuse cached artifact
   │
   └── NO → acquire historical data
                ↓
             preprocess
                ↓
             calibrate
                ↓
          validate/version
                ↓
              cache
                ↓
             decision
```

Multiple farmers sharing the same climate unit should reuse the same
calibration artifact.

Do not recalibrate separately for every farmer request.

The system should eventually support location inputs such as:

-   latitude/longitude,
-   district or geographic unit,
-   other canonical location identifiers,

but all should converge into a common climate/calibration layer.

Latitude/longitude is a geographic key for resolving the appropriate
climate data. It does not itself calculate rainfall.

------------------------------------------------------------------------

## 8. Calibration Artifacts

Calibration must be versioned and reproducible.

A calibration artifact should retain provenance such as:

-   spatial/climate unit,
-   data source,
-   source version/date,
-   historical date range,
-   calibration method/schema version,
-   rainfall states,
-   transition matrices,
-   rainfall distributions/samples,
-   artifact version,
-   content identity/hash.

Existing artifact content hashing and provenance mechanisms should be
preserved.

A location-specific calibration should never silently change underneath
a reproducibility test or historical evaluation.

------------------------------------------------------------------------

## 9. Backtesting Is a Core Lifecycle Component

Backtesting is not merely a final demo metric.

It is a central mechanism for evaluating and improving the
calibration/model.

Correct lifecycle:

``` text
Calibration
    ↓
Historical decision-date simulation
    ↓
Prediction
    ↓
Actual future rainfall/outcome
    ↓
Comparison
    ↓
Model/calibration evaluation
    ↓
Candidate recalibration/update
    ↓
Out-of-sample validation
    ↓
New validated version
```

Do not describe this as automatic self-learning unless an actual
automated learning pipeline is implemented.

The correct current concept is:

> **periodic evidence-driven recalibration and model updating.**

### Leakage rule

For a historical decision date, only information that would have been
available on that date may be used.

Do not use future rainfall to:

-   initialize the decision,
-   calibrate the model,
-   choose a model,
-   tune parameters,
-   or influence the prediction.

Future observations are used only for evaluation after the simulated
decision.

------------------------------------------------------------------------

## 10. Validation Question

The central scientific/engineering question is:

> **Given uncertain rainfall and current field conditions, does
> CropLogic-Saathi produce a more useful pre-sowing recommendation than
> simpler weather-only or rule-based approaches?**

Validation priorities:

1.  Historical leakage-safe backtesting
2.  Weather-only baseline
3.  Simple rule-based baseline
4.  CropLogic-Saathi comparison
5.  Probability calibration
6.  Decision-quality evaluation
7.  Economic comparison
8.  Cross-location evaluation
9.  Periodic recalibration
10. Eventually field validation

Simulation probabilities are estimates conditional on model/data
assumptions. They are not guarantees.

------------------------------------------------------------------------

## 11. Current Model Components

### Soil-water balance

Rainfall adds water; evapotranspiration removes water; soil properties
influence available moisture, infiltration and drainage.

### Climate/weather simulation

Historical rainfall behaviour is used to construct plausible future
rainfall scenarios.

A defensible stochastic method such as a Markov-chain approach can
represent rainfall state transitions.

Historical data describes climate behaviour; it is not treated as a
deterministic future forecast.

### Monte Carlo

Many plausible weather/soil trajectories are simulated over the decision
horizon.

The result is a probability/risk distribution rather than a single
deterministic prediction.

### Crop establishment

Simulated soil/weather conditions are compared against documented crop
establishment requirements.

Crop parameters must be clearly identified as literature values, dataset
values, assumptions, or calibrated values.

### Economic engine

The system compares the economic consequences of SOW, WAIT and SWITCH
using available:

-   input costs,
-   yield assumptions,
-   price assumptions,
-   waiting/opportunity costs,
-   establishment probabilities/risk.

Economic outputs are scenario-based and assumption-dependent.

### Decision engine

The decision engine combines physical establishment risk and economic
comparison to produce an explainable decision.

SWITCH should only be returned when alternative-crop data is
sufficiently justified.

------------------------------------------------------------------------

## 12. What the System Must Not Claim

Do not claim that CropLogic-Saathi:

-   guarantees crop success,
-   predicts exact rainfall,
-   knows exact field-level soil moisture,
-   eliminates crop failure,
-   automatically learns after every farmer request,
-   is universally valid across India without geographic validation,
-   is field-proven without field trials,
-   is more accurate merely because it uses more data,
-   is integrated with government/Kisan/agriculture platforms unless
    that integration is actually implemented,
-   has production live-weather integration unless actually implemented,
-   has validated every crop/soil/location combination.

The honest claim is:

> **CropLogic-Saathi is probabilistic decision support intended to
> reduce decision uncertainty, not eliminate agricultural uncertainty.**

------------------------------------------------------------------------

## 13. Major Technical Risks and Their Proper Interpretation

### Model validity

A sophisticated simulation can still be wrong if its assumptions do not
represent reality.

Monte Carlo does not fix a bad underlying model.

### Calibration quality

Poor rainfall calibration produces poor future scenarios.

This is why calibration artifacts must be evaluated through leakage-safe
backtesting.

### Climate nonstationarity

Historical climate behaviour may change.

Therefore calibration should be periodically reevaluated and updated.

### Spatial representativeness

Regional/grid rainfall does not equal exact farm rainfall.

Mitigation:

-   recent local rainfall,
-   farmer field observations,
-   soil type,
-   current state,
-   explicit uncertainty.

### Data quality

Historical datasets can contain:

-   missing values,
-   outliers,
-   revisions,
-   station changes,
-   interpolation effects,
-   inconsistent coverage.

Data preprocessing must be reproducible and validated.

### Cold start

A new geographic unit without a cached calibration requires historical
data acquisition and calibration.

If data is unavailable, do not fabricate a decision.

### External-source availability

External government/data-source outages should not break
already-calibrated locations.

Use adapters and persistent artifacts.

### Economic uncertainty

Prices, yields, costs and farmer-specific risk tolerance are uncertain.

Economic results must be labelled as assumptions/scenarios rather than
exact financial truth.

### Field-state uncertainty

Ball-test and other farmer observations are useful but imperfect.

Avoid false precision.

### Extreme events

Events outside the historical distribution may not be represented well.

The model cannot guarantee protection against them.

### Field validation

No field validation should be claimed until actual field
trials/observational validation are performed.

------------------------------------------------------------------------

## 14. Data-Source Strategy

Government/official sources such as IMD are preferred where appropriate
and permitted.

The system should use a source-adapter layer rather than coupling the
decision engine directly to one external API.

Potential architecture:

``` text
IMD / approved source
        ↓
source adapter
        ↓
canonical rainfall schema
        ↓
preprocessing
        ↓
calibration
        ↓
artifact/cache
        ↓
decision engine
```

The exact historical rainfall product, programmatic access method,
licensing/usage constraints, update cadence and spatial/temporal
resolution must be verified before implementation.

Do not assume an official catalog entry automatically provides the exact
daily coordinate-level dataset needed.

------------------------------------------------------------------------

## 15. SDK/API Direction

CropLogic-Saathi should remain usable as an integration component.

Conceptual API:

``` text
POST /api/v1/decision
```

Potential inputs include:

-   location
-   crop
-   soil
-   recent rainfall
-   current field state
-   current moisture/state
-   start date
-   simulation configuration where appropriate.

Output should expose useful public decision information such as:

-   decision,
-   establishment probabilities/risk,
-   economic comparison,
-   confidence/uncertainty information,
-   assumptions,
-   provenance/trace information.

Internal Monte Carlo trajectories should not automatically become part
of the public SDK contract.

The SDK should abstract production calibration by location rather than
requiring callers to provide raw transition matrices.

------------------------------------------------------------------------

## 16. Reproducibility and Provenance

A production decision should be reproducible from its relevant inputs
and model/calibration identity.

Important provenance includes:

-   location,
-   calibration artifact identity/hash,
-   calibration schema/version,
-   start date,
-   random seed,
-   simulation count,
-   simulation horizon,
-   initial rainfall state,
-   crop,
-   soil type.

Same inputs + same calibration artifact + same seed should produce
reproducible results within the defined software/model version.

------------------------------------------------------------------------

## 17. Current Repository Architecture

``` text
AgriNova/
├── app.py
├── api.py
├── README.md
├── PROJECT_CONTEXT.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── calibration/
├── src/
│   ├── crop_data.py
│   ├── soil_data.py
│   ├── weather_simulator.py
│   ├── climate_simulator.py
│   ├── decision_engine.py
│   ├── economic_engine.py
│   ├── backtesting.py
│   ├── monte_carlo_weather.py
│   ├── soil_water.py
│   ├── crop_establishment.py
│   ├── markov_calibration.py
│   ├── rainfall_amount_calibration.py
│   ├── rainfall_preprocessing.py
│   ├── rainfall_states.py
│   ├── build_calibration_artifact.py
│   ├── calibration_artifact.py
│   ├── calibration_registry.py
│   ├── earth_engine.py
│   ├── gemini_explainer.py
│   └── imd_data.py
├── croplogic_saathi/
│   ├── __init__.py
│   ├── client.py
│   └── models.py
└── tests/
```

Module responsibilities should remain separated.

Do not rebuild modules unnecessarily. Extend the existing architecture
after validating requirements.

------------------------------------------------------------------------

## 18. Current Production-Readiness Principles

The project should prioritize:

1.  Correctness
2.  Reproducibility
3.  Leakage-safe validation
4.  Explicit assumptions
5.  Model/data provenance
6.  Geographic extensibility
7.  Persistent calibration reuse
8.  API/SDK stability
9.  Production error handling
10. Test coverage
11. Clear failure behaviour
12. Honest uncertainty

Avoid adding technology merely for branding.

Google technologies, Earth Engine, Gemini or other external systems
should only be introduced when technically justified and actually
useful.

------------------------------------------------------------------------

## 19. Development Order

Preferred sequence:

``` text
Problem validity
    ↓
Existing-solution/evidence analysis
    ↓
Data strategy
    ↓
Agricultural assumptions
    ↓
Location → climate-data architecture
    ↓
Calibration
    ↓
Weather simulation
    ↓
Soil-water model
    ↓
Crop establishment
    ↓
Monte Carlo
    ↓
Economic engine
    ↓
Decision engine
    ↓
Leakage-safe backtesting
    ↓
Geographic calibration/cache
    ↓
API/SDK
    ↓
Deployment
    ↓
Monitoring/recalibration workflow
    ↓
Field validation when possible
```

Do not prioritize UI or language-model features before validating the
core engine.

------------------------------------------------------------------------

## 20. Current Engineering Philosophy

When modifying the project:

-   Inspect the existing implementation first.
-   Preserve separation of concerns.
-   Make the smallest justified change.
-   Avoid unexplained hard-coded values.
-   Keep production calibration separate from test/dev overrides.
-   Add targeted tests for every behavioural change.
-   Run the relevant targeted tests.
-   Run the full regression suite.
-   Run `git diff --check`.
-   Review the final diff before committing.
-   Never silently change established architectural decisions.
-   Record meaningful architecture/data/model/API changes in project
    documentation.

When uncertain, prefer:

> **simple + explainable + reproducible + testable**

over unnecessary complexity.

------------------------------------------------------------------------

## 21. Current Strategic Goal

The goal is **not** to make CropLogic-Saathi solve every agricultural
uncertainty.

The goal is to make one clearly defined capability technically
defensible:

> **Convert uncertain climate behaviour and current field conditions
> into a measurable, probabilistic and economically-aware pre-sowing
> decision.**

The system should continuously improve through evidence:

``` text
data
 ↓
calibration
 ↓
decision
 ↓
observed outcome
 ↓
backtesting
 ↓
evaluation
 ↓
recalibration
 ↓
validated new version
```

The strongest evidence of value will be demonstrated through
reproducible comparisons against simpler baselines, followed eventually
by broader geographic and field validation.

------------------------------------------------------------------------

## 22. Current Status / Direction

The project has already established:

-   a modular decision engine,
-   rainfall/climate simulation,
-   soil-water modelling,
-   crop establishment modelling,
-   economic comparison,
-   leakage-safe backtesting,
-   calibration artifacts,
-   calibration provenance/content hashing,
-   reproducible SDK decisions,
-   decision trace/provenance,
-   typed SDK response models,
-   structured API validation errors,
-   installable Python SDK,
-   packaged calibration data,
-   API/SDK contract tests,
-   production-style dependency/build validation.

The next major architectural focus is:

> **Location → Climate Data → Calibration Cache → Decision**

with Yavatmal used as the initial reference implementation while keeping
the engine location-general.

Do not redesign the entire engine to accomplish this. Extend the
existing calibration/data architecture.

------------------------------------------------------------------------

## 23. One-Sentence Project Definition

> **CropLogic-Saathi is a location-general probabilistic decision engine
> that combines historical climate behaviour, current local field
> observations, forecast information, soil/crop dynamics and economic
> risk to provide reproducible, explainable SOW/WAIT/SWITCH pre-sowing
> decision support through an SDK/API.**
