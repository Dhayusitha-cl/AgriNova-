\# AgriNova â€” CropLogic-Saathi



\*\*Climate-resilient pre-sowing decision support for rainfed smallholder farming.\*\*



\## Problem



Farmers may have weather forecasts and agricultural advisories, but the difficult decision is translating uncertain rainfall and current field conditions into an actionable \*\*SOW, WAIT, or SWITCH CROP\*\* decision.



\## Solution



\*\*CropLogic-Saathi\*\* is a probabilistic pre-sowing decision engine that combines:



\* Recent rainfall and current soil-water state

\* Historical rainfall behaviour

\* Plausible future weather scenarios

\* Soil-water balance

\* Crop establishment requirements

\* Monte Carlo simulation

\* Economic comparison



It estimates establishment risk and compares:



\*\*SOW TODAY â†’ WAIT 5 DAYS â†’ SWITCH TO SOYBEAN\*\*



The system provides an explainable recommendation together with probabilities, economic outcomes, assumptions, and uncertainty.



\## Technical Pipeline



```text

Field + Weather Inputs

&#x20;       â†“

Input Validation

&#x20;       â†“

Current Soil-Water State

&#x20;       â†“

Historical Climate Behaviour

&#x20;       +

Current / Recent Weather

&#x20;       â†“

Weather Scenarios

&#x20;       â†“

Soil-Water Simulation

&#x20;       â†“

Crop Establishment Evaluation

&#x20;       â†“

Monte Carlo Risk Estimation

&#x20;       â†“

Economic Comparison

&#x20;       â†“

SOW / WAIT / SWITCH

```



\## Validation



The project includes historical backtesting with leakage protection.



For each decision date:



\* Only information available \*\*before the decision date\*\* is used for the decision.

\* The following \*\*14 days are held out\*\*.

\* Held-out rainfall is used only for evaluation.

\* CropLogic-Saathi is compared with weather-only and simple rule-based baselines.



Current validation uses \*\*10 historical decision dates from 2020â€“2024\*\* and \*\*1,000 simulations per assessment\*\*.



The current simplified rainfall-based outcome proxy gives:



| Approach         | Success proxy |

| ---------------- | ------------: |

| Weather-only     |         0.800 |

| Rule-based       |         0.800 |

| CropLogic-Saathi |         0.700 |



This result does \*\*not\*\* show that CropLogic-Saathi currently outperforms the baselines. The evaluation is intended to test the decision framework and identify areas requiring further calibration and validation.



This is \*\*not field validation\*\* and does not demonstrate causal impact or agronomic superiority.



\## Limitations



\* No field trials yet

\* Soil/field observations can be subjective

\* Simplified crop and soil parameters

\* Economic assumptions may vary by farmer and location

\* Historical rainfall behaviour may not represent future climate conditions

\* Extreme events can still cause crop failure

\* The current evaluation uses a simplified outcome proxy



CropLogic-Saathi is \*\*decision support, not a guarantee of crop success\*\*.



\## Project Structure



```text

AgriNova/

â”œâ”€â”€ app.py

â”œâ”€â”€ src/

â”‚   â”œâ”€â”€ crop\_data.py

â”‚   â”œâ”€â”€ soil\_data.py

â”‚   â”œâ”€â”€ weather\_simulator.py

â”‚   â”œâ”€â”€ climate\_simulator.py

â”‚   â”œâ”€â”€ decision\_engine.py

â”‚   â””â”€â”€ economic\_engine.py

â”œâ”€â”€ validation/

â”‚   â””â”€â”€ baseline\_comparison.py

â”œâ”€â”€ tests/

â””â”€â”€ data/

```



\## Run



```powershell

pip install -r requirements.txt

pytest -q

python -m validation.baseline\_comparison

```



\## Status



Core decision engine, economic engine, baseline comparison, backtesting, and automated tests are implemented.



CropLogic-Saathi remains a research and decision-support prototype requiring further calibration and field validation.