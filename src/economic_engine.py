"""
Economic Engine for AgriNova / CropLogic-Saathi.

Purpose
-------
Compare the economic outcomes of:

    1. SOW TODAY
    2. WAIT 5 DAYS
    3. SWITCH TO SOYBEAN

The physical simulation and the economic model are kept separate.

Important
---------
- Establishment probabilities come from the physical simulation.
- This module converts those probabilities into expected monetary outcomes.
- Economic assumptions are model assumptions, not field-validated guarantees.
- Current risk labels are simple categorical indicators.
"""


from src.crop_data import crops
from src.soil_data import soils


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_inputs(
    crop_name,
    soil_type,
    germination_prob,
):
    """Validate common economic-engine inputs."""

    if crop_name not in crops:
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

    if soil_type not in soils:
        raise ValueError(
            f"Unknown soil type: {soil_type}"
        )

    if not 0 <= germination_prob <= 1:
        raise ValueError(
            "germination_prob must be between 0 and 1."
        )


# ---------------------------------------------------------------------
# Generic profit calculation
# ---------------------------------------------------------------------

def calculate_profit(
    crop_name,
    yield_per_acre,
    price_per_quintal,
    seed_cost,
    success_probability,
):
    """
    Calculate expected profit for a crop.

    Success case
    ------------
    yield * price - seed cost

    Failure case
    ------------
    -seed cost

    Expected profit
    ---------------
    P(success) * success_profit
    +
    P(failure) * failure_profit

    The probability is applied exactly once.
    """

    if crop_name not in crops:
        raise ValueError(
            f"Unknown crop: {crop_name}"
        )

    if not 0 <= success_probability <= 1:
        raise ValueError(
            "success_probability must be between 0 and 1."
        )

    profit_if_success = (
        yield_per_acre
        * price_per_quintal
        - seed_cost
    )

    profit_if_failure = -seed_cost

    expected_profit = (
        success_probability * profit_if_success
        + (1 - success_probability) * profit_if_failure
    )

    return float(expected_profit)


# ---------------------------------------------------------------------
# SOW TODAY
# ---------------------------------------------------------------------

def _calculate_sow_today(
    crop_name,
    germination_prob,
):
    """
    Calculate expected outcome when sowing the primary crop today.

    Current model assumption:
    If the primary crop fails, a late soybean recovery attempt is
    possible. The recovery economics are deliberately kept simple.
    """

    crop = crops[crop_name]
    soybean = crops["soybean"]

    # Primary crop success outcome.
    success_profit = (
        crop["average_yield_per_acre"]
        * crop["market_price_per_quintal"]
        - crop["seed_cost_per_acre"]
    )

    # Primary crop failure means seed cost is lost.
    failure_profit = -crop["seed_cost_per_acre"]

    # Late soybean recovery assumption.
    late_soybean_yield = (
        soybean["average_yield_per_acre"]
        * 0.70
    )

    late_soybean_profit = calculate_profit(
        crop_name="soybean",
        yield_per_acre=late_soybean_yield,
        price_per_quintal=soybean[
            "market_price_per_quintal"
        ],
        seed_cost=soybean[
            "seed_cost_per_acre"
        ],
        success_probability=0.80,
    )

    failure_probability = (
        1.0 - germination_prob
    )

    total_expected_profit = (
        germination_prob * success_profit
        + failure_probability
        * (
            failure_profit
            + late_soybean_profit
        )
    )

    if germination_prob > 0.70:
        risk_level = "Low"
    elif germination_prob > 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "decision": "Sow Today",
        "expected_profit": float(
            total_expected_profit
        ),
        "success_probability": float(
            germination_prob
        ),
        "best_case_profit": float(
            success_profit
        ),
        "worst_case_profit": float(
            failure_profit + late_soybean_profit
        ),
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------
# WAIT
# ---------------------------------------------------------------------

def _calculate_wait(
    crop_name,
    germination_prob,
):
    """
    Calculate expected outcome when waiting five days.

    Current model assumptions:
    - Probability of useful rain during the wait = 0.65
    - Improved establishment probability is capped at 0.85
    - Waiting causes a 6% yield reduction
    - If useful rain does not occur, late soybean is considered
      as an alternative.
    """

    crop = crops[crop_name]
    soybean = crops["soybean"]

    # Assumed probability of useful rainfall during waiting.
    rain_probability = 0.65

    # Assumed improvement in establishment probability after waiting.
    improved_germination = min(
        0.85,
        germination_prob + 0.30,
    )

    # Assumed yield penalty caused by delaying sowing.
    delay_yield_loss = 0.94

    # Primary crop outcome if useful rain occurs.
    with_rain_profit = calculate_profit(
        crop_name=crop_name,
        yield_per_acre=(
            crop["average_yield_per_acre"]
            * delay_yield_loss
        ),
        price_per_quintal=crop[
            "market_price_per_quintal"
        ],
        seed_cost=crop[
            "seed_cost_per_acre"
        ],
        success_probability=improved_germination,
    )

    # Alternative if useful rain does not occur.
    late_soybean_yield = (
        soybean["average_yield_per_acre"]
        * 0.65
    )

    no_rain_profit = calculate_profit(
        crop_name="soybean",
        yield_per_acre=late_soybean_yield,
        price_per_quintal=soybean[
            "market_price_per_quintal"
        ],
        seed_cost=soybean[
            "seed_cost_per_acre"
        ],
        success_probability=0.70,
    )

    total_expected_profit = (
        rain_probability * with_rain_profit
        + (1 - rain_probability) * no_rain_profit
    )

    # Current WAIT risk policy.
    if rain_probability > 0.70:
        risk_level = "Low"
    else:
        risk_level = "Medium"

    return {
        "decision": "Wait 5 Days",
        "expected_profit": float(
            total_expected_profit
        ),
        "success_probability": float(
            improved_germination
        ),
        "best_case_profit": float(
            with_rain_profit
        ),
        "worst_case_profit": float(
            no_rain_profit
        ),
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------
# SWITCH TO SOYBEAN
# ---------------------------------------------------------------------

def _calculate_switch(
    germination_prob,
):
    """
    Calculate expected outcome when switching to soybean.

    The supplied germination probability is interpreted as the
    estimated establishment probability for soybean.

    Current model policy:
    SWITCH is labelled as a low operational-risk alternative because
    the action itself avoids continuing with the original crop.

    The probability still directly affects expected monetary outcome.
    """

    soybean = crops["soybean"]

    soybean_expected_profit = calculate_profit(
        crop_name="soybean",
        yield_per_acre=soybean[
            "average_yield_per_acre"
        ],
        price_per_quintal=soybean[
            "market_price_per_quintal"
        ],
        seed_cost=soybean[
            "seed_cost_per_acre"
        ],
        success_probability=germination_prob,
    )

    soybean_best_case_profit = (
        soybean["average_yield_per_acre"]
        * soybean["market_price_per_quintal"]
        - soybean["seed_cost_per_acre"]
    )

    return {
        "decision": "Switch to Soybean",
        "expected_profit": float(
            soybean_expected_profit
        ),
        "success_probability": float(
            germination_prob
        ),
        "best_case_profit": float(
            soybean_best_case_profit
        ),
        "worst_case_profit": float(
            soybean_expected_profit
        ),
        "risk_level": "Low",
    }


# ---------------------------------------------------------------------
# Single-decision economic outcome
# ---------------------------------------------------------------------

def calculate_economic_outcome(
    crop_name,
    soil_type,
    decision,
    germination_prob,
    rainfall_yesterday_mm,
    current_moisture_mm,
):
    """
    Calculate the economic outcome for one decision.

    Parameters
    ----------
    crop_name : str
        Primary crop.

    soil_type : str
        Soil category.

    decision : str
        One of:
            "sow_today"
            "wait"
            "switch"

    germination_prob : float
        Establishment probability produced by the physical model.

    rainfall_yesterday_mm : float
        Recent rainfall observation.

    current_moisture_mm : float
        Current soil-water state.

    Notes
    -----
    rainfall_yesterday_mm and current_moisture_mm are retained
    for compatibility with the decision engine.

    The economic engine does not independently simulate soil or
    rainfall. Those physical conditions should first be converted
    into establishment probabilities by the physical model.
    """

    # These values are intentionally retained in the public API.
    # They are not directly used by the current economic model.
    _ = rainfall_yesterday_mm
    _ = current_moisture_mm

    _validate_inputs(
        crop_name=crop_name,
        soil_type=soil_type,
        germination_prob=germination_prob,
    )

    if decision == "sow_today":
        return _calculate_sow_today(
            crop_name=crop_name,
            germination_prob=germination_prob,
        )

    if decision == "wait":
        return _calculate_wait(
            crop_name=crop_name,
            germination_prob=germination_prob,
        )

    if decision == "switch":
        return _calculate_switch(
            germination_prob=germination_prob,
        )

    # Unknown decisions are handled explicitly.
    return {
        "decision": "Unknown",
        "expected_profit": 0.0,
        "success_probability": 0.0,
        "best_case_profit": 0.0,
        "worst_case_profit": 0.0,
        "risk_level": "Unknown",
    }


# ---------------------------------------------------------------------
# Compare all decisions
# ---------------------------------------------------------------------

def compare_all_decisions(
    crop_name,
    soil_type,
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
    rainfall_yesterday_mm,
    current_moisture_mm,
):
    """
    Compare SOW TODAY, WAIT and SWITCH economically.

    Returns
    -------
    dict
        Contains:

        - sow_today
        - wait
        - switch
        - best_decision
        - best_profit
        - all_decisions
    """

    sow_today = calculate_economic_outcome(
        crop_name=crop_name,
        soil_type=soil_type,
        decision="sow_today",
        germination_prob=germ_prob_today,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        current_moisture_mm=current_moisture_mm,
    )

    wait = calculate_economic_outcome(
        crop_name=crop_name,
        soil_type=soil_type,
        decision="wait",
        germination_prob=germ_prob_wait,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        current_moisture_mm=current_moisture_mm,
    )

    switch = calculate_economic_outcome(
        crop_name=crop_name,
        soil_type=soil_type,
        decision="switch",
        germination_prob=germ_prob_soybean,
        rainfall_yesterday_mm=rainfall_yesterday_mm,
        current_moisture_mm=current_moisture_mm,
    )

    decisions = [
        sow_today,
        wait,
        switch,
    ]

    # Select the option with the highest expected profit.
    best_decision = max(
        decisions,
        key=lambda result: result["expected_profit"],
    )

    # Calculate advantage relative to the weakest alternative.
    for decision_result in decisions:

        other_profits = [
            other["expected_profit"]
            for other in decisions
            if other is not decision_result
        ]

        decision_result["advantage_over_others"] = float(
            decision_result["expected_profit"]
            - min(other_profits)
        )

    return {
        "sow_today": sow_today,
        "wait": wait,
        "switch": switch,
        "best_decision": best_decision["decision"],
        "best_profit": float(
            best_decision["expected_profit"]
        ),
        "all_decisions": decisions,
    }


# ---------------------------------------------------------------------
# Currency formatting
# ---------------------------------------------------------------------

def format_currency(amount):
    """
    Format a number as Indian Rupees.

    Example
    -------
    45000 -> ₹45,000

    The Unicode rupee symbol is used directly.
    """

    return f"₹{amount:,.0f}"