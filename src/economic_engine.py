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
- Simulation probabilities are not guarantees of crop success.
- WAIT uses the establishment probability supplied by the physical model.
- Crop-specific yield-loss parameters are used for the five-day delay.
"""

from src.crop_data import crops
from src.soil_data import soils


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


WAIT_DAYS = 5



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

    yield_per_acre = float(yield_per_acre)
    price_per_quintal = float(price_per_quintal)
    seed_cost = float(seed_cost)
    success_probability = float(success_probability)

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

    The establishment probability is supplied by the physical model.

    Expected profit:
        P(success) * profit_if_success
        +
        P(failure) * profit_if_failure

    Failure currently represents loss of the primary crop seed cost.
    Recovery crops are not included here because their establishment
    probability is not produced by the physical simulation.
    """

    crop = crops[crop_name]

    success_profit = (
        crop["average_yield_per_acre"]
        * crop["market_price_per_quintal"]
        - crop["seed_cost_per_acre"]
    )

    failure_profit = -crop["seed_cost_per_acre"]

    failure_probability = 1.0 - germination_prob

    total_expected_profit = (
        germination_prob * success_profit
        + failure_probability * failure_profit
    )

    if germination_prob > 0.70:
        risk_level = "Low"
    elif germination_prob > 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "decision": "Sow Today",
        "expected_profit": float(total_expected_profit),
        "success_probability": float(germination_prob),
        "best_case_profit": float(success_profit),
        "worst_case_profit": float(failure_profit),
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

    Important design rule:
    ----------------------
    `germination_prob` is the establishment probability produced
    by the physical Monte Carlo / soil-water model for the WAIT
    scenario.

    The economic engine does NOT independently assume a rainfall
    probability or artificially increase the supplied probability.

    Waiting causes a crop-specific yield penalty based on the
    crop's `yield_loss_per_day_pct` parameter.
    """

    crop = crops[crop_name]

    # -------------------------------------------------------------
    # Crop-specific yield penalty for waiting.
    #
    # Example:
    # cotton = 1.2% loss/day
    # 5 days = 6% loss
    # remaining yield = 94%
    #
    # The value is capped at zero to avoid negative yield.
    # -------------------------------------------------------------

    daily_yield_loss_pct = float(
        crop["yield_loss_per_day_pct"]
    )

    total_delay_loss_pct = (
        daily_yield_loss_pct * WAIT_DAYS
    )

    delay_yield_factor = max(
        0.0,
        1.0 - (total_delay_loss_pct / 100.0),
    )

    delayed_yield = (
        crop["average_yield_per_acre"]
        * delay_yield_factor
    )

    # -------------------------------------------------------------
    # Primary crop expected profit after waiting.
    #
    # The supplied germination probability comes directly from
    # the physical WAIT simulation.
    # -------------------------------------------------------------

    primary_crop_profit = calculate_profit(
        crop_name=crop_name,
        yield_per_acre=delayed_yield,
        price_per_quintal=crop[
            "market_price_per_quintal"
        ],
        seed_cost=crop[
            "seed_cost_per_acre"
        ],
        success_probability=germination_prob,
    )


    # -------------------------------------------------------------
    # Expected WAIT outcome.
    #
    # Primary crop success/failure is represented by the physical
    # establishment probability.
    #
    # We do not introduce a separate rainfall_probability here.
    # -------------------------------------------------------------

    total_expected_profit = (
        primary_crop_profit
    )

    if germination_prob > 0.70:
        risk_level = "Low"
    elif germination_prob > 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Worst case remains a simple late-soybean recovery scenario.
    worst_case_profit = -crop["seed_cost_per_acre"]

    # Best case is successful establishment of the delayed crop.
    best_case_profit = (
        delayed_yield
        * crop["market_price_per_quintal"]
        - crop["seed_cost_per_acre"]
    )

    return {
        "decision": "Wait 5 Days",
        "expected_profit": float(
            total_expected_profit
        ),
        "success_probability": float(
            germination_prob
        ),
        "best_case_profit": float(
            best_case_profit
        ),
        "worst_case_profit": float(
            worst_case_profit
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

    The economic engine uses the soybean establishment probability
    from the physical simulation to calculate expected profit.

    Economic assumptions are model assumptions, not guarantees.
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

    soybean_worst_case_profit = (
        -soybean["seed_cost_per_acre"]
    )

    if germination_prob > 0.70:
        risk_level = "Low"
    elif germination_prob > 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

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
            soybean_worst_case_profit
        ),
        "risk_level": risk_level,
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

    The economic engine does not independently simulate soil,
    rainfall, or establishment. Those physical conditions should
    first be converted into establishment probabilities by the
    physical model.
    """

    # Retained for API compatibility.
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

    The three establishment probabilities must come from the
    physical simulation:

        germ_prob_today
        germ_prob_wait
        germ_prob_soybean

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

    # Select the option with the highest expected economic value.
    best_decision = max(
        decisions,
        key=lambda result: result["expected_profit"],
    )

    # Calculate advantage relative to the strongest alternative.
    for decision_result in decisions:

        other_profits = [
            other["expected_profit"]
            for other in decisions
            if other is not decision_result
        ]

        decision_result["advantage_over_others"] = float(
            decision_result["expected_profit"]
            - max(other_profits)
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
    """

    return f"₹{amount:,.0f}"
