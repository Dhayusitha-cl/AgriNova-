"""
AgriNova - CropLogic-Saathi
Economic Risk Engine

Compares the economic consequences of:
1. SOW NOW
2. WAIT
3. SWITCH CROP

Important:
The values in crop_data.py are model inputs/assumptions.
They are not guarantees of actual farmer income.
"""

from crop_data import crops


def calculate_crop_revenue(crop_name):
    """
    Estimate gross revenue per acre.

    Revenue = expected yield × market price

    Yield is treated as quintals per acre based on crop_data.py.
    """
    crop = crops[crop_name]

    return (
        crop["average_yield_per_acre"]
        * crop["market_price_per_quintal"]
    )


def calculate_net_return(crop_name):
    """
    Estimate simple net return before additional risk adjustments.

    Net return = gross revenue - seed cost

    This is intentionally simplified for the prototype.
    """
    revenue = calculate_crop_revenue(crop_name)
    seed_cost = crops[crop_name]["seed_cost_per_acre"]

    return revenue - seed_cost


def calculate_risk_adjusted_return(
    crop_name,
    establishment_probability
):
    """
    Estimate expected return after accounting for establishment risk.

    Expected return =
        probability of successful establishment × net return

    If establishment fails, the model currently treats the crop
    return as zero while retaining the seed cost implicitly
    through the net-return formulation.

    This is a simplified prototype metric and should be validated
    with better production-cost and yield-risk data later.
    """

    net_return = calculate_net_return(crop_name)

    return establishment_probability * net_return


def compare_crops(
    primary_crop,
    primary_probability,
    alternative_probabilities=None
):
    """
    Compare the primary crop with alternative crops.

    Parameters
    ----------
    primary_crop : str
        Crop currently selected by the farmer.

    primary_probability : float
        Estimated establishment probability for the primary crop.

    alternative_probabilities : dict, optional
        Example:
        {
            "soybean": 0.53,
            "sorghum": 0.77
        }

    Returns
    -------
    dict
        Economic comparison for each crop.
    """

    if alternative_probabilities is None:
        alternative_probabilities = {}

    probabilities = {
        primary_crop: primary_probability,
        **alternative_probabilities
    }

    comparison = {}

    for crop_name, probability in probabilities.items():

        if crop_name not in crops:
            continue

        gross_revenue = calculate_crop_revenue(crop_name)
        net_return = calculate_net_return(crop_name)

        risk_adjusted_return = calculate_risk_adjusted_return(
            crop_name,
            probability
        )

        comparison[crop_name] = {
            "establishment_probability": probability,
            "gross_revenue": gross_revenue,
            "seed_cost": crops[crop_name]["seed_cost_per_acre"],
            "simple_net_return": net_return,
            "risk_adjusted_return": risk_adjusted_return
        }

    return comparison


def select_best_economic_option(comparison):
    """
    Select the crop with the highest risk-adjusted return.
    """

    if not comparison:
        return None

    return max(
        comparison,
        key=lambda crop_name:
        comparison[crop_name]["risk_adjusted_return"]
    )
def calculate_expected_return(crop_name, establishment_probability):
    """
    Calculate expected net return after accounting for
    probability of successful crop establishment.

    Expected return = establishment probability × net return

    This is a simplified prototype economic-risk calculation.
    """

    net_return = calculate_net_return(crop_name)

    expected_return = establishment_probability * net_return

    return expected_return
def compare_decision_options(
    primary_crop,
    sow_probability,
    wait_probability,
    switch_crop,
    switch_probability
):
    """
    Compare the three decision options:

    1. SOW NOW
    2. WAIT
    3. SWITCH CROP

    This is a simplified prototype economic comparison.

    WAIT uses the same primary-crop net return but has a
    probability reflecting the expected establishment outcome
    after waiting.

    SWITCH uses the alternative crop's own net return and
    establishment probability.

    Returns a dictionary containing the economic value of
    each decision option.
    """

    sow_return = calculate_risk_adjusted_return(
        primary_crop,
        sow_probability
    )

    wait_return = calculate_risk_adjusted_return(
        primary_crop,
        wait_probability
    )

    switch_return = calculate_risk_adjusted_return(
        switch_crop,
        switch_probability
    )

    options = {
        "SOW": {
            "crop": primary_crop,
            "probability": sow_probability,
            "expected_return": sow_return
        },
        "WAIT": {
            "crop": primary_crop,
            "probability": wait_probability,
            "expected_return": wait_return
        },
        "SWITCH": {
            "crop": switch_crop,
            "probability": switch_probability,
            "expected_return": switch_return
        }
    }

    best_option = max(
        options,
        key=lambda option:
        options[option]["expected_return"]
    )

    return {
        "options": options,
        "best_option": best_option
    }
def calculate_wait_return(
    crop_name,
    establishment_probability,
    wait_days=5
):
    """
    Estimate expected return when sowing is delayed.

    Waiting can reduce expected yield because the crop is
    sown later in the available sowing window.

    This is a simplified prototype assumption.
    """

    crop = crops[crop_name]

    base_return = calculate_net_return(crop_name)

    daily_loss = crop["yield_loss_per_day_pct"] / 100

    yield_retention = max(
        0,
        1 - (daily_loss * wait_days)
    )

    adjusted_return = base_return * yield_retention

    expected_return = (
        establishment_probability * adjusted_return
    )

    return expected_return