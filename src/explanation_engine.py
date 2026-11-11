
# src/explanation_engine.py

"""
Deterministic explanation layer for CropLogic-Saathi.

This module explains the decision engine's existing output.
It must never select, modify, or override the recommendation.
"""


def _normalise(value):
    return " ".join(str(value or "").strip().lower().split())


def _format_probability(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Not available"

    if not 0 <= value <= 1:
        return "Not available"

    return f"{value * 100:.1f}%"


def _format_money(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Not available"

    return f"₹{value:,.2f}"


def _find_recommended_outcome(decision, outcomes):
    """
    Match the final decision to its existing economic outcome.
    Never independently choose the highest-profit outcome here.
    """
    target = _normalise(decision)

    for outcome in outcomes:
        if _normalise(outcome.get("decision")) == target:
            return outcome

    return None


def generate_decision_explanation(result, primary_crop=None):
    """
    Generate an explanation from a completed decision-engine result.

    Parameters
    ----------
    result : dict
        Result returned by the CropLogic-Saathi decision engine.

    primary_crop : str, optional
        The originally selected crop. Pass this from the caller
        when available; it is not currently included in every
        decision-engine result.

    Returns
    -------
    dict
        Structured explanation for presentation by the UI/API.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary.")

    decision = result.get("decision")

    if not isinstance(decision, str) or not decision.strip():
        raise ValueError("result must contain a valid decision.")

    economic = result.get("economic_comparison") or {}
    outcomes = economic.get("all_decisions") or []

    if not isinstance(outcomes, list):
        outcomes = []

    recommended_outcome = _find_recommended_outcome(
        decision,
        outcomes,
    )

    why_recommended = []
    alternative_comparison = []

    # ---------------------------------------------------------
    # Recommendation explanation
    # ---------------------------------------------------------

    if recommended_outcome:
        probability = recommended_outcome.get("success_probability")
        profit = recommended_outcome.get("expected_profit")
        risk = recommended_outcome.get("risk_level")

        why_recommended.append(
            "The decision engine selected this option through "
            "its economic comparison of eligible actions."
        )

        why_recommended.append(
            "Estimated crop-establishment probability: "
            f"{_format_probability(probability)}."
        )

        why_recommended.append(
            "Modelled expected profit: "
            f"{_format_money(profit)} per acre."
        )

        if risk:
            why_recommended.append(
                f"The model assigns this option a {risk} "
                "establishment-risk category."
            )
    else:
        why_recommended.append(
            "The final recommendation is available, but its matching "
            "economic outcome was not found in all_decisions. "
            "The explanation will not infer missing figures."
        )

    # ---------------------------------------------------------
    # Economic comparison
    # ---------------------------------------------------------

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue

        alternative_comparison.append({
            "action": outcome.get("decision", "Unknown"),
            "establishment_probability": _format_probability(
                outcome.get("success_probability")
            ),
            "expected_profit": _format_money(
                outcome.get("expected_profit")
            ),
            "risk_level": outcome.get("risk_level", "Unknown"),
            "advantage_over_others": _format_money(
                outcome.get("advantage_over_others")
            ),
        })

    # ---------------------------------------------------------
    # Field conditions
    # ---------------------------------------------------------

    field_observations = []

    current_moisture = result.get("current_moisture")
    minimum_moisture = result.get("min_moisture_required")

    if isinstance(current_moisture, (int, float)):
        field_observations.append(
            f"Current modelled soil moisture: "
            f"{current_moisture:.1f} mm."
        )

    if isinstance(minimum_moisture, (int, float)):
        field_observations.append(
            f"Configured minimum moisture requirement: "
            f"{minimum_moisture:.1f} mm."
        )

    if (
        isinstance(current_moisture, (int, float))
        and isinstance(minimum_moisture, (int, float))
    ):
        difference = current_moisture - minimum_moisture

        if difference >= 0:
            field_observations.append(
                f"Current modelled moisture is {difference:.1f} mm "
                "at or above the configured minimum."
            )
        else:
            field_observations.append(
                f"Current modelled moisture is {abs(difference):.1f} mm "
                "below the configured minimum."
            )

    rainfall_state = result.get("initial_rainfall_state")

    if rainfall_state:
        field_observations.append(
            f"Initial rainfall state used by the simulation: "
            f"{rainfall_state}."
        )


    # ---------------------------------------------------------
    # Crop eligibility
    # ---------------------------------------------------------

    alternative_crop_eligibility = []

    crop_eligibility = result.get("crop_eligibility") or {}

    if isinstance(crop_eligibility, dict):
        for crop, details in crop_eligibility.items():
            if not isinstance(details, dict):
                continue

            status = details.get("status", "unknown")

            # The primary crop is not an alternative switch option.
            if status == "primary_crop":
                continue

            alternative_crop_eligibility.append({
                "crop": crop,
                "eligible": details.get("eligible"),
                "status": status,
                "reason": details.get("reason", ""),
            })

    # ---------------------------------------------------------
    # Sowing action eligibility
    # ---------------------------------------------------------

    sowing_action_eligibility = []

    sowing_eligibility = result.get("sowing_eligibility") or {}

    if isinstance(sowing_eligibility, dict):
        for action, details in sowing_eligibility.items():
            if not isinstance(details, dict):
                continue

            sowing_action_eligibility.append({
                "crop": primary_crop or "Primary crop",
                "action": action,
                "eligible": details.get("eligible"),
                "status": details.get("status", "unknown"),
                "reason": details.get("reason", ""),
                "proposed_sowing_date": details.get(
                    "proposed_sowing_date"
                ),
            })


    # ---------------------------------------------------------
    # Uncertainty and limitations
    # ---------------------------------------------------------

    uncertainties = [
        "Establishment probabilities are simulation estimates, "
        "not guarantees of successful crop establishment or harvest.",
        "The economic comparison depends on configured yield, "
        "market-price, seed-cost, and waiting-cost assumptions.",
        "The model's Low, Medium, and High risk labels are based "
        "on fixed establishment-probability thresholds; they are "
        "not independently validated financial-risk categories.",
    ]

    assumptions = result.get("assumptions") or {}

    wait_days = assumptions.get("wait_days")

    if wait_days is not None:
        uncertainties.append(
            f"The WAIT calculation uses a configured delay of "
            f"{wait_days} days and applies the crop-specific "
            "yield-loss parameter."
        )

    uncertainties.append(
        "A crop passing its configured sowing-window timing filter "
        "does not establish regional, soil, variety, or market suitability."
    )

    uncertainties.append(
        "The simplified worst-case outcome is not a complete estimate "
        "of all possible farm losses."
    )

    # ---------------------------------------------------------
    # Final structured explanation
    # ---------------------------------------------------------

    switched_crop = None

    if _normalise(decision).startswith("switch to "):
        switched_crop = decision.strip()[len("SWITCH TO "):].strip()

    return {
        "recommended_action": decision,
        "recommended_crop": switched_crop or primary_crop,
        "why_recommended": why_recommended,
        "alternative_comparison": alternative_comparison,
        "field_observations": field_observations,
        "crop_eligibility": alternative_crop_eligibility,
        "sowing_action_eligibility": sowing_action_eligibility,
        "uncertainties": uncertainties,
    }
