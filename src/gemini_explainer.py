import os
import google.generativeai as genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


# ============================================================
# LOCAL FALLBACK EXPLANATION
# ============================================================

def generate_local_explanation(
    decision,
    crop_name,
    soil_type,
    current_moisture,
    min_moisture_required,
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
    confidence
):
    """
    Generates a farmer-friendly explanation using the actual
    CropLogic-Saathi decision-engine results.

    This is used when Gemini is unavailable.
    """

    today = germ_prob_today * 100
    wait = germ_prob_wait * 100
    soybean = germ_prob_soybean * 100
    confidence_pct = confidence * 100

    moisture_difference = (
        current_moisture - min_moisture_required
    )

    # --------------------------------------------------------
    # SOW TODAY
    # --------------------------------------------------------

    if decision == "SOW TODAY":

        if moisture_difference >= 0:

            explanation = (
                f"Based on the current field conditions, "
                f"sowing {crop_name} today is recommended. "
                f"The estimated soil moisture is "
                f"{current_moisture:.1f} mm, which is above "
                f"the minimum requirement of "
                f"{min_moisture_required:.1f} mm. "
                f"The model estimates a "
                f"{today:.1f}% probability of successful "
                f"germination if the crop is sown today, "
                f"compared with {wait:.1f}% if sowing is "
                f"delayed by 5 days. "
                f"The recommendation confidence is "
                f"{confidence_pct:.1f}%."
            )

        else:

            explanation = (
                f"The model currently recommends sowing "
                f"{crop_name} today. However, the estimated "
                f"soil moisture of {current_moisture:.1f} mm "
                f"is below the minimum requirement of "
                f"{min_moisture_required:.1f} mm. "
                f"The estimated probability of successful "
                f"germination today is {today:.1f}%, while "
                f"the probability after waiting 5 days is "
                f"{wait:.1f}%. "
                f"Recommendation confidence is "
                f"{confidence_pct:.1f}%. "
                f"Farmers should consider the local field "
                f"condition before sowing."
            )

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    elif decision == "WAIT 5 DAYS":

        explanation = (
            f"The model recommends waiting 5 days before "
            f"sowing {crop_name}. Under the current "
            f"conditions, the estimated probability of "
            f"successful germination when sowing today is "
            f"{today:.1f}%, compared with {wait:.1f}% "
            f"after waiting 5 days. "
            f"Current estimated soil moisture is "
            f"{current_moisture:.1f} mm, while the minimum "
            f"requirement is {min_moisture_required:.1f} mm. "
            f"The recommendation confidence is "
            f"{confidence_pct:.1f}%. "
            f"Waiting may provide a more favorable "
            f"establishment opportunity under the simulated "
            f"conditions."
        )

    # --------------------------------------------------------
    # SWITCH TO SOYBEAN
    # --------------------------------------------------------

    elif decision == "SWITCH TO SOYBEAN":

        explanation = (
            f"The model recommends considering soybean "
            f"instead of {crop_name} under the current "
            f"conditions. "
            f"The estimated germination probability for "
            f"{crop_name} when sown today is {today:.1f}%, "
            f"while the alternative soybean scenario has an "
            f"estimated probability of {soybean:.1f}%. "
            f"The current soil moisture is "
            f"{current_moisture:.1f} mm and the minimum "
            f"requirement for the selected crop is "
            f"{min_moisture_required:.1f} mm. "
            f"The recommendation confidence is "
            f"{confidence_pct:.1f}%. "
            f"This suggests that soybean may offer a more "
            f"favorable establishment opportunity under the "
            f"simulated conditions."
        )

    # --------------------------------------------------------
    # UNKNOWN DECISION
    # --------------------------------------------------------

    else:

        explanation = (
            f"The CropLogic-Saathi model has evaluated the "
            f"current conditions for {crop_name}. "
            f"The estimated soil moisture is "
            f"{current_moisture:.1f} mm, compared with a "
            f"minimum requirement of "
            f"{min_moisture_required:.1f} mm. "
            f"The estimated probabilities are "
            f"{today:.1f}% for sowing today, "
            f"{wait:.1f}% for waiting 5 days, and "
            f"{soybean:.1f}% for the soybean alternative. "
            f"The recommendation confidence is "
            f"{confidence_pct:.1f}%."
        )

    return explanation


# ============================================================
# GEMINI EXPLANATION
# ============================================================

def generate_farmer_explanation(
    decision,
    crop_name,
    soil_type,
    current_moisture,
    min_moisture_required,
    germ_prob_today,
    germ_prob_wait,
    germ_prob_soybean,
    confidence
):
    """
    Generate a farmer-friendly explanation.

    Gemini is used when available.

    If Gemini is unavailable, quota is exhausted, or another
    API error occurs, a local explanation based on the actual
    model output is returned instead.
    """

    # --------------------------------------------------------
    # ALWAYS PREPARE LOCAL FALLBACK
    # --------------------------------------------------------

    fallback_explanation = generate_local_explanation(
        decision=decision,
        crop_name=crop_name,
        soil_type=soil_type,
        current_moisture=current_moisture,
        min_moisture_required=min_moisture_required,
        germ_prob_today=germ_prob_today,
        germ_prob_wait=germ_prob_wait,
        germ_prob_soybean=germ_prob_soybean,
        confidence=confidence
    )

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not API_KEY:

        return fallback_explanation

    # --------------------------------------------------------
    # CALCULATE DISPLAY VALUES
    # --------------------------------------------------------

    today = germ_prob_today * 100
    wait = germ_prob_wait * 100
    soybean = germ_prob_soybean * 100
    confidence_pct = confidence * 100

    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an agricultural decision-support assistant.

Explain the following CropLogic-Saathi result to a
smallholder farmer in simple and practical language.

IMPORTANT:
- Do not change the recommendation.
- Do not invent numbers.
- Use only the information provided below.
- Do not make medical, financial, or unrelated claims.
- Keep the explanation concise.
- Mention why the recommendation was made.
- Mention the important soil-moisture and probability values.

Crop: {crop_name}

Soil type: {soil_type}

Current soil moisture: {current_moisture:.1f} mm

Minimum moisture required:
{min_moisture_required:.1f} mm

Probability of successful germination if sown today:
{today:.1f}%

Probability if farmer waits 5 days:
{wait:.1f}%

Alternative soybean probability:
{soybean:.1f}%

Recommendation:
{decision}

Recommendation confidence:
{confidence_pct:.1f}%

Write the explanation directly for the farmer.
"""

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:

        model = genai.GenerativeModel(
            "gemini-3.6-flash"
        )

        response = model.generate_content(
            prompt
        )

        if response is None:
            return fallback_explanation

        if not hasattr(response, "text"):
            return fallback_explanation

        explanation = response.text

        if not explanation:
            return fallback_explanation

        return explanation.strip()

    # --------------------------------------------------------
    # ANY GEMINI ERROR → LOCAL FALLBACK
    # --------------------------------------------------------

    except Exception:

        return fallback_explanation