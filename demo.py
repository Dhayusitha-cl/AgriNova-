import streamlit as st
import requests
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/assess-sowing-risk"

st.set_page_config(
    page_title="CropLogic-Saathi",
    page_icon="🌾",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌾 CropLogic-Saathi")
st.subheader("Probabilistic Pre-Sowing Decision Support")

st.info(
    "UNCERTAINTY → SCENARIOS → PROBABILITY → ECONOMIC RISK → DECISION"
)


# =========================================================
# SIDEBAR — FARMER INPUTS
# =========================================================

st.sidebar.header("🌱 Field Information")

crop = st.sidebar.selectbox(
    "Crop",
    ["sorghum", "cotton", "soybean", "paddy"]
)

soil_type = st.sidebar.selectbox(
    "Soil Type",
    ["sandy_loam", "medium_black", "deep_black"]
)

current_moisture_mm = st.sidebar.number_input(
    "Current Soil Moisture (mm)",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

rainfall_yesterday_mm = st.sidebar.number_input(
    "Recent Rainfall (mm)",
    min_value=0.0,
    max_value=200.0,
    value=20.0,
    step=1.0
)

week_number = st.sidebar.number_input(
    "Climate Week",
    min_value=1,
    max_value=52,
    value=1,
    step=1
)


# =========================================================
# CURRENT SCENARIO
# =========================================================

st.markdown("## 📋 Current Scenario")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Crop",
    crop.title()
)

col2.metric(
    "Soil",
    soil_type.replace("_", " ").title()
)

col3.metric(
    "Soil Moisture",
    f"{current_moisture_mm:.1f} mm"
)

col4.metric(
    "Recent Rainfall",
    f"{rainfall_yesterday_mm:.1f} mm"
)


# =========================================================
# ASSESSMENT BUTTON
# =========================================================

st.markdown("---")

if st.button(
    "🔮 Assess Sowing Risk",
    type="primary",
    use_container_width=True
):

    payload = {
        "crop": crop,
        "soil_type": soil_type,
        "current_moisture_mm": current_moisture_mm,
        "rainfall_yesterday_mm": rainfall_yesterday_mm,
        "week_number": week_number
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=60
        )

        # =================================================
        # SUCCESS
        # =================================================

        if response.status_code == 200:

            result = response.json()

            st.success(
                "Assessment completed successfully."
            )

            # =================================================
            # RECOMMENDATION
            # =================================================

            st.markdown("## 🎯 Recommendation")

            decision = result.get(
                "decision",
                "Unknown"
            )

            confidence = result.get(
                "confidence",
                0
            )

            st.header(decision)

            st.metric(
                "Decision Confidence",
                f"{confidence * 100:.1f}%"
            )

            # =================================================
            # PROBABILITIES
            # =================================================

            st.markdown(
                "### 📊 Establishment Probability"
            )

            germ_today = result.get(
                "germ_prob_today",
                0
            )

            germ_wait = result.get(
                "germ_prob_wait",
                0
            )

            germ_soybean = result.get(
                "germ_prob_soybean",
                0
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Sow Today",
                f"{germ_today * 100:.1f}%"
            )

            col2.metric(
                "Wait 5 Days",
                f"{germ_wait * 100:.1f}%"
            )

            col3.metric(
                "Switch to Soybean",
                f"{germ_soybean * 100:.1f}%"
            )

            # =================================================
            # FARMER-FRIENDLY EXPLANATION
            # =================================================

            st.markdown(
                "### 🧑‍🌾 Farmer-Friendly Explanation"
            )

            if decision == "SOW TODAY":

                st.success(
                    "The simulated conditions indicate that "
                    "sowing now has the strongest establishment "
                    "probability among the evaluated scenarios."
                )

            elif decision == "WAIT 5 DAYS":

                st.warning(
                    "The model currently favours waiting because "
                    "the simulated conditions do not provide enough "
                    "advantage for sowing immediately."
                )

            elif "SWITCH" in decision:

                st.info(
                    "The model indicates that the alternative crop "
                    "has a stronger simulated establishment outcome "
                    "under the current conditions."
                )

            else:

                st.info(
                    "The model has evaluated the available scenarios "
                    "and returned the recommendation shown above."
                )

            # =================================================
            # MONTE CARLO GRAPH
            # =================================================

            st.markdown(
                "### 📈 Monte Carlo Soil-Moisture Scenarios"
            )

            trajectories = result.get(
                "trajectories"
            )

            if trajectories:

                trajectories = np.array(
                    trajectories
                )

                number_to_show = min(
                    100,
                    trajectories.shape[0]
                )

                chart_data = pd.DataFrame(
                    trajectories[:number_to_show].T
                )

                chart_data.index.name = "Day"

                st.line_chart(
                    chart_data
                )

                st.caption(
                    "Each line represents one plausible "
                    "simulated soil-moisture trajectory."
                )

            else:

                st.info(
                    "Monte Carlo trajectory data "
                    "is not available."
                )

            # =================================================
            # MODEL EXPLANATION
            # =================================================

            st.markdown(
                "### 💡 Why this recommendation?"
            )

            st.write(
                "CropLogic-Saathi evaluates multiple plausible "
                "weather and soil-moisture trajectories using "
                "Monte Carlo simulation rather than relying "
                "on a single rainfall scenario."
            )

            st.write(
                "The simulated trajectories are evaluated against "
                "crop establishment requirements to estimate the "
                "probability of successful establishment."
            )

            st.warning(
                "Probabilities shown here are simulation estimates, "
                "not guarantees of crop success."
            )

            # =================================================
            # DECISION PIPELINE
            # =================================================

            st.markdown(
                "### ⚙️ Decision Engine Pipeline"
            )

            st.code(
                """
Farmer / Field Observation
          ↓
Current Soil & Moisture State
          ↓
Historical Weather Behaviour
          ↓
Plausible Weather Scenarios
          ↓
Soil-Water Balance
          ↓
Crop Establishment Evaluation
          ↓
Monte Carlo Simulation
          ↓
Probability Distribution
          ↓
Decision Comparison
          ↓
SOW / WAIT / SWITCH
                """,
                language="text"
            )

            # =================================================
            # WHAT MAKES THE SYSTEM DIFFERENT
            # =================================================

            st.markdown(
                "### 🌾 What CropLogic-Saathi Adds"
            )

            st.write(
                "The system converts uncertain weather and current "
                "field conditions into an actionable pre-sowing "
                "decision."
            )

            st.write(
                "Instead of presenting rainfall information alone, "
                "the engine evaluates plausible future conditions "
                "and estimates crop-establishment risk."
            )

            # =================================================
            # LIMITATIONS
            # =================================================

            st.markdown(
                "### ⚠️ Important Limitations"
            )

            st.write(
                """
CropLogic-Saathi provides probabilistic decision support,
not guaranteed crop outcomes.

Results depend on the quality of weather, soil and crop
parameters used by the model.

Current limitations include simplified soil and crop
parameters, uncertainty in rainfall and economic assumptions,
spatial data limitations, and the absence of field-trial
validation.
                """
            )

            # =================================================
            # TECHNOLOGY
            # =================================================

            st.markdown(
                "### 🛠️ Technology Used"
            )

            st.write(
                """
• Python
• FastAPI — backend API
• Streamlit — demonstration interface
• NumPy — numerical simulation
• Monte Carlo simulation
• Stochastic weather simulation
• Soil-water balance
• Crop establishment evaluation
                """
            )

            # =================================================
            # DEMO DISCLAIMER
            # =================================================

            st.markdown("---")

            st.caption(
                "CropLogic-Saathi is a decision-support prototype. "
                "Simulation probabilities should not be interpreted "
                "as guarantees of crop establishment or yield."
            )

        # =================================================
        # API ERROR
        # =================================================

        else:

            st.error(
                f"API returned HTTP {response.status_code}"
            )

            st.code(
                response.text
            )

    # =====================================================
    # CONNECTION ERROR
    # =================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the FastAPI backend."
        )

        st.info(
            "Make sure Uvicorn is running with:"
        )

        st.code(
            "uvicorn app:app --reload"
        )

    # =====================================================
    # TIMEOUT ERROR
    # =====================================================

    except requests.exceptions.Timeout:

        st.error(
            "The API took too long to respond."
        )

    # =====================================================
    # OTHER ERROR
    # =====================================================

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )