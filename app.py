import streamlit as st
import requests
import pandas as pd
import ee

from src.earth_engine import get_soil_moisture


# ============================================================
# DISTRICT COORDINATES
# ============================================================

DISTRICT_COORDINATES = {
    "Coimbatore": (11.0168, 76.9558),
    "Chennai": (13.0827, 80.2707),
    "Madurai": (9.9252, 78.1198),
    "Salem": (11.6643, 78.1460),
    "Trichy": (10.7905, 78.7047),
    "Erode": (11.3410, 77.7172),
    "Tiruppur": (11.1085, 77.3411),
    "Namakkal": (11.2189, 78.1674),
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriNova - CropLogic Saathi",
    page_icon="🌾",
    layout="wide",
)


# ============================================================
# GOOGLE EARTH ENGINE INITIALIZATION
# ============================================================

EE_PROJECT = "project-3dc0f771-1142-477c-9b2"

try:
    ee.Initialize(project=EE_PROJECT)
    EE_CONNECTED = True
    EE_ERROR = None
except Exception as e:
    EE_CONNECTED = False
    EE_ERROR = str(e)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# TITLE
# ============================================================

st.title("🌾 AgriNova")
st.subheader(
    "CropLogic-Saathi — Climate-Resilient Sowing Decision Support"
)

st.write(
    "An AI-assisted decision support system for rainfed farmers. "
    "It combines soil, rainfall and climate-related information "
    "to recommend whether to sow now, wait, or switch crops."
)


# ============================================================
# GOOGLE EARTH ENGINE STATUS
# ============================================================

if EE_CONNECTED:
    st.success("🛰️ Google Earth Engine connected successfully")
else:
    st.warning("⚠️ Google Earth Engine is not currently available.")

    if EE_ERROR:
        with st.expander("Earth Engine connection details"):
            st.code(EE_ERROR)


# ============================================================
# SIDEBAR — FARMER INPUTS
# ============================================================

st.sidebar.header("🌱 Farmer Inputs")

district = st.sidebar.selectbox(
    "District",
    [
        "Coimbatore",
        "Erode",
        "Tiruppur",
        "Salem",
        "Namakkal",
        "Chennai",
        "Madurai",
        "Trichy",
    ],
)

soil_type = st.sidebar.selectbox(
    "Soil Type",
    [
        "sandy_loam",
        "medium_black",
        "deep_black",
    ],
)

crop = st.sidebar.selectbox(
    "Current Crop",
    [
        "cotton",
        "soybean",
        "sorghum",
        "paddy",
    ],
)

rainfall = st.sidebar.number_input(
    "Rainfall Yesterday (mm)",
    min_value=0.0,
    max_value=300.0,
    value=20.0,
    step=1.0,
)

soil_moisture = st.sidebar.number_input(
    "Current Soil Moisture (mm)",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=1.0,
)

soil_ball_test = st.sidebar.selectbox(
    "Soil Ball Test",
    [
        "Moist",
        "Slightly Dry",
        "Dry",
    ],
)


# ============================================================
# GOOGLE EARTH ENGINE — SOIL MOISTURE
# ============================================================

st.subheader("🌍 Google Earth Engine")

latitude, longitude = DISTRICT_COORDINATES.get(
    district,
    DISTRICT_COORDINATES["Coimbatore"],
)

if EE_CONNECTED:

    with st.spinner(
        "Fetching environmental data from Google Earth Engine..."
    ):

        try:
            ee_moisture = get_soil_moisture(
                latitude,
                longitude,
            )
        except Exception as e:
            ee_moisture = None
            EE_MOISTURE_ERROR = str(e)

    if ee_moisture is not None:

        st.metric(
            "Earth Engine Soil Moisture",
            f"{ee_moisture:.3f}",
        )

        st.caption(
            f"Satellite/environmental observation for {district}"
        )

    else:

        st.warning(
            "Earth Engine soil-moisture data could not be retrieved."
        )

else:

    st.info(
        "Earth Engine is unavailable. "
        "The farmer-provided soil moisture will be used."
    )


# ============================================================
# EARTH ENGINE DEMO DATA
# ============================================================

def get_earth_engine_demo_data():

    if not EE_CONNECTED:
        return None

    try:

        image = ee.Image("NASA/NASADEM_HGT/001")

        elevation = image.select("elevation")

        region = ee.Geometry.Point(
            [longitude, latitude]
        )

        value = elevation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
        ).getInfo()

        return value

    except Exception:
        return None


# ============================================================
# FASTAPI DECISION FUNCTION
# ============================================================

def get_api_decision():

    payload = {
        "crop_name": crop,
        "soil_type": soil_type,
        "current_moisture_mm": soil_moisture,
        "rainfall_yesterday_mm": rainfall,

        # Default transition matrix used by the backend.
        "transition_matrix": [
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
        ],

        "num_simulations": 1000,
        "days_to_simulate": 5,
    }

    try:

        response = requests.post(
            f"{API_URL}/api/v1/decision",
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()

        st.error(
            f"API returned status code {response.status_code}"
        )

        st.code(response.text)

        return None

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to the FastAPI backend."
        )

        st.info(
            "Make sure Person 3's FastAPI backend is running."
        )

        st.code(
            "uvicorn api:app --reload"
        )

        return None

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ The FastAPI backend request timed out."
        )

        return None

    except Exception as e:

        st.error(
            f"API Error: {e}"
        )

        return None


# ============================================================
# FARMER INPUT SUMMARY
# ============================================================

st.divider()

st.subheader("📋 Current Field Conditions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "District",
        district,
    )

with col2:
    st.metric(
        "Crop",
        crop.title(),
    )

with col3:
    st.metric(
        "Rainfall Yesterday",
        f"{rainfall:.1f} mm",
    )

with col4:
    st.metric(
        "Soil Moisture",
        f"{soil_moisture:.1f} mm",
    )

st.caption(
    f"Soil type: {soil_type} | "
    f"Soil ball test: {soil_ball_test}"
)


# ============================================================
# DECISION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔍 Analyze Sowing Decision",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Analyzing climate and soil conditions..."
    ):

        result = get_api_decision()

    if result is not None:

        # ====================================================
        # DECISION
        # ====================================================

        st.divider()

        decision = result.get(
            "decision",
            "NO DECISION",
        )

        st.header("🌾 Decision")

        if "SWITCH" in decision.upper():

            st.warning(
                f"🌱 {decision}"
            )

        elif "WAIT" in decision.upper():

            st.info(
                f"⏳ {decision}"
            )

        else:

            st.success(
                f"🌱 {decision}"
            )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.subheader("📊 Establishment Probability")

        col1, col2, col3 = st.columns(3)

        germ_today = result.get(
            "germ_prob_today",
            0,
        )

        germ_wait = result.get(
            "germ_prob_wait",
            0,
        )

        germ_soybean = result.get(
            "germ_prob_soybean",
            0,
        )

        with col1:
            st.metric(
                "Sow Today",
                f"{germ_today * 100:.1f}%",
            )

        with col2:
            st.metric(
                "Wait 5 Days",
                f"{germ_wait * 100:.1f}%",
            )

        with col3:
            st.metric(
                "Soybean",
                f"{germ_soybean * 100:.1f}%",
            )


        # ====================================================
        # DECISION ANALYSIS
        # ====================================================

        st.subheader("📋 Decision Analysis")

        col1, col2, col3 = st.columns(3)

        current_moisture = result.get(
            "current_moisture",
            soil_moisture,
        )

        minimum_moisture = result.get(
            "min_moisture_required",
            0,
        )

        confidence = result.get(
            "confidence",
            0,
        )

        with col1:
            st.metric(
                "Current Soil Moisture",
                f"{current_moisture:.1f} mm",
            )

        with col2:
            st.metric(
                "Minimum Required",
                f"{minimum_moisture:.1f} mm",
            )

        with col3:
            st.metric(
                "Model Confidence",
                f"{confidence * 100:.1f}%",
            )


        # ====================================================
        # PROBABILITY COMPARISON
        # ====================================================

        st.subheader("📈 Probability Comparison")

        probability_data = pd.DataFrame(
            {
                "Scenario": [
                    "Sow Today",
                    "Wait 5 Days",
                    "Soybean",
                ],

                "Probability": [
                    germ_today * 100,
                    germ_wait * 100,
                    germ_soybean * 100,
                ],
            }
        )

        st.bar_chart(
            probability_data.set_index(
                "Scenario"
            )
        )


        # ====================================================
        # ECONOMIC COMPARISON
        # ====================================================

        economic = result.get(
            "economic_comparison"
        )

        if economic:

            st.subheader(
                "💰 Economic Comparison"
            )

            eco_col1, eco_col2, eco_col3 = st.columns(3)

            sow_today = economic.get(
                "sow_today",
                {},
            )

            wait = economic.get(
                "wait",
                {},
            )

            switch = economic.get(
                "switch",
                {},
            )

            with eco_col1:

                st.metric(
                    "Sow Today",
                    f"₹{sow_today.get('expected_profit', 0):,.0f}",
                )

            with eco_col2:

                st.metric(
                    "Wait 5 Days",
                    f"₹{wait.get('expected_profit', 0):,.0f}",
                )

            with eco_col3:

                st.metric(
                    "Switch Crop",
                    f"₹{switch.get('expected_profit', 0):,.0f}",
                )

            best_decision = economic.get(
                "best_decision"
            )

            best_profit = economic.get(
                "best_profit",
                0,
            )

            if best_decision:

                st.info(
                    f"💡 Economic recommendation: "
                    f"**{best_decision}** "
                    f"with expected profit of "
                    f"**₹{best_profit:,.0f}**."
                )


        # ====================================================
        # GOOGLE EARTH ENGINE
        # ====================================================

        st.divider()

        st.subheader(
            "🛰️ Google Earth Engine"
        )

        if EE_CONNECTED:

            with st.spinner(
                "Reading Earth Engine environmental data..."
            ):

                ee_data = get_earth_engine_demo_data()

            if ee_data:

                st.success(
                    "Earth Engine successfully returned "
                    "environmental data."
                )

                st.write(
                    "This demonstrates the Google Earth Engine "
                    "integration used as an environmental data layer."
                )

                st.json(
                    ee_data
                )

            else:

                st.info(
                    "Earth Engine is connected, but environmental "
                    "data could not be retrieved for this demo."
                )

        else:

            st.warning(
                "Earth Engine connection is unavailable."
            )


        # ====================================================
        # AI-ASSISTED EXPLANATION
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 AI-Assisted Recommendation"
        )

        st.write(
            f"""
            **Current conditions**

            - **District:** {district}
            - **Soil:** {soil_type}
            - **Current crop:** {crop}
            - **Rainfall yesterday:** {rainfall:.1f} mm
            - **Soil moisture:** {soil_moisture:.1f} mm
            - **Soil condition:** {soil_ball_test}

            The probabilistic decision engine evaluates possible
            future moisture conditions and estimates establishment
            probabilities.

            The system then compares the available actions and
            provides an explainable sowing recommendation.
            """
        )


        # ====================================================
        # RAW API RESPONSE
        # ====================================================

        with st.expander(
            "🔧 Technical API Response"
        ):

            st.json(
                result
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AgriNova | CropLogic-Saathi | "
    "AI for Climate Resilience"
)