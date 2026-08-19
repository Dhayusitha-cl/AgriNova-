import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests

from src.crop_data import crops
from src.soil_data import soils
from src.weather_simulator import create_transition_matrix
from src.decision_engine import make_decision
API_URL = "http://127.0.0.1:8000"
def get_api_decision(crop_name, soil_type, current_moisture, rainfall_24h, transition_matrix):
    payload = {
        "crop_name": crop_name,
        "soil_type": soil_type,
        "current_moisture_mm": current_moisture,
        "rainfall_yesterday_mm": rainfall_24h,
        "transition_matrix": transition_matrix.tolist(),
        "num_simulations": 500,
        "days_to_simulate": 7
    }

    response = requests.post(
        f"{API_URL}/api/v1/decision",
        json=payload
    )

    response.raise_for_status()

    return response.json()

# Page config
st.set_page_config(
    page_title="AgriNova",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 AgriNova")
st.subheader("Climate-Resilient Sowing Decision Support for Smallholder Farmers")

# Sidebar - Farmer Inputs
st.sidebar.header("📍 Farmer Information")

district = st.sidebar.selectbox(
    "District",
    ["Yavatmal (Maharashtra)", "Anantapur (Andhra Pradesh)", "Kalahandi (Odisha)"]
)

soil_type_display = st.sidebar.selectbox(
    "Soil Type",
    ["Sandy Loam", "Medium Black (Regur)", "Deep Black (Cotton Soil)"]
)

soil_map = {
    "Sandy Loam": "sandy_loam",
    "Medium Black (Regur)": "medium_black",
    "Deep Black (Cotton Soil)": "deep_black"
}
soil_type = soil_map[soil_type_display]

crop_display = st.sidebar.selectbox(
    "Primary Crop",
    ["Cotton (BT)", "Soybean (JS-335)", "Sorghum/Jowar (CSH-16)", "Paddy (MTU-1010)"]
)

crop_map = {
    "Cotton (BT)": "cotton",
    "Soybean (JS-335)": "soybean",
    "Sorghum/Jowar (CSH-16)": "sorghum",
    "Paddy (MTU-1010)": "paddy"
}
crop_name = crop_map[crop_display]

st.sidebar.markdown("---")
st.sidebar.header("🌧️ Today's Conditions")

rainfall_24h = st.sidebar.slider(
    "Rainfall in last 24 hours (mm)",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)

st.sidebar.markdown("### 🖐️ Soil Ball Test")
st.sidebar.write("Squeeze a handful of soil. What happens?")

ball_test = st.sidebar.radio(
    "Soil ball test result",
    [
        "Does not form a ball (very dry)",
        "Forms a ball but crumbles (moist)",
        "Forms a ball, holds shape (wet)",
        "Water squeezes out (saturated)"
    ]
)

ball_to_moisture = {
    "Does not form a ball (very dry)": 15,
    "Forms a ball but crumbles (moist)": 30,
    "Forms a ball, holds shape (wet)": 50,
    "Water squeezes out (saturated)": 70
}
current_moisture = ball_to_moisture[ball_test]

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Current Conditions Summary")
    st.write(f"**District:** {district}")
    st.write(f"**Soil Type:** {soil_type_display}")
    st.write(f"**Crop:** {crop_display}")
    st.write(f"**Current Soil Moisture:** ~{current_moisture} mm")
    st.write(f"**Field Capacity:** {soils[soil_type]['field_capacity_mm']} mm")
    moisture_pct = current_moisture / soils[soil_type]['field_capacity_mm'] * 100
    st.write(f"**Moisture as % of Field Capacity:** {moisture_pct:.0f}%")

with col2:
    st.markdown("### 🎯 Get Recommendation")
    if st.button("🔮 Get Sowing Recommendation", type="primary", use_container_width=True):
        with st.spinner("Running 500 simulations..."):
            transition_matrix = create_transition_matrix(week_number=3)
            result = get_api_decision(
    crop_name=crop_name,
    soil_type=soil_type,
    current_moisture=current_moisture,
    rainfall_24h=rainfall_24h,
    transition_matrix=transition_matrix
)
            
                       
            st.session_state['decision_result'] = result
            st.session_state['current_moisture'] = current_moisture
            st.session_state['crop_display'] = crop_display
            st.session_state['crop_name'] = crop_name

# Display decision if available
if 'decision_result' in st.session_state:
    result = st.session_state['decision_result']
    current_moisture = st.session_state['current_moisture']
    crop_display = st.session_state['crop_display']
    crop_name = st.session_state['crop_name']
    
    st.markdown("---")
    st.markdown(f"## 🌾 DECISION: {result['decision']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Germination Probability (Sow Today)",
            f"{result['germ_prob_today']*100:.0f}%"
        )
    
    with col2:
        st.metric(
            "Germination Probability (Wait 5 Days)",
            f"{result['germ_prob_wait']*100:.0f}%"
        )
    
    with col3:
        st.metric(
            "Germination Probability (Soybean)",
            f"{result['germ_prob_soybean']*100:.0f}%"
        )
    
       # Decision Analysis
    st.markdown("### 📊 Decision Analysis")

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.metric(
            "Current Soil Moisture",
            f"{result['current_moisture']:.1f} mm"
        )

    with analysis_col2:
        st.metric(
            "Minimum Moisture Required",
            f"{result['min_moisture_required']:.1f} mm"
        )

    st.info(
        f"The API estimates a {result['confidence']*100:.1f}% "
        f"confidence in this recommendation."
    )
    
        # Decision comparison
    st.markdown("### 📊 Decision Analysis")

    chart_data = {
        "Sow Today": result["germ_prob_today"] * 100,
        "Wait 5 Days": result["germ_prob_wait"] * 100,
        "Soybean": result["germ_prob_soybean"] * 100
    }

    st.bar_chart(chart_data)

    st.info(
        f"Recommendation confidence: "
        f"{result['confidence'] * 100:.1f}%"
    )
    
    # Reasoning
    st.markdown("### 💡 Reasoning")
    
    reasoning_text = f"""
    **Current Situation:**
    - Soil moisture: {current_moisture} mm
    - Minimum required for {crop_display} germination: {result['min_moisture_required']:.0f} mm
    - Germination period: {crops[crop_name]['germination_days']} days
    
    **Analysis:**
    - Probability of successful germination if sown today: **{result['germ_prob_today']*100:.0f}%**
    - Probability if farmer waits 5 days: **{result['germ_prob_wait']*100:.0f}%**
    - Probability with alternative crop (Soybean): **{result['germ_prob_soybean']*100:.0f}%**
    
    **Recommendation Confidence:** {result['confidence']*100:.0f}%
    """
    
    st.markdown(reasoning_text)

# Footer
st.markdown("---")
st.markdown("*AgriNova: Making climate-resilient decisions accessible to every farmer*")