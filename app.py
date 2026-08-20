import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from crop_data import crops
from soil_data import soils
from weather_simulator import create_transition_matrix
from decision_engine import make_decision

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
            
            result = make_decision(
                crop_name=crop_name,
                soil_type=soil_type,
                current_moisture_mm=current_moisture,
                rainfall_yesterday_mm=rainfall_24h,
                transition_matrix=transition_matrix,
                num_simulations=500,
                days_to_simulate=max(14, crops[crop_name]["germination_days"])
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
    st.markdown(f"## {result['color']} DECISION: {result['decision']}")
    
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

    # Economic comparison
    st.markdown("### 💰 Economic Comparison")

    economic = result["economic_comparison"]

    eco_col1, eco_col2, eco_col3 = st.columns(3)

    with eco_col1:
        st.metric(
            "Sow Today",
            f"₹{economic['sow_today']['expected_profit']:,.0f}"
        )

    with eco_col2:
        st.metric(
            "Wait 5 Days",
            f"₹{economic['wait']['expected_profit']:,.0f}"
        )

    with eco_col3:
        st.metric(
            "Switch to Soybean",
            f"₹{economic['switch']['expected_profit']:,.0f}"
        )

    st.info(
        f"💡 Economic recommendation: "
        f"**{economic['best_decision']}** "
        f"with expected profit of "
        f"**₹{economic['best_profit']:,.0f}**."
    )
    # Visualization
    st.markdown(
    f"### 📈 Soil Moisture Trajectories "
    f"({result['num_simulations']} Simulations)"
)
    st.write("Each line represents one possible future. Green zone = adequate moisture for germination.")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i in range(0, min(100, result['trajectories'].shape[0])):
        ax.plot(result['trajectories'][i, :], alpha=0.15, color='blue', linewidth=0.5)
    
    mean_trajectory = result['trajectories'].mean(axis=0)
    ax.plot(mean_trajectory, color='red', linewidth=2, label='Average')
    
    ax.axhline(y=result['min_moisture_required'], color='green', 
               linestyle='--', linewidth=2, label='Min Moisture for Germination')
    
    ax.axhspan(result['min_moisture_required'], 80, alpha=0.1, color='green')
    
    ax.set_xlabel('Days from Today')
    ax.set_ylabel('Soil Moisture (mm)')
    ax.set_title('Soil Moisture Forecast: Monte Carlo Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Probability distribution
    st.markdown(
    f"### 🎲 Final Moisture Distribution (Day {result['days_to_simulate']})"
)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    final_moistures = result['trajectories'][:, -1]
    
    ax2.hist(final_moistures, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=result['min_moisture_required'], color='green', 
                linestyle='--', linewidth=2, label='Min Moisture Required')
    
    above_threshold = (final_moistures >= result['min_moisture_required']).sum() / len(final_moistures)
    
    ax2.set_xlabel('Soil Moisture (mm)')
    ax2.set_ylabel('Number of Simulations')
    ax2.set_title(
    f"Distribution of Soil Moisture after "
    f"{result['days_to_simulate']} Days\n"
    f"{above_threshold*100:.0f}% of simulations above "
    f"germination threshold"
)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig2)
    
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