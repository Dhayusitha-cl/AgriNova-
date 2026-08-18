import numpy as np

def simulate_climate_change(transition_matrix, temperature_increase=2.0, 
                             rainfall_change_pct=-10):
    """
    Simulate how climate change affects weather patterns
    - Higher temperatures increase ET (soil dries faster)
    - Lower rainfall reduces moisture recharge
    - Both combined increase sowing failure risk
    
    Parameters:
    - transition_matrix: Original Markov Chain weather probabilities
    - temperature_increase: Degrees Celsius increase (default 2.0)
    - rainfall_change_pct: % change in rainfall (default -10%)
    
    Returns:
    - adjusted_matrix: Modified transition probabilities
    """
    
    # Copy original matrix
    adjusted_matrix = transition_matrix.copy()
    
    # Reduce transition TO rain states (columns 1 and 2)
    # Column 0 = dry, Column 1 = drizzle, Column 2 = rain
    adjusted_matrix[:, 2] *= (1 + rainfall_change_pct / 100)
    adjusted_matrix[:, 1] *= (1 + rainfall_change_pct / 200)
    
    # Re-normalize rows so probabilities sum to 1
    row_sums = adjusted_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    adjusted_matrix = adjusted_matrix / row_sums
    
    return adjusted_matrix


def calculate_et_with_warming(tmax, tmin, temperature_increase=2.0):
    """
    Calculate evapotranspiration with higher temperatures
    Uses simplified Hargreaves equation
    
    Parameters:
    - tmax: Maximum temperature (°C)
    - tmin: Minimum temperature (°C)
    - temperature_increase: Climate warming (°C)
    
    Returns:
    - ET value in mm/day
    """
    
    # Apply temperature increase
    tmax_adj = tmax + temperature_increase
    tmin_adj = tmin + temperature_increase
    tavg = (tmax_adj + tmin_adj) / 2
    
    # Simplified ET formula
    et = 0.5 * (tavg - 10)
    
    # Clamp between 1 and 12 mm/day
    return max(1, min(et, 12))


def run_climate_projection(crop_name, soil_type, current_moisture, 
                            rainfall_yesterday, transition_matrix,
                            climate_scenarios=None):
    """
    Run germination probability for different climate scenarios
    
    Parameters:
    - crop_name: 'cotton', 'soybean', 'sorghum', 'paddy'
    - soil_type: 'sandy_loam', 'medium_black', 'deep_black'
    - current_moisture: Current soil moisture (mm)
    - rainfall_yesterday: Yesterday's rainfall (mm)
    - transition_matrix: Original Markov Chain matrix
    - climate_scenarios: List of scenarios to test
    
    Returns:
    - results: Dictionary with germination probabilities for each scenario
    """
    
    # Import dependencies
    import sys
    sys.path.append('src')
    from crop_data import crops
    from soil_data import soils
    from decision_engine import simulate_soil_moisture, calculate_germination_probability
    
    if climate_scenarios is None:
        climate_scenarios = ['Current', '2030', '2050']
    
    results = {}
    
    for scenario in climate_scenarios:
        if scenario == 'Current':
            # No changes
            adjusted_matrix = transition_matrix
            temp_increase = 0
            
        elif scenario == '2030':
            # Moderate climate change
            adjusted_matrix = simulate_climate_change(
                transition_matrix, 
                temperature_increase=1.0, 
                rainfall_change_pct=-5
            )
            temp_increase = 1.0
            
        elif scenario == '2050':
            # Severe climate change
            adjusted_matrix = simulate_climate_change(
                transition_matrix, 
                temperature_increase=2.5, 
                rainfall_change_pct=-15
            )
            temp_increase = 2.5
            
        else:
            continue
        
        # Run simulation with adjusted weather
        trajectories = simulate_soil_moisture(
            initial_moisture=current_moisture,
            rainfall_today=rainfall_yesterday,
            transition_matrix=adjusted_matrix,
            soil_type=soil_type,
            num_simulations=300,
            days=7
        )
        
        # Calculate germination probability
        soil = soils[soil_type]
        crop = crops[crop_name]
        min_moisture = soil['field_capacity_mm'] * (crop['min_moisture_pct'] / 100)
        
        germ_prob = calculate_germination_probability(
            trajectories, min_moisture, crop['germination_days']
        )
        
        results[scenario] = {
            'germination_probability': germ_prob,
            'trajectories': trajectories,
            'temperature_increase': temp_increase,
            'min_moisture': min_moisture
        }
    
    return results


def get_climate_impact_summary(results):
    """
    Generate a text summary of climate impact
    
    Parameters:
    - results: Dictionary from run_climate_projection
    
    Returns:
    - summary: String with key findings
    """
    
    if 'Current' not in results:
        return "No current baseline data available."
    
    current_prob = results['Current']['germination_probability']
    
    summary = f"**Current germination probability:** {current_prob*100:.0f}%\n\n"
    
    if '2030' in results:
        prob_2030 = results['2030']['germination_probability']
        change = (prob_2030 - current_prob) * 100
        summary += f"**2030 projection:** {prob_2030*100:.0f}% "
        
        if change < -5:
            summary += f"(⚠️ {change:.0f}% decrease)\n\n"
        else:
            summary += f"({change:.0f}% change)\n\n"
    
    if '2050' in results:
        prob_2050 = results['2050']['germination_probability']
        change = (prob_2050 - current_prob) * 100
        summary += f"**2050 projection:** {prob_2050*100:.0f}% "
        
        if change < -10:
            summary += f"(🚨 {change:.0f}% significant decrease)\n\n"
        else:
            summary += f"({change:.0f}% change)\n\n"
    
    # Add recommendation
    if current_prob > 0.6 and '2050' in results:
        if results['2050']['germination_probability'] < 0.5:
            summary += "**Recommendation:** Consider switching to drought-tolerant crops (sorghum, millets) for long-term climate resilience."
    
    return summary
