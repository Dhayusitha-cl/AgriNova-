import numpy as np

def make_decision(crop_name, soil_type, current_moisture_mm, 
                  rainfall_yesterday_mm, transition_matrix, 
                  num_simulations=500, days_to_simulate=7):
    """
    THE CORE LOGIC:
    1. Simulate weather for next 7 days
    2. Calculate soil moisture trajectory
    3. Check if germination succeeds
    4. Compare decisions
    """
    from crop_data import crops
    from soil_data import soils
    from weather_simulator import generate_weather_sequence
    
    crop = crops[crop_name]
    soil = soils[soil_type]
    
    min_moisture = soil['field_capacity_mm'] * (crop['min_moisture_pct'] / 100)
    germination_period = crop['germination_days']
    
    # SCENARIO A: Sow Today
    trajectories = simulate_soil_moisture(
        initial_moisture=current_moisture_mm,
        rainfall_today=rainfall_yesterday_mm,
        transition_matrix=transition_matrix,
        soil_type=soil_type,
        num_simulations=num_simulations,
        days=days_to_simulate
    )
    
    germ_prob_today = calculate_germination_probability(
        trajectories, min_moisture, germination_period
    )
    
    # SCENARIO B: Wait 5 days
    wait_simulations = simulate_with_wait(
        current_moisture=current_moisture_mm,
        transition_matrix=transition_matrix,
        soil_type=soil_type,
        wait_days=5,
        num_simulations=num_simulations
    )
    
    germ_prob_wait = calculate_germination_probability(
        wait_simulations, min_moisture, germination_period
    )
    
    # SCENARIO C: Alternative crop (soybean)
    soybean = crops['soybean']
    soybean_min_moisture = soil['field_capacity_mm'] * (soybean['min_moisture_pct'] / 100)
    
    germ_prob_soybean = calculate_germination_probability(
        trajectories, soybean_min_moisture, soybean['germination_days']
    )
    
    # DECISION LOGIC
    if germ_prob_today >= 0.7:
        decision = "SOW TODAY"
        color = "🟢"
        confidence = germ_prob_today
    elif germ_prob_wait > germ_prob_today + 0.2:
        decision = "WAIT 5 DAYS"
        color = "🟡"
        confidence = germ_prob_wait
    elif germ_prob_soybean > germ_prob_today + 0.15:
        decision = "SWITCH TO SOYBEAN"
        color = "🔴"
        confidence = germ_prob_soybean
    else:
        decision = "WAIT 5 DAYS"
        color = "🟡"
        confidence = max(germ_prob_wait, germ_prob_today)
    
    return {
        'decision': decision,
        'color': color,
        'germ_prob_today': germ_prob_today,
        'germ_prob_wait': germ_prob_wait,
        'germ_prob_soybean': germ_prob_soybean,
        'confidence': confidence,
        'trajectories': trajectories,
        'wait_simulations': wait_simulations,
        'current_moisture': current_moisture_mm,
        'min_moisture_required': min_moisture
    }


def simulate_soil_moisture(initial_moisture, rainfall_today, transition_matrix, 
                           soil_type, num_simulations=500, days=7):
    """Run Monte Carlo simulation of soil moisture"""
    from soil_data import soils
    from weather_simulator import generate_weather_sequence
    
    soil = soils[soil_type]
    all_trajectories = np.zeros((num_simulations, days + 1))
    all_trajectories[:, 0] = initial_moisture
    
    for sim in range(num_simulations):
        weather = generate_weather_sequence(transition_matrix, days)
        
        moisture = initial_moisture
        for day in range(days):
            if day == 0:
                moisture += rainfall_today * 0.8
            else:
                moisture += weather[day-1]['rainfall'] * 0.8
            
            et = calculate_et_daily(weather[day]['tmax'], weather[day]['tmin'])
            moisture -= et
            
            if moisture > soil['field_capacity_mm']:
                moisture = soil['field_capacity_mm'] + (moisture - soil['field_capacity_mm']) * 0.3
            
            moisture = max(0, moisture)
            all_trajectories[sim, day + 1] = moisture
    
    return all_trajectories


def simulate_with_wait(current_moisture, transition_matrix, soil_type, 
                       wait_days=5, num_simulations=500):
    """Simulate what happens if farmer waits"""
    from soil_data import soils
    from weather_simulator import generate_weather_sequence
    
    soil = soils[soil_type]
    all_trajectories = np.zeros((num_simulations, 12))
    
    for sim in range(num_simulations):
        weather = generate_weather_sequence(transition_matrix, wait_days + 7)
        
        moisture = current_moisture
        trajectory = []
        
        for day in range(wait_days):
            moisture += weather[day]['rainfall'] * 0.8
            et = calculate_et_daily(weather[day]['tmax'], weather[day]['tmin'])
            moisture -= et
            moisture = max(0, min(moisture, soil['field_capacity_mm']))
            trajectory.append(moisture)
        
        for day in range(wait_days, wait_days + 7):
            moisture += weather[day]['rainfall'] * 0.8
            et = calculate_et_daily(weather[day]['tmax'], weather[day]['tmin'])
            moisture -= et
            moisture = max(0, min(moisture, soil['field_capacity_mm']))
            trajectory.append(moisture)
        
        all_trajectories[sim, :] = trajectory
    
    return all_trajectories


def calculate_germination_probability(trajectories, min_moisture, germination_period):
    """Calculate % of simulations where moisture stays above threshold"""
    success_count = 0
    total = trajectories.shape[0]
    
    for sim in range(total):
        moisture_ok = trajectories[sim, :] >= min_moisture
        
        for day in range(len(moisture_ok) - germination_period + 1):
            if all(moisture_ok[day:day + germination_period]):
                success_count += 1
                break
    
    return success_count / total


def calculate_et_daily(tmax, tmin):
    """Simplified ET calculation"""
    tavg = (tmax + tmin) / 2
    et = 0.5 * (tavg - 10)
    return max(1, min(et, 10))