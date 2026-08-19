"""
Economic Engine for AgriNova
Calculates expected profit/loss for each sowing decision
"""

def calculate_economic_outcome(crop_name, soil_type, decision, 
                                germination_prob, rainfall_yesterday_mm,
                                current_moisture_mm):
    """
    Calculate expected monetary outcome for each decision
    
    Parameters:
    - crop_name: 'cotton', 'soybean', etc.
    - soil_type: 'sandy_loam', 'medium_black', 'deep_black'
    - decision: 'sow_today', 'wait', 'switch'
    - germination_prob: probability of successful germination (0 to 1)
    - rainfall_yesterday_mm: rainfall in last 24 hours
    - current_moisture_mm: current soil moisture
    
    Returns:
    - Dictionary with expected profit and breakdown
    """
    
    from crop_data import crops
    from soil_data import soils
    
    crop = crops[crop_name]
    soil = soils[soil_type]
    soybean = crops['soybean']
    
    # Helper function to calculate profit
    def calculate_profit(crop_name, yield_per_acre, price_per_quintal, 
                          seed_cost, success_probability):
        """
        Expected profit = (Success prob × Yield × Price) - Seed cost
        
        If germination fails, farmer loses seed cost entirely
        """
        expected_revenue = success_probability * yield_per_acre * price_per_quintal
        expected_profit = expected_revenue - seed_cost
        return expected_profit
    
    if decision == 'sow_today':
        # Scenario A: Sow primary crop today
        expected_profit = calculate_profit(
            crop_name,
            crop['average_yield_per_acre'],
            crop['market_price_per_quintal'],
            crop['seed_cost_per_acre'],
            germination_prob
        )
        
        # If germination fails, farmer can re-sow soybean late
        # Late soybean has 30% lower yield
        failure_prob = 1 - germination_prob
        late_soybean_yield = soybean['average_yield_per_acre'] * 0.7
        late_soybean_profit = calculate_profit(
            'soybean',
            late_soybean_yield,
            soybean['market_price_per_quintal'],
            soybean['seed_cost_per_acre'],
            0.8  # soybean germinates better
        )
        
        # Total expected value = success case + failure case
        total_expected = (
            germination_prob * expected_profit + 
            failure_prob * late_soybean_profit
        )
        
        return {
            'decision': 'Sow Today',
            'expected_profit': total_expected,
            'success_probability': germination_prob,
            'best_case_profit': crop['average_yield_per_acre'] * crop['market_price_per_quintal'] - crop['seed_cost_per_acre'],
            'worst_case_profit': late_soybean_profit,
            'risk_level': 'Low' if germination_prob > 0.7 else 'Medium' if germination_prob > 0.5 else 'High'
        }
    
    elif decision == 'wait':
        # Scenario B: Wait 5 days
        # If rain comes (probability from Markov Chain), better germination
        # If no rain, switch to late soybean
        
        rain_probability = 0.65  # based on historical monsoon patterns
        
        # With rain: better germination, slight yield loss from delay
        improved_germination = min(0.85, germination_prob + 0.3)
        delay_yield_loss = 0.94  # 6% yield loss from 5-day delay
        
        with_rain_profit = calculate_profit(
            crop_name,
            crop['average_yield_per_acre'] * delay_yield_loss,
            crop['market_price_per_quintal'],
            crop['seed_cost_per_acre'],
            improved_germination
        )
        
        # Without rain: switch to late soybean
        late_soybean_yield = soybean['average_yield_per_acre'] * 0.65
        no_rain_profit = calculate_profit(
            'soybean',
            late_soybean_yield,
            soybean['market_price_per_quintal'],
            soybean['seed_cost_per_acre'],
            0.7
        )
        
        # Expected value
        total_expected = (
            rain_probability * with_rain_profit + 
            (1 - rain_probability) * no_rain_profit
        )
        
        return {
            'decision': 'Wait 5 Days',
            'expected_profit': total_expected,
            'success_probability': improved_germination,
            'best_case_profit': with_rain_profit,
            'worst_case_profit': no_rain_profit,
            'risk_level': 'Low' if rain_probability > 0.7 else 'Medium'
        }
    
    elif decision == 'switch':
        # Scenario C: Switch to soybean now
        soybean_germination = 0.75  # soybean more drought-tolerant
        
        expected_profit = calculate_profit(
            'soybean',
            soybean['average_yield_per_acre'],
            soybean['market_price_per_quintal'],
            soybean['seed_cost_per_acre'],
            soybean_germination
        )
        
        return {
            'decision': 'Switch to Soybean',
            'expected_profit': expected_profit,
            'success_probability': soybean_germination,
            'best_case_profit': soybean['average_yield_per_acre'] * soybean['market_price_per_quintal'] - soybean['seed_cost_per_acre'],
            'worst_case_profit': expected_profit,
            'risk_level': 'Low'
        }
    
    else:
        return {
            'decision': 'Unknown',
            'expected_profit': 0,
            'success_probability': 0,
            'best_case_profit': 0,
            'worst_case_profit': 0,
            'risk_level': 'Unknown'
        }


def compare_all_decisions(crop_name, soil_type, 
                           germ_prob_today, germ_prob_wait, germ_prob_soybean,
                           rainfall_yesterday_mm, current_moisture_mm):
    """
    Compare all three decisions and return the best one
    """
    
    # Calculate economic outcome for each decision
    sow_today = calculate_economic_outcome(
        crop_name, soil_type, 'sow_today', 
        germ_prob_today, rainfall_yesterday_mm, current_moisture_mm
    )
    
    wait = calculate_economic_outcome(
        crop_name, soil_type, 'wait', 
        germ_prob_wait, rainfall_yesterday_mm, current_moisture_mm
    )
    
    switch = calculate_economic_outcome(
        crop_name, soil_type, 'switch', 
        germ_prob_soybean, rainfall_yesterday_mm, current_moisture_mm
    )
    
    # Find best decision
    decisions = [sow_today, wait, switch]
    best_decision = max(decisions, key=lambda x: x['expected_profit'])
    
    # Calculate advantage over other options
    for d in decisions:
        d['advantage_over_others'] = d['expected_profit'] - min(
            other['expected_profit'] for other in decisions if other != d
        )
    
    return {
        'sow_today': sow_today,
        'wait': wait,
        'switch': switch,
        'best_decision': best_decision['decision'],
        'best_profit': best_decision['expected_profit'],
        'all_decisions': decisions
    }


def format_currency(amount):
    """
    Format number as Indian Rupees
    Example: 45000 → ₹45,000
    """
    return f"₹{amount:,.0f}"