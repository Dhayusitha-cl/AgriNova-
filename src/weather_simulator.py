import numpy as np

def create_transition_matrix(week_number, district='yavatmal'):
    """
    Returns 3x3 transition matrix for Markov Chain
    Rows: today's weather, Columns: tomorrow's weather
    States: dry, drizzle, rain
    """
    if week_number == 1:
        return np.array([
            [0.75, 0.18, 0.07],
            [0.55, 0.30, 0.15],
            [0.40, 0.35, 0.25]
        ])
    elif week_number == 2:
        return np.array([
            [0.65, 0.25, 0.10],
            [0.45, 0.35, 0.20],
            [0.35, 0.35, 0.30]
        ])
    elif week_number == 3:
        return np.array([
            [0.55, 0.30, 0.15],
            [0.40, 0.35, 0.25],
            [0.30, 0.35, 0.35]
        ])
    elif week_number == 4:
        return np.array([
            [0.50, 0.32, 0.18],
            [0.35, 0.38, 0.27],
            [0.28, 0.35, 0.37]
        ])
    else:
        return np.array([
            [0.45, 0.35, 0.20],
            [0.30, 0.40, 0.30],
            [0.25, 0.35, 0.40]
        ])


def generate_weather_sequence(transition_matrix, num_days):
    """
    Generate weather sequence using Markov Chain
    Returns list of dicts with state, rainfall, tmax, tmin
    """
    states = ['dry', 'drizzle', 'rain']
    
    rainfall_amounts = {
        'dry': lambda: 0,
        'drizzle': lambda: np.random.uniform(1, 10),
        'rain': lambda: np.random.uniform(10, 40)
    }
    
    temps = {
        'dry': (35, 26),
        'drizzle': (32, 24),
        'rain': (28, 22)
    }
    
    current_state_idx = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
    weather_sequence = []
    
    for day in range(num_days):
        state = states[current_state_idx]
        rainfall = rainfall_amounts[state]()
        tmax, tmin = temps[state]
        
        weather_sequence.append({
            'state': state,
            'rainfall': rainfall,
            'tmax': tmax + np.random.uniform(-2, 2),
            'tmin': tmin + np.random.uniform(-1, 1)
        })
        
        current_state_idx = np.random.choice(
            [0, 1, 2], 
            p=transition_matrix[current_state_idx]
        )
    
    return weather_sequence


def calculate_et_hargreaves(tmax, tmin, latitude=20.39, day_of_year=180):
    """
    Calculate reference evapotranspiration using Hargreaves equation
    """
    tavg = (tmax + tmin) / 2
    Ra = 40 + 5 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    et0 = 0.0023 * Ra * np.sqrt(max(tmax - tmin, 1)) * (tavg + 17.8)
    return et0


def calculate_et_daily(tmax, tmin):
    """
    Simplified daily ET calculation
    """
    tavg = (tmax + tmin) / 2
    et = 0.5 * (tavg - 10)
    return max(1, min(et, 10))