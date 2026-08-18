# Soil parameters (based on NBSS & LUP data)

soils = {
    'sandy_loam': {
        'name': 'Sandy Loam',
        'water_holding_capacity_mm_per_cm': 1.2,
        'field_capacity_mm': 36,
        'infiltration_rate_mm_per_hr': 30,
        'drainage': 'high'
    },
    'medium_black': {
        'name': 'Medium Black (Regur)',
        'water_holding_capacity_mm_per_cm': 2.0,
        'field_capacity_mm': 60,
        'infiltration_rate_mm_per_hr': 15,
        'drainage': 'medium'
    },
    'deep_black': {
        'name': 'Deep Black (Cotton Soil)',
        'water_holding_capacity_mm_per_cm': 2.4,
        'field_capacity_mm': 72,
        'infiltration_rate_mm_per_hr': 8,
        'drainage': 'low'
    }
}
