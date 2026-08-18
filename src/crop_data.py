# Crop parameters (based on ICAR/FAO data)

crops = {
    'cotton': {
        'name': 'Cotton (BT)',
        'min_moisture_pct': 25,
        'germination_days': 8,
        'seed_cost_per_acre': 3400,
        'market_price_per_quintal': 6500,
        'average_yield_per_acre': 10,
        'drought_tolerance': 4,
        'optimal_sowing_window': 'June 15 - July 15',
        'yield_loss_per_day_pct': 1.2
    },
    'soybean': {
        'name': 'Soybean (JS-335)',
        'min_moisture_pct': 20,
        'germination_days': 6,
        'seed_cost_per_acre': 2100,
        'market_price_per_quintal': 4200,
        'average_yield_per_acre': 9,
        'drought_tolerance': 6,
        'optimal_sowing_window': 'June 20 - July 20',
        'yield_loss_per_day_pct': 0.8
    },
    'sorghum': {
        'name': 'Sorghum/Jowar (CSH-16)',
        'min_moisture_pct': 15,
        'germination_days': 5,
        'seed_cost_per_acre': 800,
        'market_price_per_quintal': 3500,
        'average_yield_per_acre': 12,
        'drought_tolerance': 9,
        'optimal_sowing_window': 'June 10 - July 25',
        'yield_loss_per_day_pct': 0.5
    },
    'paddy': {
        'name': 'Paddy (MTU-1010)',
        'min_moisture_pct': 30,
        'germination_days': 8,
        'seed_cost_per_acre': 1500,
        'market_price_per_quintal': 2200,
        'average_yield_per_acre': 20,
        'drought_tolerance': 2,
        'optimal_sowing_window': 'June 1 - July 10',
        'yield_loss_per_day_pct': 1.5
    }
}
