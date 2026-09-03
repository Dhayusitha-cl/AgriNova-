"""
Simplified daily soil-water balance model.

The model tracks soil water storage using:
    rainfall input
    evapotranspiration loss
    soil field capacity

This is a simplified decision-support model.
It is not a full hydrological model.
"""

import numpy as np

from src.soil_data import soils


def validate_soil_type(soil_type):
    """Validate that a supported soil type was supplied."""

    if soil_type not in soils:
        raise ValueError(
            f"Unknown soil type: {soil_type}"
        )

    return True


def update_soil_water(
    initial_water_mm,
    rainfall_mm,
    et_mm,
    soil_type,
):
    """
    Update soil-water storage for one day.

    Parameters
    ----------
    initial_water_mm : float
        Soil-water storage at the beginning of the day.

    rainfall_mm : float
        Daily rainfall.

    et_mm : float
        Daily evapotranspiration.

    soil_type : str
        Soil type.

    Returns
    -------
    dict
        Daily soil-water balance.
    """

    validate_soil_type(soil_type)

    initial_water_mm = float(initial_water_mm)
    rainfall_mm = float(rainfall_mm)
    et_mm = float(et_mm)

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    if not np.isfinite(initial_water_mm):
        raise ValueError(
            "initial_water_mm must be a finite number."
        )

    if not np.isfinite(rainfall_mm):
        raise ValueError(
            "rainfall_mm must be a finite number."
        )

    if not np.isfinite(et_mm):
        raise ValueError(
            "et_mm must be a finite number."
        )

    if initial_water_mm < 0:
        raise ValueError(
            "initial_water_mm cannot be negative."
        )

    if initial_water_mm > field_capacity:
        raise ValueError(
            "initial_water_mm cannot exceed field capacity."
        )

    if rainfall_mm < 0:
        raise ValueError(
            "rainfall_mm cannot be negative."
        )

    if et_mm < 0:
        raise ValueError(
            "et_mm cannot be negative."
        )

    water_before_et = initial_water_mm + rainfall_mm

    excess_water = max(
        water_before_et - field_capacity,
        0.0,
    )

    water_after_rainfall = min(
        water_before_et,
        field_capacity,
    )

    final_water = max(
        water_after_rainfall - et_mm,
        0.0,
    )

    return {
        "initial_water_mm": initial_water_mm,
        "rainfall_mm": rainfall_mm,
        "et_mm": et_mm,
        "water_before_et_mm": water_before_et,
        "excess_water_mm": excess_water,
        "final_water_mm": final_water,
        "field_capacity_mm": field_capacity,
    }


def simulate_soil_water(
    rainfall_series,
    et_series,
    soil_type,
    initial_water_mm=None,
):
    """
    Simulate soil-water storage over multiple days.

    Parameters
    ----------
    rainfall_series : sequence
        Daily rainfall amounts in mm.

    et_series : sequence
        Daily evapotranspiration amounts in mm.

    soil_type : str
        Soil type.

    initial_water_mm : float or None
        Initial soil-water storage.

        If None, starts at 50% of field capacity.
        This 50% initialization is a modelling assumption used
        when observed initial soil-water storage is unavailable.
        It is not an observed field measurement.

    Returns
    -------
    list[dict]
        Daily soil-water balance results.
    """

    validate_soil_type(soil_type)

    rainfall_series = list(rainfall_series)
    et_series = list(et_series)

    if len(rainfall_series) != len(et_series):
        raise ValueError(
            "rainfall_series and et_series must have the same length."
        )

    if len(rainfall_series) == 0:
        raise ValueError(
            "At least one day is required."
        )

    field_capacity = float(
        soils[soil_type]["field_capacity_mm"]
    )

    if initial_water_mm is None:
        current_water = field_capacity * 0.5
    else:
        current_water = float(initial_water_mm)

    results = []

    for day, (rainfall, et) in enumerate(
        zip(rainfall_series, et_series),
        start=1,
    ):

        balance = update_soil_water(
            initial_water_mm=current_water,
            rainfall_mm=rainfall,
            et_mm=et,
            soil_type=soil_type,
        )

        balance["day"] = day

        results.append(balance)

        current_water = balance["final_water_mm"]

    return results