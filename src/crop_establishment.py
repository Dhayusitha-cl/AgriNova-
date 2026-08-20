"""
Crop establishment evaluation.

This module evaluates whether simulated soil-water conditions
are sufficient for crop establishment during the germination period.

Important:
    The moisture percentage used here is a normalized soil-water
    storage indicator:

        moisture_pct =
            soil_water_mm / field_capacity_mm * 100

    It is NOT a direct measurement of volumetric soil moisture.

    Establishment results are simulation outputs, not guarantees.
"""

from src.crop_data import crops
from src.soil_data import soils


def validate_crop(crop):
    """Validate that a supported crop was supplied."""

    if crop not in crops:
        raise ValueError(f"Unknown crop: {crop}")

    return True


def validate_soil_type(soil_type):
    """Validate that a supported soil type was supplied."""

    if soil_type not in soils:
        raise ValueError(f"Unknown soil type: {soil_type}")

    return True


def calculate_moisture_percentage(
    soil_water_mm,
    soil_type,
):
    """
    Convert soil-water storage into normalized moisture percentage.

    This is calculated relative to field capacity:

        moisture_pct =
            soil_water_mm / field_capacity_mm * 100
    """

    validate_soil_type(soil_type)

    soil_water_mm = float(soil_water_mm)
    field_capacity_mm = float(
        soils[soil_type]["field_capacity_mm"]
    )

    if soil_water_mm < 0:
        raise ValueError(
            "soil_water_mm cannot be negative."
        )

    if soil_water_mm > field_capacity_mm:
        raise ValueError(
            "soil_water_mm cannot exceed field capacity."
        )

    return (
        soil_water_mm / field_capacity_mm
    ) * 100.0


def evaluate_establishment(
    soil_water_results,
    crop,
    soil_type,
):
    """
    Evaluate crop establishment from soil-water simulation results.

    Establishment succeeds when the soil moisture percentage is
    at or above the crop's minimum moisture requirement on every
    day of the crop's germination period.

    Parameters
    ----------
    soil_water_results : list[dict]
        Output from simulate_soil_water().

    crop : str
        Crop identifier.

    soil_type : str
        Soil identifier.

    Returns
    -------
    dict
        Explainable establishment result.
    """

    validate_crop(crop)
    validate_soil_type(soil_type)

    if not soil_water_results:
        raise ValueError(
            "soil_water_results cannot be empty."
        )

    crop_parameters = crops[crop]

    germination_days = int(
        crop_parameters["germination_days"]
    )

    minimum_moisture_pct = float(
        crop_parameters["min_moisture_pct"]
    )

    if len(soil_water_results) < germination_days:
        raise ValueError(
            "Not enough soil-water simulation days "
            "for the crop germination period."
        )

    germination_results = soil_water_results[
        :germination_days
    ]

    daily_results = []

    for result in germination_results:

        moisture_pct = calculate_moisture_percentage(
            result["final_water_mm"],
            soil_type,
        )

        meets_requirement = (
            moisture_pct >= minimum_moisture_pct
        )

        daily_results.append(
            {
                "day": result["day"],
                "soil_water_mm": result["final_water_mm"],
                "moisture_pct": moisture_pct,
                "minimum_required_pct": minimum_moisture_pct,
                "meets_requirement": meets_requirement,
            }
        )

    successful_days = sum(
        item["meets_requirement"]
        for item in daily_results
    )

    establishment_success = (
        successful_days == germination_days
    )

    return {
        "crop": crop,
        "soil_type": soil_type,
        "germination_days": germination_days,
        "minimum_moisture_pct": minimum_moisture_pct,
        "successful_days": successful_days,
        "establishment_success": establishment_success,
        "daily_results": daily_results,
    }