"""
Shared rainfall-state classification logic for AgriNova.

This module contains lightweight model logic only.
It must not depend on the IMD preprocessing/data-loading stack.
"""

import math


def classify_rainfall(rainfall_mm):
    """
    Convert rainfall amount into a simple weather state.

    Thresholds are modelling assumptions for the current
    implementation and require future validation.

    States
    ------
    dry:
        rainfall < 1 mm

    drizzle:
        1 mm <= rainfall < 10 mm

    rain:
        rainfall >= 10 mm

    missing:
        rainfall is NaN
    """

    if isinstance(rainfall_mm, float) and math.isnan(rainfall_mm):
        return "missing"

    if rainfall_mm < 1.0:
        return "dry"

    if rainfall_mm < 10.0:
        return "drizzle"

    return "rain"