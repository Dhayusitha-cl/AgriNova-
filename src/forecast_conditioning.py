"""
Forecast conditioning for rainfall-state probabilities.

This module contains the first forecast-assimilation layer for
CropLogic-Saathi.

The historical calibrated model remains the baseline. Forecast
evidence modifies the probability of a wet day, while preserving
the historical conditional split between drizzle and rain.

Forecast rainfall amounts are intentionally not used here because
forecast-error calibration has not yet been established.
"""

import math


RAINFALL_STATES = ("dry", "drizzle", "rain")
WET_STATES = ("drizzle", "rain")


def condition_rainfall_state_probabilities(
    historical_probabilities,
    forecast_probability,
    forecast_weight=0.0,
):
    """
    Condition historical rainfall-state probabilities using
    forecast wet-day probability.

    Parameters
    ----------
    historical_probabilities : mapping
        Probabilities for ``dry``, ``drizzle``, and ``rain``.

    forecast_probability : float
        Forecast probability that daily rainfall reaches the
        CropLogic-Saathi wet-day threshold of 1 mm.

    forecast_weight : float
        Strength of forecast influence.

        0.0 -> historical distribution unchanged.
        1.0 -> forecast wet probability fully determines the
               wet/dry split.

        Intermediate values linearly blend the historical and
        forecast wet probabilities.

    Returns
    -------
    dict
        Conditioned probabilities for dry, drizzle, and rain.

    Notes
    -----
    The forecast changes only the wet/dry probability.

    The historical drizzle/rain ratio among wet days is preserved.

    Forecast rainfall amount is intentionally not used here.
    """

    if not hasattr(
        historical_probabilities,
        "get",
    ):
        raise TypeError(
            "historical_probabilities must be a mapping."
        )

    probabilities = {}

    for state in RAINFALL_STATES:
        value = historical_probabilities.get(state)

        if value is None:
            raise ValueError(
                f"Missing probability for state '{state}'."
            )

        if isinstance(value, bool):
            raise ValueError(
                f"Probability for state '{state}' must be finite."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"Probability for state '{state}' must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"Probability for state '{state}' must be "
                "between 0 and 1."
            )

        probabilities[state] = value

    total = sum(probabilities.values())

    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Historical rainfall-state probabilities must sum to 1."
        )

    if isinstance(forecast_probability, bool):
        raise ValueError(
            "forecast_probability must be finite."
        )

    forecast_probability = float(forecast_probability)

    if not math.isfinite(forecast_probability):
        raise ValueError(
            "forecast_probability must be finite."
        )

    if not 0.0 <= forecast_probability <= 1.0:
        raise ValueError(
            "forecast_probability must be between 0 and 1."
        )

    if isinstance(forecast_weight, bool):
        raise ValueError(
            "forecast_weight must be finite."
        )

    forecast_weight = float(forecast_weight)

    if not math.isfinite(forecast_weight):
        raise ValueError(
            "forecast_weight must be finite."
        )

    if not 0.0 <= forecast_weight <= 1.0:
        raise ValueError(
            "forecast_weight must be between 0 and 1."
        )

    historical_wet_probability = (
        probabilities["drizzle"]
        + probabilities["rain"]
    )

    conditioned_wet_probability = (
        (1.0 - forecast_weight)
        * historical_wet_probability
        + forecast_weight
        * forecast_probability
    )

    if historical_wet_probability == 0.0:
        raise ValueError(
            "Historical wet-day probability must be greater "
            "than zero to preserve the drizzle/rain split."
        )

    drizzle_fraction = (
        probabilities["drizzle"]
        / historical_wet_probability
    )

    rain_fraction = (
        probabilities["rain"]
        / historical_wet_probability
    )

    conditioned = {
        "dry": 1.0 - conditioned_wet_probability,
        "drizzle": (
            conditioned_wet_probability
            * drizzle_fraction
        ),
        "rain": (
            conditioned_wet_probability
            * rain_fraction
        ),
    }

    return conditioned