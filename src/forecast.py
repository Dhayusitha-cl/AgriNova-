"""
Forecast input models for CropLogic-Saathi.

This module defines the provider-neutral representation of
near-term weather forecast information.

Forecast values are external evidence. They are not treated
as deterministic future observations or guarantees.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math

WET_RAINFALL_THRESHOLD_MM = 1.0

@dataclass(frozen=True)
class DailyForecast:
    """
    Forecast information for one calendar day.

    precipitation_probability represents the probability that
    daily rainfall reaches the CropLogic-Saathi wet-day threshold
    (currently 1 mm).

    rainfall_mm is retained as provider-supplied forecast rainfall
    information, but is not currently used to deterministically
    generate simulated rainfall amounts.
    """

    forecast_date: date
    rainfall_mm: float
    precipitation_probability: float

    def __post_init__(self):
        if (
            not isinstance(self.forecast_date, date)
            or isinstance(self.forecast_date, datetime)
        ):
            raise TypeError(
                "forecast_date must be a date."
            )

        if isinstance(self.rainfall_mm, bool):
            raise ValueError(
                "rainfall_mm must be a finite number."
            )

        if not math.isfinite(float(self.rainfall_mm)):
            raise ValueError(
                "rainfall_mm must be a finite number."
            )

        if self.rainfall_mm < 0:
            raise ValueError(
                "rainfall_mm cannot be negative."
            )

        if isinstance(
            self.precipitation_probability,
            bool,
        ):
            raise ValueError(
                "precipitation_probability must be a finite number."
            )

        if not math.isfinite(
            float(self.precipitation_probability)
        ):
            raise ValueError(
                "precipitation_probability must be a finite number."
            )

        if not 0.0 <= self.precipitation_probability <= 1.0:
            raise ValueError(
                "precipitation_probability must be between 0 and 1."
            )


@dataclass(frozen=True)
class WeatherForecast:
    """
    Provider-neutral near-term weather forecast.

    The forecast is evidence supplied to the decision system.
    It is not itself a deterministic weather trajectory.
    """

    source: str
    issued_at: datetime
    daily: tuple[DailyForecast, ...]

    def __post_init__(self):
        if not isinstance(self.source, str):
            raise TypeError(
                "source must be a string."
            )

        if not self.source.strip():
            raise ValueError(
                "source must not be empty."
            )

        if not isinstance(self.issued_at, datetime):
            raise TypeError(
                "issued_at must be a datetime."
            )

        if not isinstance(self.daily, tuple):
            raise TypeError(
                "daily must be a tuple of DailyForecast objects."
            )

        if not self.daily:
            raise ValueError(
                "Forecast must contain at least one daily entry."
            )

        if any(
            not isinstance(entry, DailyForecast)
            for entry in self.daily
        ):
            raise TypeError(
                "All forecast entries must be DailyForecast objects."
            )

        dates = [
            entry.forecast_date
            for entry in self.daily
        ]

        if dates != sorted(dates):
            raise ValueError(
                "Forecast dates must be in chronological order."
            )

        if len(dates) != len(set(dates)):
            raise ValueError(
                "Forecast dates must be unique."
            )