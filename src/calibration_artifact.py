"""
Versioned calibration artifact for CropLogic-Saathi.

This module stores already-calibrated rainfall-model inputs for
production simulation.

It does not perform:
    - rainfall calibration
    - Monte Carlo simulation
    - backtesting
    - decision making

Calibration should happen offline. The resulting artifact can then
be loaded by the production simulation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np


ARTIFACT_SCHEMA_VERSION = "1.0"
RAINFALL_STATES = ("dry", "drizzle", "rain")
SOURCE_DATASET = "IMD gridded rainfall"
PREPROCESSING_VERSION = "1.0"
CALIBRATION_METHOD_VERSION = "1.0"


@dataclass
class CalibrationArtifact:
    """
    Immutable-in-practice container for calibrated rainfall behaviour.

    Attributes
    ----------
    location:
        Dataset/location identifier, e.g. "yavatmal".
    source_files:
        Historical files used during calibration.
    source_start_date:
        Earliest observation represented by the calibration data.
    source_end_date:
        Latest observation represented by the calibration data.
    monthly_transition_matrices:
        Mapping of month number to a 3x3 transition matrix.
    rainfall_samples:
        Mapping of (month, state) to observed rainfall amounts.
    fallback_transition_matrix:
        Overall transition matrix used when a monthly state has
        insufficient outgoing transitions.
    """

    location: str
    source_dataset: str
    preprocessing_version: str
    calibration_method_version: str
    source_files: list[str]
    source_start_date: str
    source_end_date: str
    monthly_transition_matrices: dict[int, np.ndarray]
    rainfall_samples: dict[tuple[int, str], np.ndarray]
    fallback_transition_matrix: np.ndarray

    def content_hash(self) -> str:
        """
        Return a deterministic SHA-256 identity for this artifact's content.

        The hash excludes the artifact identity itself so the identity
        can be verified after loading.
        """
        payload = self.to_dict()

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

        return hashlib.sha256(canonical).hexdigest()

    def validate(self) -> None:
        """Validate artifact structure before use."""

        if not self.location:
            raise ValueError("Artifact location must not be empty.")

        if not self.source_files:
            raise ValueError("Artifact must contain source files.")

        if not self.source_start_date:
            raise ValueError(
                "Artifact source_start_date must not be empty."
            )

        if not self.source_end_date:
            raise ValueError(
                "Artifact source_end_date must not be empty."
            )
        if not isinstance(self.source_dataset, str) or not self.source_dataset.strip():
            raise ValueError("Source dataset must not be empty.")

        if (
            not isinstance(self.preprocessing_version, str)
            or not self.preprocessing_version.strip()
        ):
            raise ValueError("Preprocessing version must not be empty.")

        if (
            not isinstance(self.calibration_method_version, str)
            or not self.calibration_method_version.strip()
        ):
            raise ValueError("Calibration method version must not be empty.")

        self._validate_matrix(
            self.fallback_transition_matrix,
            "fallback_transition_matrix",
        )

        expected_months = set(range(1, 13))

        if set(self.monthly_transition_matrices) != expected_months:
            raise ValueError(
                "monthly_transition_matrices must contain "
                "all months 1 through 12."
            )

        for month, matrix in self.monthly_transition_matrices.items():
            self._validate_matrix(
                matrix,
                f"monthly_transition_matrices[{month}]",
            )

        for month in range(1, 13):
            for state in RAINFALL_STATES:
                key = (month, state)

                if key not in self.rainfall_samples:
                    raise ValueError(
                        f"Missing rainfall samples for {key}."
                    )

                values = np.asarray(
                    self.rainfall_samples[key],
                    dtype=float,
                )

                if values.ndim != 1:
                    raise ValueError(
                        f"Rainfall samples for {key} must be 1-dimensional."
                    )

                if len(values) == 0:
                    raise ValueError(
                        f"Rainfall samples for {key} must not be empty."
                    )

                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Rainfall samples for {key} contain "
                        "non-finite values."
                    )

                if np.any(values < 0):
                    raise ValueError(
                        f"Rainfall samples for {key} contain "
                        "negative rainfall."
                    )

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
        name: str,
    ) -> None:
        """Validate a rainfall transition matrix."""

        matrix = np.asarray(matrix, dtype=float)

        if matrix.shape != (3, 3):
            raise ValueError(
                f"{name} must have shape (3, 3)."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{name} contains non-finite values."
            )

        if np.any(matrix < 0):
            raise ValueError(
                f"{name} contains negative probabilities."
            )

        row_sums = matrix.sum(axis=1)

        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"{name} rows must sum approximately to 1."
            )

    def to_dict(self) -> dict:
        """
        Convert the artifact into a JSON-compatible dictionary.

        Validation is performed before serialization.
        """

        self.validate()

        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_dataset": self.source_dataset,
            "preprocessing_version": self.preprocessing_version,
            "calibration_method_version": self.calibration_method_version,
            "artifact_type": "rainfall_calibration",
            "location": self.location,
            "source_files": list(self.source_files),
            "source_start_date": self.source_start_date,
            "source_end_date": self.source_end_date,
            "rainfall_states": list(RAINFALL_STATES),
            "fallback_transition_matrix": (
                self.fallback_transition_matrix.tolist()
            ),
            "monthly_transition_matrices": {
                str(month): matrix.tolist()
                for month, matrix
                in self.monthly_transition_matrices.items()
            },
            "rainfall_samples": {
                f"{month}:{state}": values.tolist()
                for (month, state), values
                in self.rainfall_samples.items()
            },
        }

    def save(self, path: str | Path) -> None:
        """Save the artifact as JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
            )

    @classmethod
    def from_dict(
        cls,
        data: dict,
    ) -> "CalibrationArtifact":
        """Construct and validate an artifact from JSON-compatible data."""

        if data.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported calibration artifact schema version: "
                f"{data.get('schema_version')!r}"
            )

        if data.get("artifact_type") != "rainfall_calibration":
            raise ValueError(
                "Unsupported calibration artifact type."
            )

        if tuple(data.get("rainfall_states", [])) != RAINFALL_STATES:
            raise ValueError(
                "Calibration artifact rainfall states do not match "
                "the supported model states."
            )

        monthly_matrices = {
            int(month): np.asarray(
                matrix,
                dtype=float,
            )
            for month, matrix
            in data["monthly_transition_matrices"].items()
        }

        rainfall_samples = {}

        for key, values in data["rainfall_samples"].items():
            month_text, state = key.split(":", 1)

            if state not in RAINFALL_STATES:
                raise ValueError(
                    f"Unsupported rainfall state in artifact: {state!r}"
                )

            rainfall_samples[
                (int(month_text), state)
            ] = np.asarray(
                values,
                dtype=float,
            )

        artifact = cls(
            location=data["location"],
            source_files=list(data["source_files"]),
            source_start_date=data["source_start_date"],
            source_end_date=data["source_end_date"],
            source_dataset=data["source_dataset"],
            preprocessing_version=data["preprocessing_version"],
            calibration_method_version=data["calibration_method_version"],
            monthly_transition_matrices=monthly_matrices,
            rainfall_samples=rainfall_samples,
            fallback_transition_matrix=np.asarray(
                data["fallback_transition_matrix"],
                dtype=float,
            ),
        )

        artifact.validate()

        return artifact

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CalibrationArtifact":
        """Load and validate a calibration artifact from JSON."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Calibration artifact not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return cls.from_dict(data)