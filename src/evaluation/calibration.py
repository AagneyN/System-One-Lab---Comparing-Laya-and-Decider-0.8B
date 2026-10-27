"""Temperature scaling fit only on development data."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .config import CALIBRATION_ROLES


def _check_inputs(
    probabilities: Sequence[Mapping[str, float]],
    y_true: Sequence[str],
    labels: Sequence[str],
) -> list[list[float]]:
    if not probabilities or len(probabilities) != len(y_true):
        raise ValueError("Probabilities and labels must have the same nonzero length.")

    matrix = []
    for row in probabilities:
        if set(row) != set(labels):
            raise ValueError("Each probability row must contain exactly the task labels.")
        values = [float(row[label]) for label in labels]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Probabilities must be finite and nonnegative.")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Each probability row must sum to 1.")
        matrix.append(values)

    if any(label not in labels for label in y_true):
        raise ValueError("A target label is outside the declared label set.")
    return matrix


def _nll(
    matrix: list[list[float]],
    y_true: Sequence[str],
    labels: Sequence[str],
    temperature: float,
) -> float:
    label_index = {label: index for index, label in enumerate(labels)}
    total = 0.0
    for row, truth in zip(matrix, y_true):
        logits = [math.log(max(probability, 1e-15)) / temperature for probability in row]
        peak = max(logits)
        exp_values = [math.exp(value - peak) for value in logits]
        denominator = sum(exp_values)
        true_probability = exp_values[label_index[truth]] / denominator
        total -= math.log(max(true_probability, 1e-15))
    return total / len(y_true)


@dataclass(frozen=True)
class TemperatureCalibrator:
    labels: tuple[str, ...]
    temperature: float
    fit_data_role: str
    sample_count: int
    nll_before: float
    nll_after: float

    @classmethod
    def fit(
        cls,
        probabilities: Sequence[Mapping[str, float]],
        y_true: Sequence[str],
        *,
        labels: Sequence[str],
        data_role: str,
        candidates: Sequence[float] | None = None,
    ) -> "TemperatureCalibrator":
        if data_role not in CALIBRATION_ROLES:
            raise ValueError(
                "Calibration fit data must be development_train or "
                "development_validation; test data is forbidden."
            )

        ordered_labels = tuple(labels)
        matrix = _check_inputs(probabilities, y_true, ordered_labels)
        if candidates is None:
            candidates = [step / 20 for step in range(5, 81)]

        valid_candidates = [
            float(value)
            for value in candidates
            if math.isfinite(float(value)) and float(value) > 0
        ]
        if not valid_candidates:
            raise ValueError("Provide at least one positive temperature candidate.")

        before = _nll(matrix, y_true, ordered_labels, temperature=1.0)
        best_temperature = min(
            valid_candidates,
            key=lambda value: (
                _nll(matrix, y_true, ordered_labels, value),
                abs(value - 1.0),
            ),
        )
        after = _nll(matrix, y_true, ordered_labels, best_temperature)

        return cls(
            labels=ordered_labels,
            temperature=best_temperature,
            fit_data_role=data_role,
            sample_count=len(y_true),
            nll_before=before,
            nll_after=after,
        )

    def transform(
        self,
        probabilities: Sequence[Mapping[str, float]],
    ) -> list[dict[str, float]]:
        matrix = _check_inputs(
            probabilities,
            [self.labels[0]] * len(probabilities),
            self.labels,
        )
        output = []
        for row in matrix:
            logits = [math.log(max(value, 1e-15)) / self.temperature for value in row]
            peak = max(logits)
            exp_values = [math.exp(value - peak) for value in logits]
            denominator = sum(exp_values)
            output.append(
                {
                    label: exp_values[index] / denominator
                    for index, label in enumerate(self.labels)
                }
            )
        return output

    def to_dict(self) -> dict:
        return asdict(self)
