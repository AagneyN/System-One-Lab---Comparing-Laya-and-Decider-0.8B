"""Shared classification metrics. No model inference happens in this module."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


def _validated_probabilities(
    probabilities: Sequence[Mapping[str, float]],
    labels: Sequence[str],
) -> tuple[list[list[float]] | None, str | None]:
    matrix: list[list[float]] = []

    if not probabilities:
        return None, "No probability rows were supplied."
    if set(probabilities[0].keys()) != set(labels):
        return None, "Probability keys must match the supplied label set."

    for row in probabilities:
        if set(row.keys()) != set(labels):
            return None, "Probability keys differ between records."
        values = [float(row[label]) for label in labels]
        if any(not math.isfinite(value) or value < 0 for value in values):
            return None, "Probabilities must be finite and nonnegative."
        total = sum(values)
        if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            return None, "Each probability row must sum to 1."
        # Some model runtimes return float32 distributions whose totals differ
        # from one by a few ulps. Preserve the original values in prediction
        # records, but normalize within tolerance for proper scoring rules.
        matrix.append([value / total for value in values])

    return matrix, None


def _reliability_bins(
    y_true: Sequence[str],
    probabilities: list[list[float]],
    labels: Sequence[str],
    bin_count: int,
) -> list[dict[str, float | int]]:
    bins = []
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        confidences: list[float] = []
        correct: list[float] = []

        for truth, row in zip(y_true, probabilities):
            best_index = max(range(len(row)), key=row.__getitem__)
            confidence = row[best_index]
            in_bin = lower <= confidence < upper or (
                bin_index == bin_count - 1 and confidence == 1.0
            )
            if in_bin:
                confidences.append(confidence)
                correct.append(float(labels[best_index] == truth))

        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(confidences),
                "mean_confidence": (
                    sum(confidences) / len(confidences) if confidences else 0.0
                ),
                "accuracy": sum(correct) / len(correct) if correct else 0.0,
            }
        )
    return bins


def evaluate_classification(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    probabilities: Sequence[Mapping[str, float]] | None = None,
    positive_label: str | None = None,
    ece_bins: int = 10,
) -> dict:
    """Calculate core metrics and probability-quality metrics when available.

    ECE uses equal-width confidence bins over top-label confidence.
    For binary Brier score, pass positive_label; otherwise multiclass Brier
    score is the mean per-record sum of squared class-probability errors.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")
    if not y_true:
        raise ValueError("At least one record is required.")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be a nonempty sequence of unique labels.")
    if any(value not in labels for value in y_true):
        raise ValueError("y_true contains a label outside the declared label set.")
    if any(value not in labels for value in y_pred):
        raise ValueError("y_pred contains a label outside the declared label set.")
    if ece_bins < 1:
        raise ValueError("ece_bins must be at least 1.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))

    result = {
        "count": len(y_true),
        "correct": int(sum(a == b for a, b in zip(y_true, y_pred))),
        "incorrect": int(sum(a != b for a, b in zip(y_true, y_pred))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(sum(precision) / len(labels)),
        "macro_recall": float(sum(recall) / len(labels)),
        "macro_f1": float(sum(f1) / len(labels)),
        "per_class": per_class,
        "confusion_matrix": {
            actual: {
                predicted: int(matrix[row, column])
                for column, predicted in enumerate(labels)
            }
            for row, actual in enumerate(labels)
        },
        "probability_metrics": {
            "available": False,
            "reason": "Probabilities were not supplied.",
        },
    }

    if probabilities is None:
        return result
    if len(probabilities) != len(y_true):
        result["probability_metrics"]["reason"] = (
            "Probability row count does not match the label row count."
        )
        return result

    probability_matrix, error = _validated_probabilities(probabilities, labels)
    if error:
        result["probability_metrics"]["reason"] = error
        return result

    assert probability_matrix is not None
    try:
        # sklearn expects probability columns in lexical label order. Preserve
        # the project's declared label order everywhere else, but reorder only
        # this call so the targets and probability columns remain aligned.
        lexical_labels = sorted(labels)
        lexical_columns = [list(labels).index(label) for label in lexical_labels]
        lexical_probabilities = [
            [row[index] for index in lexical_columns]
            for row in probability_matrix
        ]
        calculated_log_loss = float(
            log_loss(y_true, lexical_probabilities, labels=lexical_labels)
        )
    except ValueError as exc:
        result["probability_metrics"]["reason"] = str(exc)
        return result

    label_index = {label: index for index, label in enumerate(labels)}
    if len(labels) == 2 and positive_label is not None:
        if positive_label not in labels:
            raise ValueError("positive_label must be one of labels.")
        pos_index = label_index[positive_label]
        brier = sum(
            (row[pos_index] - float(truth == positive_label)) ** 2
            for truth, row in zip(y_true, probability_matrix)
        ) / len(y_true)
    else:
        brier = sum(
            sum(
                (row[index] - float(truth == label)) ** 2
                for index, label in enumerate(labels)
            )
            for truth, row in zip(y_true, probability_matrix)
        ) / len(y_true)

    bins = _reliability_bins(y_true, probability_matrix, labels, ece_bins)
    ece = sum(
        item["count"]
        / len(y_true)
        * abs(item["accuracy"] - item["mean_confidence"])
        for item in bins
    )

    result["probability_metrics"] = {
        "available": True,
        "normalization": (
            "Rows already close to 1 within 1e-3 were normalized for scoring; "
            "saved model-returned probabilities remain unchanged."
        ),
        "log_loss": calculated_log_loss,
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
        "ece_method": f"Equal-width bins; {ece_bins} bins; top-label confidence.",
        "reliability_bins": bins,
    }
    return result
