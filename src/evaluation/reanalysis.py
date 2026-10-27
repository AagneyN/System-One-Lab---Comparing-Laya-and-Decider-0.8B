"""Recompute metrics from persisted predictions without rerunning models."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from evaluation.metrics import evaluate_classification


def recover_probability_distribution(
    record: Mapping[str, Any],
    task: str,
    labels: Sequence[str],
    *,
    sum_tolerance: float = 1e-3,
) -> dict[str, float] | None:
    """Recover a model-returned distribution preserved in `raw_output` if needed."""
    probabilities = record.get("probabilities")
    candidate = probabilities.get(task) if isinstance(probabilities, Mapping) else None
    if candidate is None:
        raw = record.get("raw_output")
        answers = raw.get("answers") if isinstance(raw, Mapping) else None
        answer = answers.get(task) if isinstance(answers, Mapping) else None
        candidate = answer.get("probabilities") if isinstance(answer, Mapping) else None
    if not isinstance(candidate, Mapping) or set(candidate) != set(labels):
        return None
    try:
        output = {label: float(candidate[label]) for label in labels}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in output.values()):
        return None
    if not math.isclose(sum(output.values()), 1.0, rel_tol=sum_tolerance, abs_tol=sum_tolerance):
        return None
    return output


def metrics_from_saved_predictions(
    records: Sequence[Mapping[str, Any]],
    task_labels: Mapping[str, Sequence[str]],
    *,
    positive_labels: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    """Recompute core and probability metrics from existing predictions/raw outputs."""
    positive_labels = positive_labels or {}
    metrics = {}
    for task, labels in task_labels.items():
        truth = [str(record["ground_truth"][task]) for record in records]
        predictions = [str(record["predictions"][task]) for record in records]
        probabilities = [
            recover_probability_distribution(record, task, labels)
            for record in records
        ]
        metrics[task] = evaluate_classification(
            truth,
            predictions,
            labels=labels,
            probabilities=(probabilities if probabilities and all(probabilities) else None),
            positive_label=positive_labels.get(task),
        )
    return metrics
