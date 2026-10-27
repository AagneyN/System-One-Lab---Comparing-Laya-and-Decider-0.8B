"""Paired model comparisons for predictions on shared records."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def paired_comparison(
    y_true: Sequence[str],
    predictions_a: Sequence[str],
    predictions_b: Sequence[str],
    *,
    labels: Sequence[str],
    seed: int = 20261027,
    bootstrap_replicates: int = 2000,
) -> dict:
    """Return B-minus-A paired bootstrap intervals and an exact McNemar test.

    Bootstrap sampling resamples record indices jointly, preserving the paired
    structure. Macro-F1 uses the complete declared label set in every replicate.
    """
    if not (len(y_true) == len(predictions_a) == len(predictions_b)):
        raise ValueError("Paired comparisons require equal-length record vectors.")
    if not y_true:
        raise ValueError("At least one paired record is required.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")

    truth = np.asarray(y_true, dtype=object)
    pred_a = np.asarray(predictions_a, dtype=object)
    pred_b = np.asarray(predictions_b, dtype=object)
    correct_a = pred_a == truth
    correct_b = pred_b == truth

    def macro_f1(indices: np.ndarray, predicted: np.ndarray) -> float:
        return float(
            f1_score(
                truth[indices],
                predicted[indices],
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        )

    full_indices = np.arange(len(truth))
    observed_accuracy_delta = float(correct_b.mean() - correct_a.mean())
    observed_f1_delta = macro_f1(full_indices, pred_b) - macro_f1(full_indices, pred_a)

    rng = np.random.default_rng(seed)
    accuracy_deltas = np.empty(bootstrap_replicates, dtype=float)
    f1_deltas = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, len(truth), size=len(truth))
        accuracy_deltas[replicate] = float(
            correct_b[indices].mean() - correct_a[indices].mean()
        )
        f1_deltas[replicate] = macro_f1(indices, pred_b) - macro_f1(indices, pred_a)

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = (
        float(binomtest(min(a_only, b_only), n=discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )

    return {
        "n": len(truth),
        "direction": "model_b_minus_model_a",
        "model_a_correct": int(correct_a.sum()),
        "model_b_correct": int(correct_b.sum()),
        "accuracy_difference": observed_accuracy_delta,
        "accuracy_difference_ci_95": [
            float(np.quantile(accuracy_deltas, 0.025)),
            float(np.quantile(accuracy_deltas, 0.975)),
        ],
        "macro_f1_difference": float(observed_f1_delta),
        "macro_f1_difference_ci_95": [
            float(np.quantile(f1_deltas, 0.025)),
            float(np.quantile(f1_deltas, 0.975)),
        ],
        "mcnemar": {
            "model_a_only_correct": a_only,
            "model_b_only_correct": b_only,
            "discordant_records": discordant,
            "exact_two_sided_p_value": p_value,
        },
        "bootstrap": {
            "method": "paired record bootstrap with replacement",
            "replicates": bootstrap_replicates,
            "seed": seed,
            "interval": "percentile 95%",
        },
        "caveat": (
            "Intervals and tests are uncertain at this sample size; interpret rare classes "
            "and small difficulty groups cautiously."
        ),
    }


def compare_models_by_task(
    records: Sequence[dict],
    model_predictions: dict[str, dict[str, dict]],
    task_labels: dict[str, Sequence[str]],
    *,
    seed: int = 20261027,
    bootstrap_replicates: int = 2000,
) -> dict:
    """Compare every model pair and task using ticket IDs to align records."""
    import itertools

    ticket_ids = [str(row["ticket_id"]) for row in records]
    output: dict[str, dict] = {}
    for model_a, model_b in itertools.combinations(model_predictions, 2):
        output[f"{model_a}_vs_{model_b}"] = {}
        for task, labels in task_labels.items():
            truth = [
                str(row.get("ground_truth", row)[task])
                for row in records
            ]
            pred_a = [model_predictions[model_a][ticket][task] for ticket in ticket_ids]
            pred_b = [model_predictions[model_b][ticket][task] for ticket in ticket_ids]
            output[f"{model_a}_vs_{model_b}"][task] = paired_comparison(
                truth,
                pred_a,
                pred_b,
                labels=labels,
                seed=seed,
                bootstrap_replicates=bootstrap_replicates,
            )
    return output
