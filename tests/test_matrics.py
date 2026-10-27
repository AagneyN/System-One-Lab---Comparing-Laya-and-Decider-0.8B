import math

from evaluation.metrics import evaluate_classification


def test_core_classification_metrics():
    result = evaluate_classification(
        ["cat", "dog", "dog"],
        ["cat", "cat", "dog"],
        labels=["cat", "dog"],
    )

    assert result["count"] == 3
    assert result["correct"] == 2
    assert result["incorrect"] == 1
    assert math.isclose(result["accuracy"], 2 / 3)
    assert result["confusion_matrix"]["dog"]["cat"] == 1
    assert result["probability_metrics"]["available"] is False


def test_probability_metrics_when_distributions_are_valid():
    result = evaluate_classification(
        ["false", "true"],
        ["false", "true"],
        labels=["true", "false"],
        positive_label="true",
        probabilities=[
            {"true": 0.1, "false": 0.9},
            {"true": 0.9, "false": 0.1},
        ],
    )

    probability_metrics = result["probability_metrics"]
    assert probability_metrics["available"] is True
    assert probability_metrics["log_loss"] > 0
    assert probability_metrics["brier_score"] >= 0
    assert len(probability_metrics["reliability_bins"]) == 10


def test_invalid_probability_rows_are_reported_not_fabricated():
    result = evaluate_classification(
        ["yes"],
        ["yes"],
        labels=["yes", "no"],
        probabilities=[{"yes": 0.7, "no": 0.7}],
    )

    assert result["probability_metrics"]["available"] is False
    assert "sum to 1" in result["probability_metrics"]["reason"]