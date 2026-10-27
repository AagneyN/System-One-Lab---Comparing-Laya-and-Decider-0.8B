import math
import warnings

from evaluation.metrics import evaluate_classification


def test_log_loss_respects_probability_columns_for_declared_label_order():
    labels = ("billing", "technical", "sales", "account", "other")
    probabilities = [
        {
            "billing": 0.8,
            "technical": 0.05,
            "sales": 0.05,
            "account": 0.05,
            "other": 0.05,
        }
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluate_classification(
            ["billing"],
            ["billing"],
            labels=labels,
            probabilities=probabilities,
        )

    assert math.isclose(
        result["probability_metrics"]["log_loss"], -math.log(0.8), rel_tol=1e-6
    )
    assert not any("lexicographically" in str(item.message) for item in caught)


def test_tiny_probability_sum_rounding_is_normalized_for_scoring():
    result = evaluate_classification(
        ["yes", "no"],
        ["yes", "no"],
        labels=["yes", "no"],
        probabilities=[
            {"yes": 0.80004, "no": 0.20004},
            {"yes": 0.10001, "no": 0.90001},
        ],
        positive_label="yes",
    )

    assert result["probability_metrics"]["available"] is True
    assert "normalized for scoring" in result["probability_metrics"]["normalization"]
