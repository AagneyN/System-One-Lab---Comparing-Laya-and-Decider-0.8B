from evaluation.reanalysis import metrics_from_saved_predictions, recover_probability_distribution


def test_raw_model_probabilities_can_be_recovered_from_saved_output():
    record = {
        "ground_truth": {"department": "billing"},
        "predictions": {"department": "billing"},
        "probabilities": {"department": None},
        "raw_output": {
            "answers": {
                "department": {
                    "probabilities": {
                        "billing": 0.80004,
                        "technical": 0.050015,
                        "sales": 0.050015,
                        "account": 0.050015,
                        "other": 0.050015,
                    }
                }
            }
        },
    }
    labels = ["billing", "technical", "sales", "account", "other"]

    recovered = recover_probability_distribution(record, "department", labels)
    metrics = metrics_from_saved_predictions(
        [record],
        {"department": labels},
    )

    assert recovered is not None
    assert recovered["billing"] == 0.80004
    assert metrics["department"]["probability_metrics"]["available"] is True


def test_invalid_saved_probability_vector_remains_unavailable():
    record = {
        "probabilities": {"refund": None},
        "raw_output": {
            "answers": {"refund": {"probabilities": {"true": 0.2, "false": 0.2}}}
        },
    }
    assert recover_probability_distribution(record, "refund", ["true", "false"]) is None
