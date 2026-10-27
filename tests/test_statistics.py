from evaluation.statistics import paired_comparison


def test_paired_comparison_is_deterministic_and_preserves_pair_count():
    truth = ["a", "b", "a", "b", "a", "b"]
    model_a = ["a", "b", "b", "a", "a", "a"]
    model_b = ["a", "a", "a", "b", "a", "b"]

    first = paired_comparison(
        truth,
        model_a,
        model_b,
        labels=["a", "b"],
        seed=44,
        bootstrap_replicates=100,
    )
    second = paired_comparison(
        truth,
        model_a,
        model_b,
        labels=["a", "b"],
        seed=44,
        bootstrap_replicates=100,
    )

    assert first == second
    assert first["n"] == 6
    assert abs(first["accuracy_difference"] - 1 / 3) < 1e-12
    assert first["mcnemar"]["discordant_records"] == 4
    assert len(first["accuracy_difference_ci_95"]) == 2
