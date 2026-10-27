import pytest

from evaluation.calibration import TemperatureCalibrator


def test_calibrator_fits_and_preserves_probability_sums():
    probabilities = [
        {"yes": 0.95, "no": 0.05},
        {"yes": 0.80, "no": 0.20},
        {"yes": 0.10, "no": 0.90},
        {"yes": 0.25, "no": 0.75},
    ]
    truth = ["yes", "yes", "no", "no"]

    calibrator = TemperatureCalibrator.fit(
        probabilities,
        truth,
        labels=["yes", "no"],
        data_role="development_validation",
    )
    transformed = calibrator.transform(probabilities)

    assert calibrator.sample_count == 4
    assert calibrator.fit_data_role == "development_validation"
    assert all(abs(sum(row.values()) - 1.0) < 1e-8 for row in transformed)


def test_calibration_rejects_test_data():
    with pytest.raises(ValueError, match="test data is forbidden"):
        TemperatureCalibrator.fit(
            [{"yes": 0.8, "no": 0.2}],
            ["yes"],
            labels=["yes", "no"],
            data_role="test",
        )