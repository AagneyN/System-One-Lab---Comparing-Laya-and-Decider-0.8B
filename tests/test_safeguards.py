import pytest

from evaluation.safeguards import (
    assert_development_only,
    assert_final_test_ready,
)


def test_development_operations_reject_test_role():
    with pytest.raises(ValueError, match="must use development data"):
        assert_development_only("test", "calibration fitting")


def test_final_test_requires_all_preconditions():
    with pytest.raises(ValueError, match="blocked"):
        assert_final_test_ready(
            validation_passed=True,
            configuration_frozen=True,
            models_available=True,
            explicitly_confirmed=False,
        )