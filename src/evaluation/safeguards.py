"""Guards against using the held-out test split during development."""


DEVELOPMENT_ROLES = {
    "dev",
    "development_train",
    "development_validation",
}


def assert_development_only(dataset_role: str, operation: str) -> None:
    if dataset_role not in DEVELOPMENT_ROLES:
        raise ValueError(
            f"{operation} must use development data; got {dataset_role!r}."
        )


def assert_final_test_ready(
    *,
    validation_passed: bool,
    configuration_frozen: bool,
    models_available: bool,
    explicitly_confirmed: bool,
) -> None:
    missing = []
    if not validation_passed:
        missing.append("dataset validation has not passed")
    if not configuration_frozen:
        missing.append("configuration is not frozen")
    if not models_available:
        missing.append("required checkpoints are unavailable")
    if not explicitly_confirmed:
        missing.append("final test evaluation was not explicitly confirmed")

    if missing:
        raise ValueError("Final test evaluation blocked: " + "; ".join(missing))