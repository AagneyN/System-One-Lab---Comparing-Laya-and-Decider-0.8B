import json
from pathlib import Path

import pytest

from evaluation.config import freeze_config
from evaluation.store import RunStore


def test_config_hash_is_stable():
    config = {
        "phase": "development",
        "dataset_hashes": {"dev": "a" * 64},
        "seed": 42,
    }

    assert freeze_config(config).sha256 == freeze_config(config).sha256


def test_fine_tuning_config_rejects_test_training_data():
    with pytest.raises(ValueError, match="development dataset only"):
        freeze_config(
            {
                "phase": "fine_tuning",
                "dataset_hashes": {"dev": "a" * 64},
                "training_dataset_role": "test",
            }
        )


def test_development_config_rejects_test_hash():
    with pytest.raises(ValueError, match="cannot include the held-out test"):
        freeze_config(
            {
                "phase": "development",
                "dataset_hashes": {"dev": "a" * 64, "test": "b" * 64},
            }
        )


def test_run_store_refuses_to_overwrite(tmp_path):
    store = RunStore(tmp_path)
    saved = store.save_run("development", "dev-001", {"status": "complete"})

    assert json.loads((saved / "run.json").read_text())["run_id"] == "dev-001"
    with pytest.raises(FileExistsError):
        store.save_run("development", "dev-001", {"status": "different"})


def test_run_store_serializes_windows_paths_as_strings(tmp_path):
    store = RunStore(tmp_path)
    saved = store.save_run(
        "development",
        "dev-path-001",
        {"checkpoint": Path("models") / "laya-finetuned"},
    )

    payload = json.loads((saved / "run.json").read_text(encoding="utf-8"))["payload"]
    assert payload["checkpoint"] == str(Path("models") / "laya-finetuned")


def test_final_test_requires_confirmation_and_repeat_reason(tmp_path):
    store = RunStore(tmp_path)
    payload = {"experiment_fingerprint": "frozen-config-sha"}

    with pytest.raises(ValueError, match="explicit confirmation"):
        store.save_run("final_test", "test-001", payload)

    first = store.save_run(
        "final_test",
        "test-001",
        payload,
        test_evaluation_confirmed=True,
    )

    with pytest.raises(ValueError, match="repeat_reason"):
        store.save_run(
            "final_test",
            "test-002",
            payload,
            test_evaluation_confirmed=True,
        )

    second = store.save_run(
        "final_test",
        "test-002",
        payload,
        test_evaluation_confirmed=True,
        repeat_reason="Recovering after a documented technical interruption.",
    )

    assert (first / "run.json").exists()
    assert (second / "run.json").exists()
