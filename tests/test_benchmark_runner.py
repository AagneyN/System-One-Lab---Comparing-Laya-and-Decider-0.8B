import csv

from benchmark.runner import run_development_benchmark
from models.adapters import Prediction
from models.tasks import TASK_LABELS


class FakeAdapter:
    def __init__(self, model_id, *, fail_message=None):
        self.model_id = model_id
        self.load_seconds = 0.01
        self.backend = object()
        self.fail_message = fail_message
        self.seen_messages = []

    def predict(self, message):
        self.seen_messages.append(message)
        if message == self.fail_message:
            raise RuntimeError("synthetic adapter failure")

        predictions = {task: labels[0] for task, labels in TASK_LABELS.items()}
        probabilities = {
            task: {label: 1.0 / len(labels) for label in labels}
            for task, labels in TASK_LABELS.items()
        }
        return Prediction(
            model_id=self.model_id,
            predictions=predictions,
            probabilities=probabilities,
            probability_notes={},
            inference_ms=2.0,
            load_seconds=self.load_seconds,
            raw_output={"answers": {}},
            metadata={"device": "cpu", "precision": "torch.float32"},
        )


def write_dev_csv(path, messages=("first message", "second message")):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ticket_id": f"dev-{index + 1}",
            "message": message,
            "department_label": "billing" if index == 0 else "technical",
            "urgency_label": "low" if index == 0 else "high",
            "refund_label": "false" if index == 0 else "true",
            "difficulty": "straightforward" if index == 0 else "ambiguous",
        }
        for index, message in enumerate(messages)
    ]
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_development_run_saves_both_models_and_only_dev_hashes(tmp_path):
    dev_path = tmp_path / "data" / "tickets_dev.csv"
    write_dev_csv(dev_path)
    laya = FakeAdapter("laya")
    decider = FakeAdapter("decider")
    progress = []

    result = run_development_benchmark(
        dev_path=dev_path,
        results_root=tmp_path / "results",
        expected_rows=2,
        device="cpu",
        adapter_factories={
            "laya_base": lambda: laya,
            "decider_08b": lambda: decider,
        },
        progress_callback=progress.append,
    )

    record = result["record"]
    assert record["status"] == "complete"
    assert record["dataset"]["rows"] == 2
    assert set(record["configuration"]["dataset_hashes"]) == {"dev"}
    assert record["models"]["laya_base"]["successful_records"] == 2
    assert record["models"]["decider_08b"]["successful_records"] == 2
    assert record["models"]["laya_base"]["metrics"]["department"]["count"] == 2
    assert laya.seen_messages == ["first message", "second message"]
    assert decider.seen_messages == ["first message", "second message"]
    assert progress[-1]["phase"] == "completed"
    assert (tmp_path / "results" / "development" / record["run_id"] / "run.json").is_file()


def test_failed_prediction_is_saved_as_partial_not_complete(tmp_path):
    dev_path = tmp_path / "data" / "tickets_dev.csv"
    write_dev_csv(dev_path)
    laya = FakeAdapter("laya", fail_message="second message")
    decider = FakeAdapter("decider")

    result = run_development_benchmark(
        dev_path=dev_path,
        results_root=tmp_path / "results",
        expected_rows=2,
        device="cpu",
        adapter_factories={
            "laya_base": lambda: laya,
            "decider_08b": lambda: decider,
        },
    )

    record = result["record"]
    assert record["status"] == "partial"
    assert record["models"]["laya_base"]["status"] == "partial"
    assert record["models"]["laya_base"]["failed_records"] == 1
    assert record["models"]["decider_08b"]["status"] == "complete"
