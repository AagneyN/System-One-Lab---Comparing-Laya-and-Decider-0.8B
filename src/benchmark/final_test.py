"""One-shot held-out evaluation of the three frozen model configurations."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import logging
import os
import time
import uuid
import warnings
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmark.runner import (
    DECIDER_REVISION,
    LAYA_REVISION,
    MODEL_NAMES,
    TASK_TO_COLUMN,
    load_latest_development_run,
)
from evaluation.calibration import TemperatureCalibrator
from evaluation.config import freeze_config, sha256_file
from evaluation.dataset import read_dataset, validate_datasets
from evaluation.metrics import evaluate_classification
from evaluation.safeguards import assert_final_test_ready
from evaluation.statistics import compare_models_by_task
from evaluation.store import RunStore
from finetuning.runner import (
    CALIBRATION_ROLE,
    FINETUNED_DISPLAY_NAME,
    FINETUNED_MODEL_KEY,
    _apply_calibrator,
    _versions,
    load_latest_finetuning_run,
)
from models.adapters import Decider08BAdapter, LayaBaseAdapter, LayaFineTunedAdapter
from models.tasks import TASK_LABELS, TASK_QUESTIONS


FINAL_MODEL_ORDER = ("laya_base", "decider_08b", FINETUNED_MODEL_KEY)
FINAL_MODEL_NAMES = {
    "laya_base": MODEL_NAMES["laya_base"],
    "decider_08b": MODEL_NAMES["decider_08b"],
    FINETUNED_MODEL_KEY: FINETUNED_DISPLAY_NAME,
}


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


def load_latest_final_test_run(results_root: str | Path = "results") -> dict[str, Any] | None:
    run_root = Path(results_root) / "final_test"
    manifests = list(run_root.glob("*/run.json")) if run_root.is_dir() else []
    if not manifests:
        return None
    latest = max(manifests, key=lambda path: path.stat().st_mtime)
    record = json.loads(latest.read_text(encoding="utf-8"))
    return {"path": str(latest.parent), **record["payload"]}


def _package_versions() -> dict[str, str | None]:
    versions = _versions()
    try:
        versions["decider-ai"] = importlib.metadata.version("decider-ai")
    except importlib.metadata.PackageNotFoundError:
        versions["decider-ai"] = None
    return versions


def _calibrator_from_spec(spec: dict[str, Any] | None) -> TemperatureCalibrator | None:
    if not spec or not spec.get("available"):
        return None
    return TemperatureCalibrator(
        labels=tuple(spec["labels"]),
        temperature=float(spec["temperature"]),
        fit_data_role=str(spec["fit_data_role"]),
        sample_count=int(spec["sample_count"]),
        nll_before=float(spec["nll_before"]),
        nll_after=float(spec["nll_after"]),
    )


def _make_record(
    frame_row,
    model_key: str,
    prediction,
    calibrators: dict[str, Any],
) -> dict[str, Any]:
    truth = {
        task: str(frame_row[column])
        for task, column in TASK_TO_COLUMN.items()
    }
    calibrated = _apply_calibrator(prediction.probabilities, calibrators)
    return {
        "ticket_id": str(frame_row["ticket_id"]),
        "message": str(frame_row["message"]),
        "difficulty": str(frame_row["difficulty"]),
        "ground_truth": truth,
        "predictions": prediction.predictions,
        "probabilities": prediction.probabilities,
        "calibrated_probabilities": calibrated,
        "probability_notes": prediction.probability_notes,
        "inference_ms": prediction.inference_ms,
        "raw_output": prediction.raw_output,
        "metadata": prediction.metadata,
    }


def _task_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task, labels in TASK_LABELS.items():
        truth = [record["ground_truth"][task] for record in records]
        predicted = [record["predictions"][task] for record in records]
        raw_probs = [record.get("probabilities", {}).get(task) for record in records]
        calibrated_probs = [record.get("calibrated_probabilities", {}).get(task) for record in records]
        raw_result = evaluate_classification(
            truth,
            predicted,
            labels=labels,
            probabilities=(raw_probs if raw_probs and all(row is not None for row in raw_probs) else None),
            positive_label="true" if task == "refund" else None,
        )
        calibrated_result = evaluate_classification(
            truth,
            predicted,
            labels=labels,
            probabilities=(
                calibrated_probs
                if calibrated_probs and all(row is not None for row in calibrated_probs)
                else None
            ),
            positive_label="true" if task == "refund" else None,
        )
        output[task] = {
            **{key: value for key, value in raw_result.items() if key != "probability_metrics"},
            "probability_metrics": {
                "raw": raw_result["probability_metrics"],
                "calibrated": calibrated_result["probability_metrics"],
            },
        }
    return output


def _difficulty_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for difficulty in sorted({record["difficulty"] for record in records}):
        group = [record for record in records if record["difficulty"] == difficulty]
        tasks = {}
        for task, labels in TASK_LABELS.items():
            tasks[task] = evaluate_classification(
                [row["ground_truth"][task] for row in group],
                [row["predictions"][task] for row in group],
                labels=labels,
            )
        output[difficulty] = {
            "count": len(group),
            "small_subgroup": len(group) < 10,
            "tasks": tasks,
        }
    return output


def _markdown_report(record: dict[str, Any]) -> str:
    lines = [
        "# System One Lab — Final Held-Out Evaluation",
        "",
        f"- Run ID: `{record['run_id']}`",
        f"- Completed (UTC): `{record['completed_at_utc']}`",
        f"- Dataset: `{record['dataset']['rows']}` held-out test records",
        f"- Dataset SHA-256: `{record['dataset']['sha256']}`",
        f"- Frozen configuration SHA-256: `{record['configuration_sha256']}`",
        f"- Status: **{record['status']}**",
        "",
        "These metrics were computed after freezing the models, inference schema, calibration, and device/precision policy. The test set was not used for training, model selection, or calibration.",
        "",
        "## Per-task results",
        "",
        "| Model | Task | N | Accuracy | Macro precision | Macro recall | Macro F1 | Raw log loss | Calibrated log loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in FINAL_MODEL_ORDER:
        model = record.get("models", {}).get(key, {})
        for task, metric in model.get("metrics", {}).items():
            probability = metric.get("probability_metrics", {})
            raw = probability.get("raw", {}).get("log_loss")
            calibrated = probability.get("calibrated", {}).get("log_loss")
            def fmt(value):
                return f"{value:.4f}" if isinstance(value, (float, int)) else "n/a"
            lines.append(
                f"| {model.get('display_name', FINAL_MODEL_NAMES[key])} | {task} | {metric.get('count', 0)} | "
                f"{fmt(metric.get('accuracy'))} | {fmt(metric.get('macro_precision'))} | "
                f"{fmt(metric.get('macro_recall'))} | {fmt(metric.get('macro_f1'))} | "
                f"{fmt(raw)} | {fmt(calibrated)} |"
            )
    lines.extend(["", "## Paired statistical comparisons", ""])
    for pair, tasks in record.get("paired_comparisons", {}).items():
        lines.append(f"### {pair}")
        lines.append("")
        lines.append("| Task | N | Accuracy delta (B−A), 95% CI | Macro-F1 delta (B−A), 95% CI | McNemar p |")
        lines.append("|---|---:|---:|---:|---:|")
        for task, result in tasks.items():
            aci = result["accuracy_difference_ci_95"]
            fci = result["macro_f1_difference_ci_95"]
            lines.append(
                f"| {task} | {result['n']} | {result['accuracy_difference']:+.3f} "
                f"[{aci[0]:+.3f}, {aci[1]:+.3f}] | {result['macro_f1_difference']:+.3f} "
                f"[{fci[0]:+.3f}, {fci[1]:+.3f}] | "
                f"{result['mcnemar']['exact_two_sided_p_value']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- All models were evaluated on the same ticket IDs; paired intervals use record-level bootstrap resampling with a fixed seed.",
            "- McNemar tests compare per-record correctness. These tests are exploratory and no multiple-comparison correction is applied.",
            "- The 150-record test set gives uncertain estimates, especially for rare labels and difficulty groups. Small point-estimate differences do not establish superiority.",
            "- The fine-tuned checkpoint used head-only training (`freeze_encoder=True`) due to the 6 GB GPU constraint. This is not a full-encoder fine-tune.",
            "- The training dataset contains 245 tickets; subgroup results and generalization claims should remain cautious.",
            "",
        ]
    )
    return "\n".join(lines)


def run_final_test_evaluation(
    *,
    dev_path: str | Path,
    test_path: str | Path,
    fine_tuning_run: dict[str, Any] | None = None,
    results_root: str | Path = "results",
    reports_root: str | Path = "reports",
    expected_dev_rows: int = 350,
    expected_test_rows: int = 150,
    device: str | None = None,
    explicitly_confirmed: bool = False,
    repeat_reason: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    adapter_factories: dict[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    """Run all three frozen models once on the same held-out test set.

    The explicit confirmation is required by the project methodology. No test
    messages or labels are passed to a model or calibration routine.
    """
    if explicitly_confirmed is not True:
        raise ValueError("Final test evaluation requires explicit user confirmation.")
    dev_path = Path(dev_path).resolve()
    test_path = Path(test_path).resolve()
    reports = validate_datasets(dev_path, test_path)
    validation_passed = (
        reports.is_valid
        and reports.dev.row_count == expected_dev_rows
        and reports.test.row_count == expected_test_rows
    )
    if not validation_passed:
        raise ValueError("Dataset validation failed or dataset counts differ from the frozen protocol.")

    fine_tuning_run = fine_tuning_run or load_latest_finetuning_run(results_root)
    if fine_tuning_run is None or fine_tuning_run.get("status") != "complete":
        raise ValueError("A complete, verified fine-tuning run is required for final evaluation.")
    if not fine_tuning_run.get("checkpoint", {}).get("verified_loadable"):
        raise ValueError("The fine-tuned checkpoint has not been verified by loading and predicting.")
    checkpoint = Path(fine_tuning_run["checkpoint"]["path"]).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Fine-tuned checkpoint directory is missing: {checkpoint}")
    analysis_revision = fine_tuning_run.get("analysis_revision", {})
    calibrators = analysis_revision.get("calibrators", fine_tuning_run.get("calibrators", {}))
    if any(key not in calibrators for key in FINAL_MODEL_ORDER):
        raise ValueError("Development calibration is missing for one or more final model configurations.")

    import torch

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    packages_available = all(
        find_spec(module) is not None
        for module in ("laya", "decider", "huggingface_hub", "torch")
    )
    dev_hash = sha256_file(dev_path)
    test_hash = sha256_file(test_path)
    if dev_hash != fine_tuning_run.get("dataset", {}).get("sha256"):
        raise ValueError("Fine-tuned checkpoint was trained from a different development dataset hash.")
    baseline = load_latest_development_run(results_root)
    if baseline is None or baseline.get("run_id") != fine_tuning_run.get("baseline_run_id"):
        raise ValueError("The baseline referenced by the fine-tuning run is no longer the latest saved baseline.")
    if baseline.get("dataset", {}).get("sha256") != dev_hash:
        raise ValueError("Baseline development predictions use a different development dataset hash.")

    project_root = dev_path.parent.parent
    hf_home = project_root / "models" / "hf-cache"
    os.environ["HF_HOME"] = str(hf_home)
    # Use the exact revisions already downloaded for the development baselines.
    # Direct local paths avoid a Hub metadata lookup during the one-shot test run.
    decider_checkpoint = (
        hf_home
        / "hub"
        / "models--Mapika--decider-0.8b"
        / "snapshots"
        / DECIDER_REVISION
    )
    if not (decider_checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pinned Decider checkpoint is missing locally: {decider_checkpoint}")
    if not (decider_checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Pinned Decider config is missing locally: {decider_checkpoint}")
    if adapter_factories is None:
        adapter_factories = {
            "laya_base": lambda: LayaBaseAdapter.load(
                device=chosen_device,
                revision=LAYA_REVISION,
            ),
            "decider_08b": lambda: Decider08BAdapter.load(
                checkpoint=decider_checkpoint,
                device=chosen_device,
                use_graphs=False,
            ),
            FINETUNED_MODEL_KEY: lambda: LayaFineTunedAdapter.load(
                checkpoint=checkpoint,
                device=chosen_device,
            ),
        }

    run_started = datetime.now(timezone.utc)
    run_id = f"test-{run_started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    config = {
        "phase": "final_test",
        "dataset_role": "test",
        "dataset_hashes": {"dev": dev_hash, "test": test_hash},
        "dataset_rows": {"dev": expected_dev_rows, "test": expected_test_rows},
        "model_order": list(FINAL_MODEL_ORDER),
        "models": {
            "laya_base": {
                "model_id": "convaiinnovations/laya",
                "checkpoint_revision": LAYA_REVISION,
            },
            "decider_08b": {
                "model_id": "Mapika/decider-0.8b",
                "checkpoint_revision": DECIDER_REVISION,
                "execution_mode": "eager",
            },
            FINETUNED_MODEL_KEY: {
                "model_id": "system-one-lab/laya-finetuned",
                "base_checkpoint_revision": LAYA_REVISION,
                "checkpoint_path": str(checkpoint),
                "checkpoint_run_id": fine_tuning_run["run_id"],
                "checkpoint_sha256": fine_tuning_run["checkpoint"]["file_manifest"],
                "training_method": fine_tuning_run["training_method"],
                "training_settings": fine_tuning_run["training_settings"],
            },
        },
        "task_labels": {task: list(labels) for task, labels in TASK_LABELS.items()},
        "task_schema": TASK_QUESTIONS,
        "inference_protocol": {
            "input_field": "message only",
            "difficulty_used_as_input": False,
            "all_models_receive_same_task_schema": True,
            "all_models_evaluated_on_same_test_ticket_ids": True,
            "models_loaded_sequentially": True,
            "decider_execution_mode": "eager",
            "probability_policy": "retain validated model-returned distributions; calibrated probabilities are separate",
        },
        "calibration": {
            "method": "per-model, per-task scalar temperature scaling",
            "fit_data_role": CALIBRATION_ROLE,
            "fit_ticket_ids": fine_tuning_run["split"]["ticket_ids"]["calibration"],
            "fit_ticket_count": len(fine_tuning_run["split"]["ticket_ids"]["calibration"]),
            "calibrators": calibrators,
            "config_source": analysis_revision.get(
                "calibrators_path",
                fine_tuning_run["checkpoint"]["calibrators_path"],
            ),
        },
        "statistical_comparison": {
            "paired_bootstrap_seed": 20261027,
            "bootstrap_replicates": 2000,
            "mcnemar": "exact two-sided binomial test on discordant correctness counts",
            "multiple_comparison_correction": None,
        },
        "hardware_policy": {
            "selected_device": chosen_device,
            "gpu_name": torch.cuda.get_device_name(0) if chosen_device.startswith("cuda") and torch.cuda.is_available() else None,
            "precision": "record actual model-reported dtype; no silent precision change",
        },
        "package_versions": _package_versions(),
        "random_seed": 20261027,
        "user_confirmation": "explicitly requested final fine-tuned model analysis",
        "test_set_consumed_for_development": False,
    }
    frozen = freeze_config(config)
    assert_final_test_ready(
        validation_passed=validation_passed,
        configuration_frozen=bool(frozen.sha256),
        models_available=packages_available,
        explicitly_confirmed=explicitly_confirmed,
    )

    # Read test records for scoring only after the frozen configuration is established.
    test_frame = read_dataset(test_path)
    if len(test_frame) != expected_test_rows:
        raise ValueError(f"Expected {expected_test_rows} test records, found {len(test_frame)}.")
    root = Path(results_root)
    report_path = Path(reports_root) / "final_test" / f"{run_id}.md"

    # The guard and frozen hash are established before the test labels are used for scoring.
    model_results: dict[str, Any] = {}
    all_errors: list[dict[str, str]] = []
    ground_truth_records: list[dict[str, Any]] = []
    for row in test_frame.to_dict(orient="records"):
        ground_truth_records.append(
            {
                "ticket_id": str(row["ticket_id"]),
                "ground_truth": {
                    task: str(row[column]) for task, column in TASK_TO_COLUMN.items()
                },
            }
        )

    for model_index, model_key in enumerate(FINAL_MODEL_ORDER):
        if progress_callback:
            progress_callback({
                "phase": "loading",
                "model": model_key,
                "model_name": FINAL_MODEL_NAMES[model_key],
                "model_index": model_index,
                "total_models": len(FINAL_MODEL_ORDER),
            })
        collector = _WarningCollector()
        root_logger = logging.getLogger()
        root_logger.addHandler(collector)
        captured: list[warnings.WarningMessage] = []
        adapter = None
        model_records: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        times: list[float] = []
        load_seconds = None
        runtime: dict[str, Any] = {}
        wall_started = time.perf_counter()
        if chosen_device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                adapter = adapter_factories[model_key]()
                load_seconds = getattr(adapter, "load_seconds", None)
                cal_specs = calibrators[model_key]
                for index, row in enumerate(test_frame.to_dict(orient="records"), start=1):
                    try:
                        prediction = adapter.predict(str(row["message"]))
                        record = _make_record(row, model_key, prediction, cal_specs)
                        model_records.append(record)
                        times.append(float(prediction.inference_ms))
                        runtime.update(prediction.metadata)
                    except Exception as exc:
                        error = {
                            "ticket_id": str(row["ticket_id"]),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        errors.append(error)
                        all_errors.append({"model": model_key, **error})
                        model_records.append(error)
                    if progress_callback and (index % 10 == 0 or index == len(test_frame)):
                        progress_callback({
                            "phase": "predicting",
                            "model": model_key,
                            "model_name": FINAL_MODEL_NAMES[model_key],
                            "completed": index,
                            "total": len(test_frame),
                            "model_index": model_index,
                            "total_models": len(FINAL_MODEL_ORDER),
                        })
        except Exception as exc:
            error = {"stage": "load_or_run", "error": f"{type(exc).__name__}: {exc}"}
            errors.append(error)
            all_errors.append({"model": model_key, **error})
        finally:
            root_logger.removeHandler(collector)
            if adapter is not None:
                backend = getattr(adapter, "backend", None)
                unload = getattr(backend, "unload", None)
                if callable(unload):
                    try:
                        unload("english")
                    except Exception:
                        pass
                adapter.backend = None
                del adapter
            gc.collect()
            if chosen_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        successful = [row for row in model_records if "predictions" in row]
        if chosen_device.startswith("cuda") and torch.cuda.is_available():
            peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        else:
            peak_gpu_bytes = None
        wall_seconds = time.perf_counter() - wall_started
        metrics = _task_metrics(successful) if successful else {}
        difficulty = _difficulty_metrics(successful) if successful else {}
        model_results[model_key] = {
            "display_name": FINAL_MODEL_NAMES[model_key],
            "status": "complete" if len(successful) == len(test_frame) and not errors else "partial",
            "expected_records": len(test_frame),
            "successful_records": len(successful),
            "failed_records": len(test_frame) - len(successful),
            "load_seconds": load_seconds,
            "cold_start_seconds": (
                load_seconds + times[0] / 1000.0 if load_seconds is not None and times else None
            ),
            "latency": {
                "count": len(times),
                "median_ms": float(np.median(times)) if times else None,
                "p95_ms": float(np.percentile(times, 95)) if times else None,
                "mean_ms": float(np.mean(times)) if times else None,
                "throughput_records_per_second": len(times) / (sum(times) / 1000.0) if times and sum(times) else None,
                "wall_seconds": wall_seconds,
            },
            "resource_usage": {
                "peak_gpu_memory_bytes": peak_gpu_bytes,
                "peak_gpu_memory_mb": peak_gpu_bytes / (1024 * 1024) if peak_gpu_bytes is not None else None,
                "cpu_memory_bytes": None,
                "cpu_memory_unavailable_reason": "psutil is not installed; CPU memory was not measured.",
            },
            "runtime": runtime,
            "metrics": metrics,
            "difficulty": difficulty,
            "calibrators": calibrators[model_key],
            "predictions": model_records,
            "warnings": [str(item.message) for item in captured] + collector.messages,
            "errors": errors,
        }
        if progress_callback:
            progress_callback({
                "phase": model_results[model_key]["status"],
                "model": model_key,
                "model_name": FINAL_MODEL_NAMES[model_key],
                "completed": len(test_frame),
                "total": len(test_frame),
                "model_index": model_index,
                "total_models": len(FINAL_MODEL_ORDER),
            })

    complete = all(model_results[key]["status"] == "complete" for key in FINAL_MODEL_ORDER)
    prediction_map = {}
    for model_key, model in model_results.items():
        prediction_map[model_key] = {
            row["ticket_id"]: row["predictions"]
            for row in model["predictions"]
            if "predictions" in row
        }
    paired = (
        compare_models_by_task(
            ground_truth_records,
            prediction_map,
            TASK_LABELS,
            seed=20261027,
            bootstrap_replicates=2000,
        )
        if complete
        else {}
    )
    ended_at = datetime.now(timezone.utc)
    record = {
        "run_id": run_id,
        "status": "complete" if complete else "partial",
        "started_at_utc": run_started.isoformat(),
        "completed_at_utc": ended_at.isoformat(),
        "dataset": {
            "role": "test",
            "path": str(test_path),
            "rows": len(test_frame),
            "sha256": test_hash,
            "validation": reports.test.to_dict(),
            "development_test_overlap": reports.overlap,
        },
        "configuration": frozen.to_dict(),
        "configuration_sha256": frozen.sha256,
        "configuration_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_fingerprint": frozen.sha256,
        "models": model_results,
        "paired_comparisons": paired,
        "errors": all_errors,
        "warnings": [
            "The 150-ticket test set yields uncertain subgroup and rare-class estimates; interpret paired intervals with that limit in mind.",
            "McNemar p-values are exploratory; no multiple-comparison correction was applied.",
            "Any future model, prompt, threshold, or calibration change based on these test results consumes this test set for development and requires a new untouched test set for a new final claim.",
        ],
        "test_set_consumed_for_development": False,
        "limitations": [
            "Laya was fine-tuned with the official RLCD implementation and supported freeze_encoder mode; only the decision head was updated.",
            "The fine-tuning split contains 245 train, 35 calibration and 70 validation tickets from a 350-ticket development dataset.",
            "The original Laya base and Decider 0.8B have different pretrained histories; their training histories are not treated as identical.",
        ],
        "report_path": str(report_path.resolve()),
    }
    saved_path = RunStore(root).save_run(
        "final_test",
        run_id,
        record,
        test_evaluation_confirmed=explicitly_confirmed,
        repeat_reason=repeat_reason,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(record), encoding="utf-8")
    return {"path": str(saved_path), "report_path": str(report_path), "record": record}
