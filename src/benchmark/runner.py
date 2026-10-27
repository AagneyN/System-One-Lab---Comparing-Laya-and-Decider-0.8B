"""Development-only baseline benchmark runner with append-only persistence."""

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
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from evaluation.config import freeze_config, sha256_file
from evaluation.dataset import read_dataset, validate_csv
from evaluation.metrics import evaluate_classification
from evaluation.safeguards import assert_development_only
from evaluation.store import RunStore
from models.adapters import Decider08BAdapter, LayaBaseAdapter
from models.tasks import TASK_LABELS, TASK_QUESTIONS


LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
DECIDER_REVISION = "a0a01d6f8135298f400a8c856b355793012ae971"
TASK_TO_COLUMN = {
    "department": "department_label",
    "urgency": "urgency_label",
    "refund": "refund_label",
}
MODEL_ORDER = ("laya_base", "decider_08b")
MODEL_NAMES = {
    "laya_base": "Laya English Base",
    "decider_08b": "Decider 0.8B",
}


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


def _package_versions() -> dict[str, str | None]:
    names = (
        "laya",
        "decider-ai",
        "torch",
        "transformers",
        "huggingface-hub",
        "pandas",
        "numpy",
        "scikit-learn",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _ground_truth(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        task: str(row[column])
        for task, column in TASK_TO_COLUMN.items()
    }


def _summarize_model(
    records: list[dict[str, Any]],
    *,
    total_records: int,
    load_seconds: float | None,
    wall_seconds: float,
    inference_times_ms: list[float],
    metadata: dict[str, Any],
    errors: list[dict[str, str]],
    warnings_found: list[str],
) -> dict[str, Any]:
    successful = [record for record in records if "predictions" in record]
    task_metrics: dict[str, Any] = {}

    for task, labels in TASK_LABELS.items():
        y_true = [record["ground_truth"][task] for record in successful]
        y_pred = [record["predictions"][task] for record in successful]
        probability_rows = [record["probabilities"].get(task) for record in successful]
        probabilities = (
            probability_rows
            if probability_rows and all(row is not None for row in probability_rows)
            else None
        )
        task_metrics[task] = evaluate_classification(
            y_true,
            y_pred,
            labels=labels,
            probabilities=probabilities,
            positive_label="true" if task == "refund" else None,
        ) if successful else None

    difficulties: dict[str, Any] = {}
    for difficulty in sorted({str(row.get("difficulty", "")) for row in records}):
        subgroup = [
            record
            for record in successful
            if str(record.get("difficulty", "")) == difficulty
        ]
        task_breakdown = {}
        for task, labels in TASK_LABELS.items():
            if subgroup:
                task_breakdown[task] = evaluate_classification(
                    [row["ground_truth"][task] for row in subgroup],
                    [row["predictions"][task] for row in subgroup],
                    labels=labels,
                )
            else:
                task_breakdown[task] = None
        difficulties[difficulty] = {
            "count": len(subgroup),
            "small_subgroup": len(subgroup) < 10,
            "tasks": task_breakdown,
        }

    latency = {
        "count": len(inference_times_ms),
        "median_ms": float(np.median(inference_times_ms)) if inference_times_ms else None,
        "p95_ms": float(np.percentile(inference_times_ms, 95)) if inference_times_ms else None,
        "mean_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
        "throughput_records_per_second": (
            len(inference_times_ms) / (sum(inference_times_ms) / 1000.0)
            if inference_times_ms and sum(inference_times_ms) > 0
            else None
        ),
        "wall_seconds": wall_seconds,
    }

    return {
        "status": "complete" if len(successful) == total_records and not errors else "partial",
        "expected_records": total_records,
        "successful_records": len(successful),
        "failed_records": total_records - len(successful),
        "load_seconds": load_seconds,
        "cold_start_seconds": (
            load_seconds + inference_times_ms[0] / 1000.0
            if load_seconds is not None and inference_times_ms
            else None
        ),
        "latency": latency,
        "resource_usage": metadata.pop("resource_usage", {}),
        "runtime": metadata,
        "metrics": task_metrics,
        "difficulty": difficulties,
        "warnings": warnings_found,
        "errors": errors,
        "predictions": records,
    }


def run_development_benchmark(
    *,
    dev_path: str | Path,
    results_root: str | Path = "results",
    expected_rows: int = 350,
    device: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    adapter_factories: Mapping[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    """Run both baseline adapters sequentially on development records only.

    `adapter_factories` is for tests and local mock runs. Normal application use
    loads the pinned official checkpoints.
    """
    assert_development_only("dev", "baseline development benchmark")
    dev_path = Path(dev_path)
    validation = validate_csv(dev_path, role="dev", expected_rows=expected_rows)
    if not validation.is_valid:
        messages = "; ".join(issue.message for issue in validation.issues if issue.severity == "error")
        raise ValueError(f"Development dataset validation failed: {messages}")
    if validation.row_count != expected_rows:
        raise ValueError(
            f"Expected exactly {expected_rows} development rows; found {validation.row_count}."
        )

    frame = read_dataset(dev_path)
    if frame.empty:
        raise ValueError("Development dataset contains no records.")

    import torch

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if chosen_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    project_root = dev_path.resolve().parent.parent
    hf_home = project_root / "models" / "hf-cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    if adapter_factories is None:
        def load_laya() -> LayaBaseAdapter:
            return LayaBaseAdapter.load(device=chosen_device, revision=LAYA_REVISION)

        def load_decider() -> Decider08BAdapter:
            from huggingface_hub import snapshot_download

            checkpoint = snapshot_download(
                repo_id="Mapika/decider-0.8b",
                revision=DECIDER_REVISION,
                cache_dir=str(hf_home / "hub"),
            )
            return Decider08BAdapter.load(
                checkpoint=checkpoint,
                device=chosen_device,
                use_graphs=False,
            )

        factories: Mapping[str, Callable[[], Any]] = {
            "laya_base": load_laya,
            "decider_08b": load_decider,
        }
    else:
        factories = adapter_factories

    started_at = datetime.now(timezone.utc)
    run_id = f"dev-{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    config = {
        "phase": "development",
        "dataset_role": "dev",
        "dataset_hashes": {"dev": sha256_file(dev_path)},
        "dataset_rows": len(frame),
        "model_order": list(MODEL_ORDER),
        "hardware": {
            "selected_device": chosen_device,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if chosen_device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        },
        "models": {
            "laya_base": {
                "model_id": "convaiinnovations/laya",
                "checkpoint_revision": LAYA_REVISION,
                "device": chosen_device,
                "precision_policy": "official Laya runtime-selected precision; actual dtype recorded after load",
            },
            "decider_08b": {
                "model_id": "Mapika/decider-0.8b",
                "checkpoint_revision": DECIDER_REVISION,
                "device": chosen_device,
                "precision_policy": "official Decider default for selected device",
                "execution_mode": "eager (use_graphs=False)",
            },
        },
        "task_labels": {key: list(value) for key, value in TASK_LABELS.items()},
        "task_schema": TASK_QUESTIONS,
        "inference_protocol": {
            "input_field": "message only",
            "difficulty_used_as_input": False,
            "decider_cuda_graphs": False,
            "probabilities": "preserve only validated model-returned distributions",
        },
        "package_versions": _package_versions(),
        "random_seed": None,
        "calibration": {"method": None, "fit_role": None},
    }
    frozen = freeze_config(config)

    model_results: dict[str, Any] = {}
    all_errors: list[dict[str, str]] = []
    for model_index, model_key in enumerate(MODEL_ORDER):
        if model_key not in factories:
            raise ValueError(f"Missing adapter factory for {model_key}.")
        model_name = MODEL_NAMES[model_key]
        if progress_callback:
            progress_callback({
                "model": model_key,
                "model_name": model_name,
                "phase": "loading",
                "completed": 0,
                "total": len(frame),
                "model_index": model_index,
            })

        adapter = None
        model_records: list[dict[str, Any]] = []
        inference_times_ms: list[float] = []
        model_errors: list[dict[str, str]] = []
        warning_collector = _WarningCollector()
        root_logger = logging.getLogger()
        root_logger.addHandler(warning_collector)
        captured_warnings: list[warnings.WarningMessage] = []
        load_seconds: float | None = None
        runtime_metadata: dict[str, Any] = {}
        wall_started = time.perf_counter()
        if chosen_device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                adapter = factories[model_key]()
                load_seconds = getattr(adapter, "load_seconds", None)
                for row_index, row in frame.iterrows():
                    truth = _ground_truth(row)
                    base_record = {
                        "ticket_id": str(row["ticket_id"]),
                        "message": str(row["message"]),
                        "difficulty": str(row["difficulty"]),
                        "ground_truth": truth,
                    }
                    try:
                        prediction = adapter.predict(str(row["message"]))
                        record = {
                            **base_record,
                            "predictions": prediction.predictions,
                            "probabilities": prediction.probabilities,
                            "probability_notes": prediction.probability_notes,
                            "inference_ms": prediction.inference_ms,
                            "raw_output": prediction.raw_output,
                        }
                        model_records.append(record)
                        inference_times_ms.append(float(prediction.inference_ms))
                        runtime_metadata.update(prediction.metadata)
                    except Exception as exc:
                        error = {
                            "ticket_id": str(row["ticket_id"]),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        model_errors.append(error)
                        all_errors.append({"model": model_key, **error})
                        model_records.append({**base_record, **error})

                    completed = row_index + 1
                    if progress_callback and (completed % 5 == 0 or completed == len(frame)):
                        progress_callback({
                            "model": model_key,
                            "model_name": model_name,
                            "phase": "predicting",
                            "completed": completed,
                            "total": len(frame),
                            "model_index": model_index,
                        })
        except Exception as exc:
            error = {"stage": "load_or_run", "error": f"{type(exc).__name__}: {exc}"}
            model_errors.append(error)
            all_errors.append({"model": model_key, **error})
        finally:
            root_logger.removeHandler(warning_collector)
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

        warnings_found = [str(item.message) for item in captured_warnings]
        warnings_found.extend(warning_collector.messages)
        if chosen_device.startswith("cuda") and torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())
        else:
            peak_memory = None
        wall_seconds = time.perf_counter() - wall_started
        metadata = {
            "resource_usage": {
                "peak_gpu_memory_bytes": peak_memory,
                "peak_gpu_memory_mb": peak_memory / (1024 * 1024) if peak_memory is not None else None,
                "cpu_memory_bytes": None,
                "cpu_memory_unavailable_reason": "psutil is not installed; no CPU memory measurement was taken.",
            }
        }
        model_results[model_key] = {
            "display_name": model_name,
            **_summarize_model(
                model_records,
                total_records=len(frame),
                load_seconds=load_seconds,
                wall_seconds=wall_seconds,
                inference_times_ms=inference_times_ms,
                metadata={**runtime_metadata, **metadata},
                errors=model_errors,
                warnings_found=warnings_found,
            ),
        }
        if progress_callback:
            progress_callback({
                "model": model_key,
                "model_name": model_name,
                "phase": "completed",
                "completed": len(frame),
                "total": len(frame),
                "model_index": model_index,
            })

    ended_at = datetime.now(timezone.utc)
    status = (
        "complete"
        if all(model_results[key]["status"] == "complete" for key in MODEL_ORDER)
        else "partial"
    )
    payload = {
        "run_id": run_id,
        "status": status,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": ended_at.isoformat(),
        "dataset": {
            "role": "dev",
            "path": str(dev_path.resolve()),
            "rows": len(frame),
            "sha256": config["dataset_hashes"]["dev"],
            "validation": validation.to_dict(),
        },
        "configuration": frozen.to_dict(),
        "configuration_sha256": frozen.sha256,
        "experiment_fingerprint": frozen.sha256,
        "models": model_results,
        "errors": all_errors,
        "warnings": sorted({warning for model in model_results.values() for warning in model["warnings"]}),
        "limitations": [
            "Development metrics are for development and pipeline checks; they are not final held-out performance claims.",
            "The two model libraries use different inference implementations. Device, precision and Decider eager mode are recorded for interpretation.",
        ],
    }
    saved_path = RunStore(results_root).save_run("development", run_id, payload)
    return {"path": str(saved_path), "record": payload}


def load_latest_development_run(results_root: str | Path = "results") -> dict[str, Any] | None:
    """Load the newest saved development run without opening any dataset files."""
    run_root = Path(results_root) / "development"
    manifests = list(run_root.glob("*/run.json")) if run_root.is_dir() else []
    if not manifests:
        return None
    latest = max(manifests, key=lambda path: path.stat().st_mtime)
    record = json.loads(latest.read_text(encoding="utf-8"))
    payload = {"path": str(latest.parent), **record["payload"]}
    sidecars = sorted(latest.parent.glob("probability_metrics_reanalysis_v*.json"))
    if sidecars:
        sidecar = json.loads(sidecars[-1].read_text(encoding="utf-8"))
        payload["probability_metrics_reanalysis"] = sidecar
        for model_key, model_metrics in sidecar.get("models", {}).items():
            if model_key in payload.get("models", {}):
                payload["models"][model_key]["metrics"] = model_metrics
    return payload


def reanalyze_saved_development_probabilities(
    results_root: str | Path = "results",
) -> dict[str, Any]:
    """Recompute saved baseline probability metrics from raw model outputs only."""
    from evaluation.reanalysis import metrics_from_saved_predictions

    run = load_latest_development_run(results_root)
    if run is None:
        raise ValueError("No saved development run is available for reanalysis.")
    sidecar_dir = Path(run["path"])
    existing = sorted(sidecar_dir.glob("probability_metrics_reanalysis_v*.json"))
    version = len(existing) + 1
    sidecar_path = sidecar_dir / f"probability_metrics_reanalysis_v{version}.json"
    models = {}
    for model_key, model in run.get("models", {}).items():
        records = [record for record in model.get("predictions", []) if "predictions" in record]
        models[model_key] = metrics_from_saved_predictions(
            records,
            TASK_LABELS,
            positive_labels={"refund": "true"},
        )
    sidecar = {
        "analysis_version": version,
        "source_run_id": run["run_id"],
        "method": "Recomputed from saved model outputs; rows with sums within 1e-3 are normalized only for scoring.",
        "test_data_used": False,
        "models": models,
    }
    with sidecar_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"path": str(sidecar_path), **sidecar}
