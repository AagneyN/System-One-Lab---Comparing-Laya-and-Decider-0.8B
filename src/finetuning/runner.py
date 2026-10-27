"""Reproducible, development-only fine-tuning for the official Laya checkpoint."""

from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from benchmark.runner import (
    DECIDER_REVISION,
    LAYA_REVISION,
    MODEL_NAMES,
    TASK_TO_COLUMN,
    load_latest_development_run,
)
from evaluation.calibration import TemperatureCalibrator
from evaluation.config import freeze_config, sha256_file
from evaluation.dataset import read_dataset, validate_csv
from evaluation.safeguards import assert_development_only
from evaluation.statistics import paired_comparison
from evaluation.store import RunStore
from evaluation.metrics import evaluate_classification
from evaluation.reanalysis import recover_probability_distribution
from models.adapters import LayaFineTunedAdapter
from models.tasks import TASK_LABELS, TASK_QUESTIONS


FINETUNED_MODEL_KEY = "laya_finetuned"
FINETUNED_DISPLAY_NAME = "Laya Fine-Tuned"
SPLIT_SEED = 20261027
CALIBRATION_ROLE = "development_calibration"


def load_latest_finetuning_run(results_root: str | Path = "results") -> dict[str, Any] | None:
    run_root = Path(results_root) / "fine_tuning"
    manifests = list(run_root.glob("*/run.json")) if run_root.is_dir() else []
    if not manifests:
        return None
    latest = max(manifests, key=lambda path: path.stat().st_mtime)
    record = json.loads(latest.read_text(encoding="utf-8"))
    payload = {"path": str(latest.parent), **record["payload"]}
    artifact_dir = Path(payload.get("artifacts", {}).get("directory", ""))
    revisions = sorted(artifact_dir.glob("probability_analysis_reanalysis_v*.json")) if artifact_dir.is_dir() else []
    if revisions:
        payload["analysis_revision"] = json.loads(revisions[-1].read_text(encoding="utf-8"))
        payload["calibrators"] = payload["analysis_revision"].get(
            "calibrators", payload.get("calibrators", {})
        )
    return payload


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _versions() -> dict[str, str | None]:
    output = {}
    for name in (
        "laya",
        "torch",
        "transformers",
        "huggingface-hub",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def _split_development(
    frame,
    *,
    seed: int = SPLIT_SEED,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.20,
) -> dict[str, list[int]]:
    """Split development rows by ticket, stratifying both holdouts on refund label."""
    if not 0 < calibration_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be strictly between zero and one.")
    if calibration_fraction + validation_fraction >= 1:
        raise ValueError("Training, calibration and validation splits must all be nonempty.")
    if len(frame) < 3:
        raise ValueError("At least three development records are needed for the split.")

    indices = np.arange(len(frame))
    train_calibration_ids, validation_ids = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=frame["refund_label"].astype(str),
    )
    calibration_share_of_remainder = calibration_fraction / (
        1.0 - validation_fraction
    )
    train_ids, calibration_ids = train_test_split(
        train_calibration_ids,
        test_size=calibration_share_of_remainder,
        random_state=seed + 1,
        stratify=frame.iloc[train_calibration_ids]["refund_label"].astype(str),
    )
    result = {
        "train": sorted(int(i) for i in train_ids),
        "calibration": sorted(int(i) for i in calibration_ids),
        "validation": sorted(int(i) for i in validation_ids),
    }
    groups = [set(values) for values in result.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Development split groups unexpectedly overlap.")
    if set.union(*groups) != set(indices.tolist()):
        raise RuntimeError("Development split did not assign every ticket exactly once.")
    return result


def _distribution(frame, indices: Sequence[int]) -> dict[str, dict[str, int]]:
    selected = frame.iloc[list(indices)]
    return {
        column: {str(label): int(count) for label, count in selected[column].value_counts().items()}
        for column in (
            "department_label",
            "urgency_label",
            "refund_label",
            "difficulty",
        )
    }


def _rows_for_laya(frame, indices: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        row = frame.iloc[index]
        rows.append(
            {
                # Keep the same structured state passed by both Laya adapters.
                "state": {"ticket": str(row["message"])},
                "questions": TASK_QUESTIONS,
                "expected": {
                    task: str(row[column])
                    for task, column in TASK_TO_COLUMN.items()
                },
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _summarize_predictions(
    frame,
    indices: Sequence[int],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = [predictions[str(frame.iloc[i]["ticket_id"])] for i in indices]
    metrics: dict[str, Any] = {}
    for task, labels in TASK_LABELS.items():
        truth = [str(frame.iloc[i][TASK_TO_COLUMN[task]]) for i in indices]
        predicted = [str(record["predictions"][task]) for record in records]
        raw_probs = [
            recover_probability_distribution(record, task, labels)
            for record in records
        ]
        cal_probs = [record.get("calibrated_probabilities", {}).get(task) for record in records]
        raw_metric = evaluate_classification(
            truth,
            predicted,
            labels=labels,
            probabilities=(raw_probs if raw_probs and all(p is not None for p in raw_probs) else None),
            positive_label="true" if task == "refund" else None,
        )
        cal_metric = evaluate_classification(
            truth,
            predicted,
            labels=labels,
            probabilities=(cal_probs if cal_probs and all(p is not None for p in cal_probs) else None),
            positive_label="true" if task == "refund" else None,
        )
        metrics[task] = {
            **{key: value for key, value in raw_metric.items() if key != "probability_metrics"},
            "probability_metrics": {
                "raw": raw_metric["probability_metrics"],
                "calibrated": cal_metric["probability_metrics"],
            },
        }
    return {"count": len(records), "metrics": metrics, "predictions": records}


def _fit_calibrators(
    *,
    frame,
    indices: Sequence[int],
    prediction_map: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    fitted: dict[str, dict[str, Any]] = {}
    for model_key, by_ticket in prediction_map.items():
        fitted[model_key] = {}
        for task, labels in TASK_LABELS.items():
            selected = [by_ticket.get(str(frame.iloc[i]["ticket_id"])) for i in indices]
            recovered = [
                recover_probability_distribution(row, task, labels) if row else None
                for row in selected
            ]
            valid = [probability for probability in recovered if probability is not None]
            if len(valid) != len(indices):
                fitted[model_key][task] = {
                    "available": False,
                    "reason": "At least one calibration record has no valid model-returned distribution.",
                    "sample_count": len(valid),
                }
                continue
            try:
                calibrator = TemperatureCalibrator.fit(
                    valid,
                    [str(frame.iloc[i][TASK_TO_COLUMN[task]]) for i in indices],
                    labels=labels,
                    data_role=CALIBRATION_ROLE,
                )
                fitted[model_key][task] = {
                    "available": True,
                    **calibrator.to_dict(),
                }
            except (TypeError, ValueError) as exc:
                fitted[model_key][task] = {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "sample_count": len(valid),
                }
    return fitted


def _apply_calibrator(
    probabilities: Mapping[str, Mapping[str, float] | None],
    calibrators: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float] | None]:
    from evaluation.calibration import TemperatureCalibrator

    output: dict[str, dict[str, float] | None] = {}
    for task, labels in TASK_LABELS.items():
        row = probabilities.get(task)
        spec = calibrators.get(task, {})
        if row is None or not spec.get("available"):
            output[task] = None
            continue
        calibrator = TemperatureCalibrator(
            labels=tuple(spec["labels"]),
            temperature=float(spec["temperature"]),
            fit_data_role=str(spec["fit_data_role"]),
            sample_count=int(spec["sample_count"]),
            nll_before=float(spec["nll_before"]),
            nll_after=float(spec["nll_after"]),
        )
        output[task] = calibrator.transform([row])[0]
    return output


def _checkpoint_manifest(checkpoint: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(p for p in checkpoint.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files or not any(item["path"] == "model.safetensors" for item in files):
        raise RuntimeError("Laya fine-tuning returned without a model.safetensors checkpoint.")
    if not any(item["path"] == "rl_agent_config.json" for item in files):
        raise RuntimeError("The saved checkpoint is missing rl_agent_config.json.")
    if not any(item["path"] == "questions.json" for item in files):
        raise RuntimeError("The saved checkpoint is missing questions.json.")
    return files


def run_laya_finetuning(
    *,
    dev_path: str | Path,
    results_root: str | Path = "results",
    models_root: str | Path = "models",
    expected_rows: int = 350,
    seed: int = SPLIT_SEED,
    device: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    finetune_callable: Callable[..., dict[str, Any]] | None = None,
    adapter_factory: Callable[..., LayaFineTunedAdapter] | None = None,
) -> dict[str, Any]:
    """Fine-tune the pinned Laya English base using development tickets only.

    Three ticket-disjoint development splits are used: training, calibration and
    validation. Calibration is performed after training with the shared scalar
    probability calibrators, and validation is reserved for independent assessment.
    """
    assert_development_only("dev", "Laya fine-tuning")
    dev_path = Path(dev_path).resolve()
    if dev_path.name.lower() != "tickets_dev.csv":
        raise ValueError("Fine-tuning accepts only a file named tickets_dev.csv.")
    validation = validate_csv(dev_path, role="dev", expected_rows=expected_rows)
    if not validation.is_valid or validation.row_count != expected_rows:
        details = "; ".join(issue.message for issue in validation.issues if issue.severity == "error")
        raise ValueError(
            f"Development dataset must validate with exactly {expected_rows} rows. "
            f"Found {validation.row_count}; {details}"
        )
    frame = read_dataset(dev_path)

    baseline = load_latest_development_run(results_root)
    if baseline is None:
        raise ValueError("A saved complete Laya/Decider development baseline is required before fine-tuning.")
    if baseline.get("status") != "complete":
        raise ValueError("The latest development baseline is not complete.")
    if baseline.get("dataset", {}).get("sha256") != sha256_file(dev_path):
        raise ValueError("The saved baseline was run on a different development CSV revision.")
    for key in ("laya_base", "decider_08b"):
        model = baseline.get("models", {}).get(key, {})
        if model.get("status") != "complete" or len(model.get("predictions", [])) != expected_rows:
            raise ValueError(f"Baseline {key} does not contain all development predictions.")

    project_root = dev_path.parent.parent
    hf_home = project_root / "models" / "hf-cache"
    os.environ["HF_HOME"] = str(hf_home)
    base_checkpoint = (
        hf_home
        / "hub"
        / "models--convaiinnovations--laya"
        / "snapshots"
        / LAYA_REVISION
    )
    required_checkpoint_paths = (
        base_checkpoint / "model.safetensors",
        base_checkpoint / "rl_agent_config.json",
        base_checkpoint / "encoder" / "config.json",
        base_checkpoint / "tokenizer" / "tokenizer.json",
    )
    missing_checkpoint_files = [str(path) for path in required_checkpoint_paths if not path.is_file()]
    if missing_checkpoint_files:
        raise FileNotFoundError(
            "The exact pinned Laya English base checkpoint is not complete in the local cache: "
            + ", ".join(missing_checkpoint_files)
        )

    splits = _split_development(frame, seed=seed)
    run_started = datetime.now(timezone.utc)
    run_id = f"ft-{run_started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    root = Path(results_root)
    artifact_dir = root / "fine_tuning_artifacts" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(models_root) / "laya-finetuned" / run_id
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint directory: {checkpoint}")

    input_paths = {
        name: artifact_dir / f"{name}.jsonl"
        for name in ("train", "calibration", "validation")
    }
    for name, indices in splits.items():
        _write_jsonl(input_paths[name], _rows_for_laya(frame, indices))

    split_info = {
        "seed": seed,
        "method": "ticket-level sequential stratified split on refund_label",
        "fractions_requested": {"train": 0.70, "calibration": 0.10, "validation": 0.20},
        "counts": {name: len(values) for name, values in splits.items()},
        "ticket_ids": {
            name: [str(frame.iloc[index]["ticket_id"]) for index in indices]
            for name, indices in splits.items()
        },
        "label_distributions": {
            name: _distribution(frame, indices)
            for name, indices in splits.items()
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
    }
    (artifact_dir / "split.json").write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    import torch

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if chosen_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA fine-tuning was requested but CUDA is unavailable.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if chosen_device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info()
    else:
        gpu_name = None
        free_gpu_bytes = total_gpu_bytes = None

    from laya.train import TrainConfig, finetune as official_finetune

    official_settings = {
        "epochs": 4,
        "micro_batch": 2,
        "grad_accum": 4,
        "encoder_lr": 2.5e-5,
        "head_lr": 1e-4,
        "loss": "rlcd",
        "freeze_encoder": True,
        "amp": chosen_device.startswith("cuda"),
        "gradient_checkpointing": False,
        "seed": seed,
        "calib_max": 0,
        "calib_frac": 0.0,
        "calib_seed": seed,
        "eval_data": str(input_paths["validation"].resolve()),
        "log_every": 20,
    }
    config = {
        "phase": "fine_tuning",
        "training_dataset_role": "dev",
        "calibration_fit_role": CALIBRATION_ROLE,
        "dataset_hashes": {
            "dev": sha256_file(dev_path),
            **{f"dev_{name}": sha256_file(path) for name, path in input_paths.items()},
        },
        "dataset_rows": len(frame),
        "baseline_run_id": baseline["run_id"],
        "base_model": {
            "model_id": "convaiinnovations/laya",
            "checkpoint_revision": LAYA_REVISION,
            "local_checkpoint": str(Path(base_checkpoint).resolve()),
        },
        "output_checkpoint": str(checkpoint.resolve()),
        "split": split_info,
        "training_method": "official laya.train.finetune; RLCD; supported head-only fine-tuning",
        "training_settings": official_settings,
        "validation_policy": "Independent ticket split; fixed epochs, no early stopping/checkpoint search in the official trainer.",
        "calibration_policy": "Shared task-wise temperature scaling on the separate development_calibration ticket split; never on validation or test labels.",
        "task_labels": {task: list(labels) for task, labels in TASK_LABELS.items()},
        "task_schema": TASK_QUESTIONS,
        "input_policy": "Message is the only model state; difficulty is metadata and never an input.",
        "hardware": {
            "device": chosen_device,
            "gpu_name": gpu_name,
            "free_gpu_memory_before_training_bytes": free_gpu_bytes,
            "total_gpu_memory_bytes": total_gpu_bytes,
        },
        "package_versions": _versions(),
        "random_seed": seed,
    }
    frozen = freeze_config(config)
    if finetune_callable is None:
        finetune_callable = official_finetune
    if adapter_factory is None:
        adapter_factory = LayaFineTunedAdapter.load

    if progress_callback:
        progress_callback({"phase": "training", "run_id": run_id, "checkpoint": str(checkpoint)})

    started_clock = time.perf_counter()
    training_summary = None
    errors: list[str] = []
    warnings_found: list[str] = []
    calibration_predictions: dict[str, dict] = {}
    validation_predictions: dict[str, dict] = {}
    checkpoint_files: list[dict[str, Any]] = []
    checkpoint_verified = False
    try:
        train_config = TrainConfig(**official_settings)
        train_config.validate()
        log_path = artifact_dir / "training.log"
        with log_path.open("x", encoding="utf-8", newline="\n") as log_handle:
            tee = _Tee(sys.stdout, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                training_summary = finetune_callable(
                    data=str(input_paths["train"].resolve()),
                    model_dir=str(Path(base_checkpoint).resolve()),
                    output_dir=str(checkpoint.resolve()),
                    config=train_config,
                    device=chosen_device,
                )
        checkpoint_files = _checkpoint_manifest(checkpoint)

        if progress_callback:
            progress_callback({"phase": "loading_finetuned_checkpoint", "run_id": run_id})
        adapter = adapter_factory(checkpoint=checkpoint, device=chosen_device)
        try:
            for part, indices, output in (
                ("calibration", splits["calibration"], calibration_predictions),
                ("validation", splits["validation"], validation_predictions),
            ):
                for position, index in enumerate(indices, start=1):
                    row = frame.iloc[index]
                    prediction = adapter.predict(str(row["message"]))
                    output[str(row["ticket_id"])] = {
                        "ticket_id": str(row["ticket_id"]),
                        "message": str(row["message"]),
                        "difficulty": str(row["difficulty"]),
                        "ground_truth": {
                            task: str(row[column])
                            for task, column in TASK_TO_COLUMN.items()
                        },
                        "predictions": prediction.predictions,
                        "probabilities": prediction.probabilities,
                        "probability_notes": prediction.probability_notes,
                        "inference_ms": prediction.inference_ms,
                        "raw_output": prediction.raw_output,
                        "metadata": prediction.metadata,
                    }
                    if progress_callback and position % 10 == 0:
                        progress_callback({
                            "phase": f"predicting_{part}",
                            "completed": position,
                            "total": len(indices),
                            "run_id": run_id,
                        })
        finally:
            adapter.backend = None
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        base_dev = baseline["models"]["laya_base"]["predictions"]
        decider_dev = baseline["models"]["decider_08b"]["predictions"]
        base_by_id = {row["ticket_id"]: row for row in base_dev if "predictions" in row}
        decider_by_id = {row["ticket_id"]: row for row in decider_dev if "predictions" in row}
        all_calibration_ids = [
            str(frame.iloc[index]["ticket_id"]) for index in splits["calibration"]
        ]
        if any(ticket not in base_by_id or ticket not in decider_by_id for ticket in all_calibration_ids):
            raise RuntimeError("The saved baseline is missing calibration ticket predictions.")

        calibration_map = {
            "laya_base": {ticket: base_by_id[ticket] for ticket in all_calibration_ids},
            "decider_08b": {ticket: decider_by_id[ticket] for ticket in all_calibration_ids},
            FINETUNED_MODEL_KEY: calibration_predictions,
        }
        all_calibrators = _fit_calibrators(
            frame=frame,
            indices=splits["calibration"],
            prediction_map=calibration_map,
        )
        (artifact_dir / "calibrators.json").write_text(
            json.dumps(all_calibrators, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        for output in (calibration_predictions, validation_predictions):
            for ticket, record in output.items():
                record["calibrated_probabilities"] = _apply_calibrator(
                    record["probabilities"],
                    all_calibrators[FINETUNED_MODEL_KEY],
                )
        (artifact_dir / "calibration_predictions.json").write_text(
            json.dumps(list(calibration_predictions.values()), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        (artifact_dir / "validation_predictions.json").write_text(
            json.dumps(list(validation_predictions.values()), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

        base_validation = {
            ticket: base_by_id[ticket]
            for ticket in [str(frame.iloc[i]["ticket_id"]) for i in splits["validation"]]
        }
        for record in base_validation.values():
            record["calibrated_probabilities"] = _apply_calibrator(
                record.get("probabilities", {}), all_calibrators["laya_base"]
            )

        base_summary = _summarize_predictions(frame, splits["validation"], base_validation)
        finetuned_summary = _summarize_predictions(frame, splits["validation"], validation_predictions)
        comparison = {}
        for task, labels in TASK_LABELS.items():
            paired_truth = [str(frame.iloc[i][TASK_TO_COLUMN[task]]) for i in splits["validation"]]
            comparison[task] = paired_comparison(
                paired_truth,
                [base_validation[str(frame.iloc[i]["ticket_id"])]["predictions"][task] for i in splits["validation"]],
                [validation_predictions[str(frame.iloc[i]["ticket_id"])]["predictions"][task] for i in splits["validation"]],
                labels=labels,
                seed=seed,
                bootstrap_replicates=1000,
            )

        checkpoint_verified = (
            len(calibration_predictions) == len(splits["calibration"])
            and len(validation_predictions) == len(splits["validation"])
        )
        if not checkpoint_verified:
            raise RuntimeError("Fine-tuned checkpoint loaded, but its development predictions were incomplete.")

        training_duration = time.perf_counter() - started_clock
        status = "complete"
        warnings_found.extend(
            [
                "Only 245 development tickets were used for training; validation and calibration are development-only. Generalization claims remain limited by the small dataset.",
                "Laya's official training API has a fixed epoch budget but no early-stopping or checkpoint-selection hook; validation was assessed after the frozen four-epoch run.",
                "The supported freeze_encoder option updates the decision head while keeping the base encoder frozen; this is a constrained-GPU fine-tuning configuration, not full-encoder fine-tuning.",
            ]
        )
        record = {
            "run_id": run_id,
            "status": status,
            "started_at_utc": run_started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_duration_seconds": training_duration,
            "dataset": {
                "role": "dev",
                "path": str(dev_path),
                "rows": len(frame),
                "sha256": sha256_file(dev_path),
                "validation": validation.to_dict(),
            },
            "configuration": frozen.to_dict(),
            "configuration_sha256": frozen.sha256,
            "experiment_fingerprint": frozen.sha256,
            "baseline_run_id": baseline["run_id"],
            "split": split_info,
            "training_method": config["training_method"],
            "training_settings": official_settings,
            "official_training_summary": training_summary,
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "verified_loadable": checkpoint_verified,
                "file_manifest": checkpoint_files,
                "calibrators_path": str((artifact_dir / "calibrators.json").resolve()),
            },
            "calibrators": all_calibrators,
            "validation_comparison": {
                "role": "development_validation",
                "rows": len(splits["validation"]),
                "models": {
                    "laya_base": {
                        "display_name": MODEL_NAMES["laya_base"],
                        **base_summary,
                    },
                    FINETUNED_MODEL_KEY: {
                        "display_name": FINETUNED_DISPLAY_NAME,
                        **finetuned_summary,
                    },
                },
                "paired_comparisons": comparison,
            },
            "artifacts": {
                "directory": str(artifact_dir.resolve()),
                "train_log": str((artifact_dir / "training.log").resolve()),
                "split_manifest": str((artifact_dir / "split.json").resolve()),
                "training_inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
            },
            "hardware": {
                "device": chosen_device,
                "gpu_name": gpu_name,
                "free_gpu_memory_before_training_bytes": free_gpu_bytes,
                "total_gpu_memory_bytes": total_gpu_bytes,
            },
            "package_versions": _versions(),
            "errors": errors,
            "warnings": warnings_found,
        }
    except Exception as exc:
        training_duration = time.perf_counter() - started_clock
        errors.append(f"{type(exc).__name__}: {exc}")
        record = {
            "run_id": run_id,
            "status": "failed",
            "started_at_utc": run_started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_duration_seconds": training_duration,
            "dataset": {
                "role": "dev",
                "path": str(dev_path),
                "rows": len(frame),
                "sha256": sha256_file(dev_path),
            },
            "configuration": frozen.to_dict(),
            "configuration_sha256": frozen.sha256,
            "experiment_fingerprint": frozen.sha256,
            "baseline_run_id": baseline["run_id"],
            "split": split_info,
            "training_settings": official_settings,
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "verified_loadable": checkpoint_verified,
                "file_manifest": checkpoint_files,
            },
            "artifacts": {
                "directory": str(artifact_dir.resolve()),
                "train_log": str((artifact_dir / "training.log").resolve()),
                "split_manifest": str((artifact_dir / "split.json").resolve()),
            },
            "hardware": {
                "device": chosen_device,
                "gpu_name": gpu_name,
                "free_gpu_memory_before_training_bytes": free_gpu_bytes,
                "total_gpu_memory_bytes": total_gpu_bytes,
            },
            "package_versions": _versions(),
            "errors": errors,
            "warnings": warnings_found,
        }

    saved_path = RunStore(root).save_run("fine_tuning", run_id, record)
    if progress_callback:
        progress_callback({"phase": record["status"], "run_id": run_id, "path": str(saved_path)})
    return {"path": str(saved_path), "artifacts_path": str(artifact_dir), "record": record}


def reanalyze_saved_finetuning_probabilities(
    *,
    dev_path: str | Path,
    results_root: str | Path = "results",
) -> dict[str, Any]:
    """Recover rounded model probability vectors and recompute saved dev metrics.

    This function performs no inference and no training. It reads only the saved
    development outputs, their raw model responses, and the frozen dev split IDs.
    """
    dev_path = Path(dev_path).resolve()
    fine_run = load_latest_finetuning_run(results_root)
    baseline = load_latest_development_run(results_root)
    if fine_run is None or fine_run.get("status") != "complete":
        raise ValueError("A complete saved fine-tuning run is required for reanalysis.")
    if baseline is None or baseline.get("run_id") != fine_run.get("baseline_run_id"):
        raise ValueError("The fine-tuning run's referenced baseline is not the latest development run.")
    if sha256_file(dev_path) != fine_run.get("dataset", {}).get("sha256"):
        raise ValueError("The current development CSV does not match the saved fine-tuning run.")
    frame = read_dataset(dev_path)
    id_to_index = {str(row["ticket_id"]): index for index, row in frame.iterrows()}
    split_ids = fine_run["split"]["ticket_ids"]
    split_indices = {
        part: [id_to_index[ticket] for ticket in ids]
        for part, ids in split_ids.items()
    }

    def restore(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        restored = {}
        for record in records:
            if "predictions" not in record:
                continue
            ticket = str(record["ticket_id"])
            item = dict(record)
            item["probabilities"] = {
                task: recover_probability_distribution(record, task, labels)
                for task, labels in TASK_LABELS.items()
            }
            restored[ticket] = item
        return restored

    baseline_by_model = {
        key: restore(baseline["models"][key]["predictions"])
        for key in ("laya_base", "decider_08b")
    }
    artifact_dir = Path(fine_run["artifacts"]["directory"])
    calibration_records = json.loads(
        (artifact_dir / "calibration_predictions.json").read_text(encoding="utf-8")
    )
    validation_records = json.loads(
        (artifact_dir / "validation_predictions.json").read_text(encoding="utf-8")
    )
    fine_calibration = restore(calibration_records)
    fine_validation = restore(validation_records)

    calibration_tickets = split_ids["calibration"]
    validation_tickets = split_ids["validation"]
    calibration_map = {
        "laya_base": {ticket: baseline_by_model["laya_base"][ticket] for ticket in calibration_tickets},
        "decider_08b": {ticket: baseline_by_model["decider_08b"][ticket] for ticket in calibration_tickets},
        FINETUNED_MODEL_KEY: fine_calibration,
    }
    calibrators = _fit_calibrators(
        frame=frame,
        indices=split_indices["calibration"],
        prediction_map=calibration_map,
    )

    base_validation = {
        ticket: baseline_by_model["laya_base"][ticket]
        for ticket in validation_tickets
    }
    for ticket, record in base_validation.items():
        record["calibrated_probabilities"] = _apply_calibrator(
            record["probabilities"], calibrators["laya_base"]
        )
    for predictions in (fine_calibration, fine_validation):
        for record in predictions.values():
            record["calibrated_probabilities"] = _apply_calibrator(
                record["probabilities"], calibrators[FINETUNED_MODEL_KEY]
            )

    base_summary = _summarize_predictions(frame, split_indices["validation"], base_validation)
    fine_summary = _summarize_predictions(frame, split_indices["validation"], fine_validation)
    paired = {}
    for task, labels in TASK_LABELS.items():
        truth = [str(frame.iloc[index][TASK_TO_COLUMN[task]]) for index in split_indices["validation"]]
        paired[task] = paired_comparison(
            truth,
            [base_validation[ticket]["predictions"][task] for ticket in validation_tickets],
            [fine_validation[ticket]["predictions"][task] for ticket in validation_tickets],
            labels=labels,
            seed=int(fine_run["split"]["seed"]),
            bootstrap_replicates=1000,
        )

    revision_files = sorted(artifact_dir.glob("probability_analysis_reanalysis_v*.json"))
    revision_number = len(revision_files) + 1
    calibrators_path = artifact_dir / f"calibrators_reanalysis_v{revision_number}.json"
    analysis = {
        "analysis_version": "probability-recovery-v1",
        "source_fine_tuning_run_id": fine_run["run_id"],
        "source_development_run_id": baseline["run_id"],
        "method": (
            "Recovered model-returned distributions from saved raw outputs when the adapter had marked only their small sum-rounding drift unavailable; "
            "raw output values are preserved and rows within 1e-3 are normalized only for scoring."
        ),
        "test_data_used": False,
        "calibrators": calibrators,
        "calibrators_path": str(calibrators_path.resolve()),
        "validation_comparison": {
            "role": "development_validation",
            "rows": len(validation_tickets),
            "models": {
                "laya_base": {"display_name": MODEL_NAMES["laya_base"], **base_summary},
                FINETUNED_MODEL_KEY: {"display_name": FINETUNED_DISPLAY_NAME, **fine_summary},
            },
            "paired_comparisons": paired,
        },
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with calibrators_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(calibrators, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    revision_path = artifact_dir / f"probability_analysis_reanalysis_v{revision_number}.json"
    with revision_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return {"path": str(revision_path), **analysis}
