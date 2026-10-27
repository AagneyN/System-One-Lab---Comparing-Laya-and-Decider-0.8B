from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from benchmark.runner import (
    MODEL_NAMES,
    load_latest_development_run,
    run_development_benchmark,
)
from benchmark.final_test import (
    FINAL_MODEL_NAMES,
    FINAL_MODEL_ORDER,
    load_latest_final_test_run,
    run_final_test_evaluation,
)
from evaluation.dataset import read_dataset, validate_datasets
from models.tasks import TASK_LABELS
from finetuning.runner import (
    FINETUNED_MODEL_KEY,
    load_latest_finetuning_run,
    run_laya_finetuning,
)


DEV_PATH = ROOT / "data" / "tickets_dev.csv"
TEST_PATH = ROOT / "data" / "tickets_test.csv"
RESULTS_ROOT = ROOT / "results"

st.set_page_config(page_title="System One Lab", page_icon="◎", layout="wide")
st.title("System One Lab")
st.caption("Local, reproducible model benchmarking")

report = validate_datasets(DEV_PATH, TEST_PATH)
dev_ready = report.dev.is_valid and report.dev.row_count == 350
models_installed = all(
    find_spec(name) is not None for name in ("laya", "decider", "huggingface_hub")
)

if report.is_valid:
    st.success("Dataset validation passed. The test set remains held out.")
else:
    st.error("Dataset validation found issues. Review them before running a benchmark.")

overview_tab, runner_tab, finetuning_tab, results_tab, final_test_tab = st.tabs(
    [
        "Overview & Data",
        "Benchmark Runner",
        "Fine-Tuning Lab",
        "Development Results",
        "Final Test Analysis",
    ]
)

with overview_tab:
    col1, col2, col3 = st.columns(3)
    col1.metric("Development records", report.dev.row_count or 0)
    col2.metric("Held-out test records", report.test.row_count or 0)
    col3.metric("Validation errors", report.error_count)

    with st.expander("Dataset validation details", expanded=not report.is_valid):
        for dataset in (report.dev, report.test):
            st.subheader(f"{dataset.role.title()} dataset")
            st.write(
                {
                    "rows": dataset.row_count,
                    "expected_rows": dataset.expected_rows,
                    "columns": dataset.columns,
                    "duplicate_ticket_ids": dataset.duplicate_ticket_ids,
                    "duplicate_messages": dataset.duplicate_messages,
                    "refund_labels_remain_text": dataset.refund_labels_remain_text,
                }
            )
            for issue in dataset.issues:
                line = f"{issue.severity.upper()} — {issue.message}"
                if issue.severity == "error":
                    st.error(line)
                else:
                    st.warning(line)
        st.write("Development/test overlap:", report.overlap)
        for issue in report.issues:
            st.error(issue.message)

    data_tab, heldout_tab = st.tabs(["Development data", "Held-out test data"])
    with data_tab:
        if DEV_PATH.is_file():
            dev_frame = read_dataset(DEV_PATH)
            st.subheader("Development label distributions")
            for column in (
                "department_label",
                "urgency_label",
                "refund_label",
                "difficulty",
            ):
                counts = (
                    dev_frame[column]
                    .value_counts()
                    .rename_axis("label")
                    .reset_index(name="records")
                )
                st.markdown(f"**{column}**")
                st.plotly_chart(
                    px.bar(counts, x="label", y="records", text="records"),
                    use_container_width=True,
                )
            st.dataframe(
                dev_frame[["ticket_id", "message", "difficulty"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Development CSV is missing.")

    with heldout_tab:
        st.warning(
            "Held out: test labels are not displayed and are not used by the development runner."
        )
        st.write(
            {
                "file_exists": TEST_PATH.is_file(),
                "record_count": report.test.row_count,
                "schema_valid": report.test.is_valid,
            }
        )

with runner_tab:
    st.subheader("Run the development baselines")
    st.write(
        "This run evaluates Laya English Base and Decider 0.8B, in that order, "
        "on the same development messages. It does not open the test CSV for inference."
    )
    st.caption(
        "The first model is unloaded before the second is loaded. Decider uses eager CUDA mode "
        "because CUDA graphs require a working Triton installation on this Windows setup."
    )
    status_cols = st.columns(3)
    status_cols[0].metric("Development data", "Ready" if dev_ready else "Not ready")
    status_cols[1].metric("Baseline packages", "Available" if models_installed else "Missing")
    status_cols[2].metric("Run size", f"{report.dev.row_count or 0} records")

    if not models_installed:
        st.error("Laya, Decider, or Hugging Face Hub is not installed in this Python environment.")

    if st.button(
        "Run both models on all 350 development records",
        type="primary",
        disabled=not (report.is_valid and dev_ready and models_installed),
    ):
        progress_bar = st.progress(0.0, text="Preparing development run…")
        with st.status("Starting development benchmark…", expanded=True) as status:
            def update_progress(event: dict) -> None:
                phase = event["phase"]
                model_index = int(event.get("model_index", 0))
                model_name = event.get("model_name", "Model")
                completed = int(event.get("completed", 0))
                total = max(1, int(event.get("total", 1)))
                if phase == "loading":
                    status.update(label=f"Loading {model_name}…", state="running")
                else:
                    fraction = min(1.0, (model_index + completed / total) / 2.0)
                    progress_bar.progress(
                        fraction,
                        text=f"{model_name}: {completed}/{total} records",
                    )
                    if phase == "completed":
                        status.write(f"Finished {model_name}.")

            try:
                outcome = run_development_benchmark(
                    dev_path=DEV_PATH,
                    results_root=RESULTS_ROOT,
                    expected_rows=350,
                    progress_callback=update_progress,
                )
                status.update(
                    label=f"Development run {outcome['record']['status']}",
                    state="complete" if outcome["record"]["status"] == "complete" else "error",
                )
                st.success(f"Saved run: {outcome['record']['run_id']}")
                st.rerun()
            except Exception as exc:
                status.update(label="Development run failed", state="error")
                st.exception(exc)

    st.info(
        "Only saved results appear on the Development Results tab. A partial or failed run is "
        "labelled clearly and is never presented as complete."
    )

with finetuning_tab:
    st.subheader("Fine-Tuning Lab")
    st.write(
        "Fine-tune the pinned Laya English base checkpoint with development training tickets. "
        "Calibration and validation tickets are held out by ticket ID. The held-out test CSV is not opened by the training runner."
    )
    st.warning(
        "The supported RTX 3050 configuration freezes Laya's encoder and trains its decision head. "
        "The official training API has a fixed epoch count and does not expose early stopping. "
        "The 350-ticket development set is small, so validation findings are exploratory."
    )
    latest_finetune = load_latest_finetuning_run(RESULTS_ROOT)
    baseline_run = load_latest_development_run(RESULTS_ROOT)
    baseline_ready = bool(
        baseline_run
        and baseline_run.get("status") == "complete"
        and baseline_run.get("dataset", {}).get("sha256")
    )
    if latest_finetune:
        ft_status = latest_finetune.get("status", "unknown")
        if ft_status == "complete":
            st.success(f"Latest fine-tuning run complete — {latest_finetune['run_id']}")
        else:
            st.warning(f"Latest fine-tuning run {ft_status} — {latest_finetune['run_id']}")
        st.write(
            {
                "training_tickets": latest_finetune.get("split", {}).get("counts", {}).get("train"),
                "calibration_tickets": latest_finetune.get("split", {}).get("counts", {}).get("calibration"),
                "validation_tickets": latest_finetune.get("split", {}).get("counts", {}).get("validation"),
                "checkpoint": latest_finetune.get("checkpoint", {}).get("path"),
                "checkpoint_verified_loadable": latest_finetune.get("checkpoint", {}).get("verified_loadable"),
                "training_seconds": latest_finetune.get("training_duration_seconds"),
            }
        )
        validation_comparison = latest_finetune.get("analysis_revision", {}).get(
            "validation_comparison", latest_finetune.get("validation_comparison", {})
        )
        if latest_finetune.get("analysis_revision"):
            st.caption("Probability analysis was recomputed from saved raw model outputs; no model was rerun.")
        comparison = validation_comparison.get("models", {})
        rows = []
        for model_key, model in comparison.items():
            for task, metric in model.get("metrics", {}).items():
                rows.append(
                    {
                        "Model": model.get("display_name", model_key),
                        "Task": task,
                        "N": metric.get("count"),
                        "Accuracy": metric.get("accuracy"),
                        "Macro F1": metric.get("macro_f1"),
                        "Raw log loss": metric.get("probability_metrics", {}).get("raw", {}).get("log_loss"),
                        "Calibrated log loss": metric.get("probability_metrics", {}).get("calibrated", {}).get("log_loss"),
                    }
                )
        if rows:
            st.markdown("#### Laya Base vs Fine-Tuned Laya — development validation")
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {"Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "Raw log loss": "{:.3f}", "Calibrated log loss": "{:.3f}"},
                    na_rep="unavailable",
                ),
                use_container_width=True,
                hide_index=True,
            )
        if latest_finetune.get("warnings"):
            with st.expander("Fine-tuning warnings and interpretation limits"):
                for warning in latest_finetune["warnings"]:
                    st.warning(warning)
        log_path = latest_finetune.get("artifacts", {}).get("train_log")
        if log_path and Path(log_path).is_file():
            with st.expander("Training log"):
                st.code("\n".join(Path(log_path).read_text(encoding="utf-8").splitlines()[-200:]))
    else:
        st.info("No Laya fine-tuning run has been saved yet.")

    if not baseline_ready:
        st.error("A complete baseline development run on this exact development CSV is required first.")
    if st.button(
        "Start Laya fine-tuning on development data",
        type="primary",
        disabled=not (dev_ready and baseline_ready),
    ):
        bar = st.progress(0.0, text="Preparing development-only splits…")
        with st.status("Fine-tuning Laya…", expanded=True) as status:
            def update_finetuning(event: dict) -> None:
                phase = event.get("phase", "working")
                if phase.startswith("predicting_"):
                    fraction = event.get("completed", 0) / max(1, event.get("total", 1))
                    bar.progress(min(0.99, fraction), text=f"Verifying checkpoint: {phase.removeprefix('predicting_')} {event.get('completed')}/{event.get('total')}")
                status.update(label=f"Fine-tuning: {phase.replace('_', ' ')}", state="running")

            try:
                result = run_laya_finetuning(
                    dev_path=DEV_PATH,
                    results_root=RESULTS_ROOT,
                    models_root=ROOT / "models",
                    progress_callback=update_finetuning,
                )
                if result["record"]["status"] == "complete":
                    bar.progress(1.0, text="Fine-tuning and checkpoint verification complete")
                    status.update(label=f"Fine-tuning run {result['record']['status']}", state="complete")
                    st.rerun()
                else:
                    status.update(label="Fine-tuning failed", state="error")
                    st.error("Training did not complete. The run record and logs were preserved.")
            except Exception as exc:
                status.update(label="Fine-tuning failed", state="error")
                st.exception(exc)

with results_tab:
    st.subheader("Latest development run")
    latest = load_latest_development_run(RESULTS_ROOT)
    if latest is None:
        st.info("No development benchmark has been saved yet. Run both baselines from the Benchmark Runner tab.")
    else:
        run_status = latest.get("status", "unknown")
        if run_status == "complete":
            st.success(f"Complete — {latest['run_id']}")
        else:
            st.warning(
                f"{run_status.title()} — {latest['run_id']}; inspect failed records before interpreting metrics."
            )
        st.caption(
            f"Saved: {latest.get('completed_at_utc', 'unknown')} · "
            f"Development records: {latest.get('dataset', {}).get('rows', 0)} · "
            f"Config SHA-256: {latest.get('configuration_sha256', 'unknown')}"
        )

        model_results = latest.get("models", {})
        metric_rows = []
        summary_cols = st.columns(len(model_results) or 1)
        for col, model_key in zip(summary_cols, model_results):
            model = model_results[model_key]
            col.metric(f"{model['display_name']} status", model["status"].title())
            col.caption(
                f"{model['successful_records']}/{model['expected_records']} predictions · "
                f"load {model.get('load_seconds') if model.get('load_seconds') is not None else 'n/a'} s"
            )
            runtime = model.get("runtime", {})
            st.write(
                f"**{model['display_name']} runtime:** "
                f"{runtime.get('device', 'device unavailable')} · "
                f"{runtime.get('precision', 'precision unavailable')} · "
                f"{runtime.get('execution_mode', 'library default')}"
            )
            latency = model.get("latency", {})
            st.write(
                f"Median {latency.get('median_ms')} ms · P95 {latency.get('p95_ms')} ms · "
                f"throughput {latency.get('throughput_records_per_second')} records/s · "
                f"peak GPU {model.get('resource_usage', {}).get('peak_gpu_memory_mb')} MB"
            )
            for task, metric in model.get("metrics", {}).items():
                if metric is None:
                    continue
                probability = metric.get("probability_metrics", {})
                metric_rows.append(
                    {
                        "Model": model["display_name"],
                        "Task": task,
                        "Records": metric["count"],
                        "Accuracy": metric["accuracy"],
                        "Macro F1": metric["macro_f1"],
                        "Macro precision": metric["macro_precision"],
                        "Macro recall": metric["macro_recall"],
                        "Log loss": probability.get("log_loss"),
                        "Brier": probability.get("brier_score"),
                        "ECE": probability.get("expected_calibration_error"),
                    }
                )

        if metric_rows:
            metrics_frame = pd.DataFrame(metric_rows)
            st.markdown("#### Task metrics")
            st.dataframe(
                metrics_frame.style.format(
                    {
                        "Accuracy": "{:.3f}",
                        "Macro F1": "{:.3f}",
                        "Macro precision": "{:.3f}",
                        "Macro recall": "{:.3f}",
                        "Log loss": "{:.3f}",
                        "Brier": "{:.3f}",
                        "ECE": "{:.3f}",
                    },
                    na_rep="unavailable",
                ),
                use_container_width=True,
                hide_index=True,
            )
            chart_frame = metrics_frame.melt(
                id_vars=["Model", "Task"],
                value_vars=["Accuracy", "Macro F1"],
                var_name="Metric",
                value_name="Score",
            )
            st.plotly_chart(
                px.bar(
                    chart_frame,
                    x="Task",
                    y="Score",
                    color="Model",
                    facet_col="Metric",
                    barmode="group",
                    title="Development accuracy and macro F1",
                    range_y=[0, 1],
                ),
                use_container_width=True,
            )

        if model_results:
            model_key = st.selectbox(
                "Confusion matrix model",
                list(model_results),
                format_func=lambda key: MODEL_NAMES.get(key, key),
            )
            task = st.selectbox("Task", list(TASK_LABELS))
            metric = model_results[model_key].get("metrics", {}).get(task)
            if metric:
                labels = list(TASK_LABELS[task])
                matrix = metric["confusion_matrix"]
                figure = go.Figure(
                    data=go.Heatmap(
                        z=[[matrix[actual][predicted] for predicted in labels] for actual in labels],
                        x=labels,
                        y=labels,
                        colorscale="Blues",
                        text=[[matrix[actual][predicted] for predicted in labels] for actual in labels],
                        texttemplate="%{text}",
                        colorbar={"title": "Records"},
                    )
                )
                figure.update_layout(
                    title=f"{MODEL_NAMES.get(model_key, model_key)} · {task}",
                    xaxis_title="Predicted label",
                    yaxis_title="True label",
                )
                st.plotly_chart(figure, use_container_width=True)

        difficulty_rows = []
        for model in model_results.values():
            for difficulty, group in model.get("difficulty", {}).items():
                for task, metric in group.get("tasks", {}).items():
                    if metric is not None:
                        difficulty_rows.append(
                            {
                                "Model": model["display_name"],
                                "Difficulty": difficulty,
                                "Task": task,
                                "Records": group["count"],
                                "Accuracy": metric["accuracy"],
                                "Macro F1": metric["macro_f1"],
                                "Low sample size": group["small_subgroup"],
                            }
                        )
        if difficulty_rows:
            st.markdown("#### Difficulty breakdown")
            st.dataframe(difficulty_rows, use_container_width=True, hide_index=True)

        prediction_rows: dict[str, dict[str, dict]] = {}
        for model_key, model in model_results.items():
            for record in model.get("predictions", []):
                if "predictions" in record:
                    prediction_rows.setdefault(record["ticket_id"], {})[model_key] = record
        if prediction_rows:
            st.markdown("#### Prediction inspector")
            ticket_ids = list(prediction_rows)
            selected_id = st.selectbox("Development ticket", ticket_ids)
            selected = prediction_rows[selected_id]
            reference = next(iter(selected.values()))
            st.write({"ticket_id": selected_id, "difficulty": reference.get("difficulty")})
            st.text_area("Message", reference.get("message", ""), height=120, disabled=True)
            st.write("Ground truth:", reference.get("ground_truth", {}))
            for model_key, record in selected.items():
                with st.expander(MODEL_NAMES.get(model_key, model_key)):
                    st.write("Predictions:", record.get("predictions"))
                    st.write("Model-returned probabilities:", record.get("probabilities"))
                    st.write("Probability notes:", record.get("probability_notes"))
                    st.write("Inference latency (ms):", record.get("inference_ms"))
                    st.json(record.get("raw_output", {}))

        for model in model_results.values():
            if model.get("warnings"):
                with st.expander(f"{model['display_name']} warnings", expanded=False):
                    for warning in model["warnings"]:
                        st.warning(warning)
            if model.get("errors"):
                with st.expander(f"{model['display_name']} failed records", expanded=True):
                    st.dataframe(model["errors"], use_container_width=True, hide_index=True)

        st.caption(
            "These are development-set results used to verify and refine the pipeline; "
            "they are not final held-out performance claims."
        )

with final_test_tab:
    st.subheader("Final Held-Out Test Analysis")
    st.write(
        "This is a one-shot evaluation of frozen Laya Base, Decider 0.8B, and Fine-Tuned Laya "
        "on the same held-out test ticket IDs. The test labels are used only for scoring after model predictions are frozen."
    )
    latest_test = load_latest_final_test_run(RESULTS_ROOT)
    latest_finetune = load_latest_finetuning_run(RESULTS_ROOT)
    if latest_test:
        if latest_test.get("status") == "complete":
            st.success(f"Final test run complete — {latest_test['run_id']}")
        else:
            st.warning(f"Final test run {latest_test.get('status')} — {latest_test['run_id']}")
        st.caption(
            f"Test records: {latest_test.get('dataset', {}).get('rows')} · "
            f"Dataset SHA-256: {latest_test.get('dataset', {}).get('sha256')} · "
            f"Frozen config SHA-256: {latest_test.get('configuration_sha256')}"
        )
        final_rows = []
        for model_key in FINAL_MODEL_ORDER:
            model = latest_test.get("models", {}).get(model_key, {})
            for task, metric in model.get("metrics", {}).items():
                probs = metric.get("probability_metrics", {})
                final_rows.append(
                    {
                        "Model": model.get("display_name", FINAL_MODEL_NAMES[model_key]),
                        "Task": task,
                        "N": metric.get("count"),
                        "Accuracy": metric.get("accuracy"),
                        "Macro precision": metric.get("macro_precision"),
                        "Macro recall": metric.get("macro_recall"),
                        "Macro F1": metric.get("macro_f1"),
                        "Raw log loss": probs.get("raw", {}).get("log_loss"),
                        "Calibrated log loss": probs.get("calibrated", {}).get("log_loss"),
                    }
                )
        if final_rows:
            st.markdown("#### Task metrics")
            st.dataframe(
                pd.DataFrame(final_rows).style.format(
                    {
                        "Accuracy": "{:.3f}", "Macro precision": "{:.3f}",
                        "Macro recall": "{:.3f}", "Macro F1": "{:.3f}",
                        "Raw log loss": "{:.3f}", "Calibrated log loss": "{:.3f}",
                    },
                    na_rep="unavailable",
                ),
                use_container_width=True,
                hide_index=True,
            )
            plot_rows = pd.DataFrame(final_rows).melt(
                id_vars=["Model", "Task"],
                value_vars=["Accuracy", "Macro F1"],
                var_name="Metric",
                value_name="Score",
            )
            st.plotly_chart(
                px.bar(
                    plot_rows,
                    x="Task",
                    y="Score",
                    color="Model",
                    facet_col="Metric",
                    barmode="group",
                    range_y=[0, 1],
                    title="Final held-out accuracy and macro F1",
                ),
                use_container_width=True,
            )
        if latest_test.get("paired_comparisons"):
            comparison_rows = []
            for pair, tasks in latest_test["paired_comparisons"].items():
                for task, value in tasks.items():
                    acc_ci = value["accuracy_difference_ci_95"]
                    f1_ci = value["macro_f1_difference_ci_95"]
                    comparison_rows.append(
                        {
                            "Pair (B − A)": pair,
                            "Task": task,
                            "N": value["n"],
                            "Accuracy difference": value["accuracy_difference"],
                            "Accuracy CI low": acc_ci[0],
                            "Accuracy CI high": acc_ci[1],
                            "Macro-F1 difference": value["macro_f1_difference"],
                            "Macro-F1 CI low": f1_ci[0],
                            "Macro-F1 CI high": f1_ci[1],
                            "McNemar p": value["mcnemar"]["exact_two_sided_p_value"],
                        }
                    )
            st.markdown("#### Paired comparisons (B minus A)")
            st.dataframe(
                pd.DataFrame(comparison_rows).style.format(
                    {column: "{:+.3f}" for column in comparison_rows[0] if "difference" in column or "CI" in column}
                    | {"McNemar p": "{:.4f}"},
                    na_rep="unavailable",
                ),
                use_container_width=True,
                hide_index=True,
            )
        report_path = latest_test.get("report_path")
        if report_path and Path(report_path).is_file():
            st.download_button(
                "Download final test Markdown report",
                Path(report_path).read_text(encoding="utf-8"),
                file_name=Path(report_path).name,
                mime="text/markdown",
            )
        st.warning(
            "The 150-record test set has limited power, especially for rare classes and difficulty subgroups. "
            "If these test results guide model or prompt changes, the test set is consumed for development and a new untouched test set is needed for a new final claim."
        )
    else:
        st.info("No final held-out test run has been saved.")

    if not latest_finetune or latest_finetune.get("status") != "complete":
        st.error("A complete fine-tuning run with a verified checkpoint is required before final evaluation.")
    confirm_final_test = st.checkbox(
        "I confirm the model, task schema, calibration, and inference settings are frozen and authorize the one-shot held-out test evaluation.",
        disabled=not (latest_finetune and latest_finetune.get("status") == "complete"),
    )
    if st.button(
        "Run final evaluation on all 150 held-out test records",
        type="primary",
        disabled=not (confirm_final_test and latest_finetune and latest_finetune.get("status") == "complete"),
    ):
        bar = st.progress(0.0, text="Freezing evaluation configuration…")
        with st.status("Running final held-out evaluation…", expanded=True) as status:
            def update_test(event: dict) -> None:
                model_name = event.get("model_name", "Final evaluation")
                phase = event.get("phase", "working")
                model_index = int(event.get("model_index", 0))
                completed = int(event.get("completed", 0))
                total = max(1, int(event.get("total", 150)))
                if phase == "loading":
                    status.update(label=f"Loading {model_name}…", state="running")
                else:
                    fraction = min(1.0, (model_index + completed / total) / len(FINAL_MODEL_ORDER))
                    bar.progress(fraction, text=f"{model_name}: {completed}/{total} records")
                    if phase in {"complete", "partial"}:
                        status.write(f"{model_name}: {phase}.")

            try:
                outcome = run_final_test_evaluation(
                    dev_path=DEV_PATH,
                    test_path=TEST_PATH,
                    fine_tuning_run=latest_finetune,
                    results_root=RESULTS_ROOT,
                    reports_root=ROOT / "reports",
                    explicitly_confirmed=confirm_final_test,
                    progress_callback=update_test,
                )
                state = outcome["record"]["status"]
                status.update(label=f"Final test run {state}", state="complete" if state == "complete" else "error")
                if state == "complete":
                    st.rerun()
                else:
                    st.error("The held-out run was partial; inspect its saved errors before interpreting results.")
            except Exception as exc:
                status.update(label="Final test evaluation failed", state="error")
                st.exception(exc)
