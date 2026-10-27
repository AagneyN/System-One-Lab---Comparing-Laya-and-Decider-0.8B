# System One Lab

System One Lab is a local Streamlit dashboard for reproducible support-ticket
classification benchmarks using Laya English Base, Decider 0.8B, and a separate
Laya checkpoint fine-tuned by this project.

## Current workflow

1. Validate `data/tickets_dev.csv` and `data/tickets_test.csv`.
2. Run Laya Base and Decider on development data; each run is saved append-only.
3. Fine-tune Laya using a deterministic ticket-level split of development data:
   245 training tickets, 35 calibration tickets, and 70 validation tickets.
4. Freeze the model/checkpoint, shared task schema, calibration, and device settings.
5. Run the explicitly confirmed, one-shot evaluation of all three models on the
   same 150 held-out test records.

The original CSV files and pinned Laya base checkpoint are preserved. Development
and fine-tuning runs do not use test labels. If final test results are used to
change models or settings, the test set is consumed for development; a new
untouched test set is then required for a new final performance claim.

## Requirements

- Windows 10/11, Python 3.12, and an NVIDIA driver compatible with the pinned
  CUDA-enabled PyTorch wheel for GPU training/inference.
- The two validated CSV files in `data/`.
- Approximately 6 GB of GPU memory is supported for inference and head-only Laya
  fine-tuning. Full encoder fine-tuning is not enabled by this project profile.
- Internet access is needed only for package/model downloads when they are not
  already cached.

## Setup in PowerShell

Open the project folder in VS Code, then open a PowerShell terminal in the
project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependencies are installed in `.venv`, not globally. If PowerShell blocks
virtual-environment activation, invoke the `.venv` Python directly as shown
below; changing the system execution policy is unnecessary.

## Validate, test, and launch

```powershell
.\.venv\Scripts\python.exe validate_dataset.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local URL, usually `http://localhost:8501`.

## Dashboard workflow

- **Overview & Data:** inspect dataset validation and development-only statistics.
- **Benchmark Runner:** run the Laya Base and Decider development comparison.
- **Fine-Tuning Lab:** start the verified Laya trainer, inspect its training log,
  development calibration, validation comparison, and checkpoint verification.
- **Development Results:** inspect saved baseline metrics and ticket predictions.
- **Final Test Analysis:** after fine-tuning and config freeze, check the explicit
  confirmation box and run all three models once on the held-out test set.

The trainer uses the official `laya.train.finetune` workflow with `RLCD`, a fixed
four-epoch budget, and the official `freeze_encoder` option for constrained GPU
memory. It does not expose early stopping; the independent validation split is
reported after the fixed training run. Per-model, per-task probability
temperature calibrators are fitted only on the separate development calibration
split and saved with the fine-tuning run.

## Saved artifacts

- `results/development/<run_id>/run.json` — baseline predictions, metrics,
  configuration, dataset hash, latency, resource readings, warnings, and errors.
- `results/fine_tuning/<run_id>/run.json` — split, training summary, validation
  analysis, calibrators, checkpoint verification, and package/hardware metadata.
- `results/fine_tuning_artifacts/<run_id>/` — actual split inputs, training log,
  calibration predictions, validation predictions, and calibrator JSON.
- `models/laya-finetuned/<run_id>/` — distinct, loadable checkpoint; the original
  Laya base checkpoint is not overwritten.
- `results/final_test/<run_id>/run.json` — all three test predictions, raw and
  calibrated probability metrics, per-class metrics, confusion matrices,
  difficulty analysis, paired bootstrap intervals, McNemar comparisons, and
  measured runtime/resource data.
- `reports/final_test/<run_id>.md` — Markdown summary of the frozen final run.

Run folders use unique IDs. Previous runs are retained; repeated final-test runs
with the same frozen configuration are rejected unless a technical repeat reason
is supplied.

## Hardware and interpretation notes

Models run sequentially so their weights are not intentionally held in GPU memory
at the same time. Decider uses its supported eager execution path on this Windows
setup because its CUDA-graph path requires Triton. The fine-tuned Laya adapter
loads the saved local checkpoint through Laya's public loader. The original Laya
and Decider checkpoints have different pretrained histories.

The development set has 350 tickets and the held-out test set has 150. These
sample sizes limit the reliability of rare-class, difficulty-subgroup, and small
point-estimate differences. Read confidence intervals and calibration metrics
alongside accuracy; do not treat a small observed lead as proof of superiority.
