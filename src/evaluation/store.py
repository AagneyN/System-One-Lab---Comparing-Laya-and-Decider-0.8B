"""Append-only JSON result storage with safeguards for repeated test runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


RUN_KINDS = {"development", "fine_tuning", "final_test"}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def _json_default(value: Any):
    """Serialize common filesystem/numeric metadata without masking other errors."""
    if isinstance(value, Path):
        return str(value)
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return scalar()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


class RunStore:
    def __init__(self, root: str | Path = "results"):
        self.root = Path(root)

    def _matching_test_fingerprint(self, fingerprint: str) -> list[Path]:
        final_root = self.root / "final_test"
        if not final_root.exists():
            return []

        matches = []
        for manifest in final_root.glob("*/run.json"):
            try:
                record = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot check prior final test run {manifest}: {exc}"
                ) from exc
            if record.get("experiment_fingerprint") == fingerprint:
                matches.append(manifest.parent)
        return matches

    def save_run(
        self,
        kind: str,
        run_id: str,
        payload: Mapping[str, Any],
        *,
        test_evaluation_confirmed: bool = False,
        repeat_reason: str | None = None,
    ) -> Path:
        if kind not in RUN_KINDS:
            raise ValueError(f"kind must be one of {sorted(RUN_KINDS)}")
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id contains unsafe characters or is too long.")

        data = dict(payload)
        fingerprint = data.get("experiment_fingerprint")

        if kind == "final_test":
            if test_evaluation_confirmed is not True:
                raise ValueError("Final test runs require explicit confirmation.")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("Final test runs require an experiment_fingerprint.")

            previous = self._matching_test_fingerprint(fingerprint)
            if previous and not (repeat_reason and repeat_reason.strip()):
                raise ValueError(
                    "A final test run with this fingerprint already exists. "
                    "Provide a technical repeat_reason to preserve and repeat it."
                )

        # Validate JSON before creating the run directory.
        serialized = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )

        kind_root = self.root / kind
        kind_root.mkdir(parents=True, exist_ok=True)
        run_directory = kind_root / run_id
        run_directory.mkdir(exist_ok=False)

        record = {
            "run_id": run_id,
            "kind": kind,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_fingerprint": fingerprint,
            "repeat_reason": repeat_reason,
            "payload": json.loads(serialized),
        }

        manifest = run_directory / "run.json"
        with manifest.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                record,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")

        return run_directory
