"""Canonical, hashable experiment configurations and dataset hashing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CALIBRATION_ROLES = {
    "development_train",
    "development_calibration",
    "development_validation",
}


@dataclass(frozen=True)
class FrozenConfig:
    canonical_json: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 identity for a dataset or other input file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_config(config: Mapping[str, Any]) -> FrozenConfig:
    """Validate leakage-related fields and produce a stable configuration hash."""
    payload = json.loads(
        json.dumps(config, sort_keys=True, allow_nan=False)
    )

    phase = payload.get("phase")
    allowed_phases = {"development", "fine_tuning", "final_test"}
    if phase not in allowed_phases:
        raise ValueError(f"phase must be one of {sorted(allowed_phases)}")

    dataset_hashes = payload.get("dataset_hashes", {})
    if not isinstance(dataset_hashes, dict):
        raise ValueError("dataset_hashes must be an object keyed by dataset role.")

    for role, digest in dataset_hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"dataset_hashes[{role!r}] must be a SHA-256 hex digest.")

    if phase in {"development", "fine_tuning"} and "test" in dataset_hashes:
        raise ValueError(
            f"{phase} configurations cannot include the held-out test dataset."
        )

    if phase == "fine_tuning" and payload.get("training_dataset_role") != "dev":
        raise ValueError("Fine-tuning must use the development dataset only.")

    calibration_role = payload.get("calibration_fit_role")
    if calibration_role is not None and calibration_role not in CALIBRATION_ROLES:
        raise ValueError(
            "Calibration can be fit only on development_train or "
            "development_validation."
        )

    if phase == "final_test" and "test" not in dataset_hashes:
        raise ValueError("A final_test configuration must identify the test dataset.")

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FrozenConfig(canonical_json=canonical, sha256=digest)
