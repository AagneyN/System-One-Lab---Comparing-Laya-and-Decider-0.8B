"""Read-only validation for the System One Lab ticket CSVs."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = (
    "ticket_id",
    "message",
    "department_label",
    "urgency_label",
    "refund_label",
    "difficulty",
)

ALLOWED_LABELS = {
    "department_label": {"billing", "technical", "sales", "account", "other"},
    "urgency_label": {"low", "medium", "high"},
    "refund_label": {"true", "false"},
    "difficulty": {
        "straightforward",
        "ambiguous",
        "multi_issue",
        "misleading",
        "out_of_scope",
        "incomplete",
    },
}

EXPECTED_ROWS = {"dev": 350, "test": 150}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    count: int = 1


@dataclass
class DatasetReport:
    path: str
    role: str
    expected_rows: int | None
    exists: bool = False
    row_count: int | None = None
    columns: list[str] = field(default_factory=list)
    blank_counts: dict[str, int] = field(default_factory=dict)
    invalid_label_counts: dict[str, int] = field(default_factory=dict)
    non_normalized_label_counts: dict[str, int] = field(default_factory=dict)
    duplicate_ticket_ids: int = 0
    duplicate_messages: int = 0
    refund_labels_remain_text: bool | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    # Kept in memory for development/test overlap checks. Excluded from to_dict().
    _records: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def error_count(self) -> int:
        return sum(i.count for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(i.count for i in self.issues if i.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return self.exists and self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "expected_rows": self.expected_rows,
            "exists": self.exists,
            "row_count": self.row_count,
            "columns": self.columns,
            "blank_counts": self.blank_counts,
            "invalid_label_counts": self.invalid_label_counts,
            "non_normalized_label_counts": self.non_normalized_label_counts,
            "duplicate_ticket_ids": self.duplicate_ticket_ids,
            "duplicate_messages": self.duplicate_messages,
            "refund_labels_remain_text": self.refund_labels_remain_text,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "is_valid": self.is_valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass
class PairValidationReport:
    dev: DatasetReport
    test: DatasetReport
    overlap: dict[str, int]
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.dev.is_valid
            and self.test.is_valid
            and not any(i.severity == "error" for i in self.issues)
        )

    @property
    def error_count(self) -> int:
        return (
            self.dev.error_count
            + self.test.error_count
            + sum(i.count for i in self.issues if i.severity == "error")
        )

    @property
    def warning_count(self) -> int:
        return (
            self.dev.warning_count
            + self.test.warning_count
            + sum(i.count for i in self.issues if i.severity == "warning")
        )


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read every CSV field as text; do not infer booleans or normalize values."""
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def _read_raw_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle, strict=True), [])


def _normalize_message(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _duplicate_count(values: list[str]) -> int:
    nonempty = [value for value in values if value]
    return int(pd.Series(nonempty, dtype="string").duplicated(keep="first").sum())


def _add(
    report: DatasetReport,
    code: str,
    severity: str,
    message: str,
    count: int = 1,
) -> None:
    report.issues.append(
        ValidationIssue(code=code, severity=severity, message=message, count=count)
    )


def validate_csv(
    path: str | Path,
    *,
    role: str,
    expected_rows: int | None = None,
) -> DatasetReport:
    """Validate one CSV without changing it."""
    csv_path = Path(path)
    expected = (
        expected_rows
        if expected_rows is not None
        else EXPECTED_ROWS.get(role)
    )
    report = DatasetReport(
        path=str(csv_path),
        role=role,
        expected_rows=expected,
        exists=csv_path.is_file(),
    )

    if not report.exists:
        _add(report, "file_missing", "error", f"File does not exist: {csv_path}")
        return report

    try:
        report.columns = _read_raw_header(csv_path)
        if not report.columns:
            _add(report, "empty_file", "error", "The CSV is empty.")
            return report

        if len(set(report.columns)) != len(report.columns):
            _add(
                report,
                "duplicate_column_names",
                "error",
                "The CSV header contains duplicate column names.",
            )

        missing = [c for c in EXPECTED_COLUMNS if c not in report.columns]
        unexpected = [c for c in report.columns if c not in EXPECTED_COLUMNS]

        if missing:
            _add(
                report,
                "missing_columns",
                "error",
                "Missing required columns: " + ", ".join(missing),
            )
        if unexpected:
            _add(
                report,
                "unexpected_columns",
                "error",
                "Unexpected columns: " + ", ".join(unexpected),
            )
        if tuple(report.columns) != EXPECTED_COLUMNS:
            _add(
                report,
                "column_order_mismatch",
                "error",
                "Columns must appear in this order: "
                + ", ".join(EXPECTED_COLUMNS),
            )

        frame = read_dataset(csv_path)
        report.row_count = len(frame)
        report._records = frame.to_dict(orient="records")

        # CSV input is intentionally loaded as text so true/false cannot be
        # silently converted to Python booleans by pandas.
        if "refund_label" in frame.columns:
            report.refund_labels_remain_text = all(
                isinstance(value, str) for value in frame["refund_label"].tolist()
            )
            if not report.refund_labels_remain_text:
                _add(
                    report,
                    "refund_label_type_changed",
                    "error",
                    "refund_label contains non-text values after CSV parsing.",
                )

        # Every nonempty cell should remain text under read_dataset().
        for column in frame.columns:
            non_text = sum(
                not isinstance(value, str) for value in frame[column].tolist()
            )
            if non_text:
                _add(
                    report,
                    "non_text_values",
                    "error",
                    f"{column} contains {non_text} non-text values.",
                    count=non_text,
                )

        for column in EXPECTED_COLUMNS:
            if column not in frame.columns:
                continue

            values = frame[column].fillna("").astype(str)
            blank_count = int(values.str.strip().eq("").sum())
            if blank_count:
                report.blank_counts[column] = blank_count
                _add(
                    report,
                    "blank_values",
                    "error",
                    f"{column} contains {blank_count} blank values.",
                    count=blank_count,
                )

            allowed = ALLOWED_LABELS.get(column)
            if allowed is None:
                continue

            nonblank = values[values.str.strip().ne("")]
            invalid_count = int((~nonblank.isin(allowed)).sum())
            if invalid_count:
                report.invalid_label_counts[column] = invalid_count
                _add(
                    report,
                    "invalid_label_values",
                    "error",
                    f"{column} contains {invalid_count} values outside its allowed labels.",
                    count=invalid_count,
                )

            normalized_count = int(
                (nonblank != nonblank.str.strip().str.lower()).sum()
            )
            if normalized_count:
                report.non_normalized_label_counts[column] = normalized_count
                _add(
                    report,
                    "labels_not_normalized",
                    "error",
                    f"{column} contains {normalized_count} labels that are not trimmed lowercase text.",
                    count=normalized_count,
                )

        if "ticket_id" in frame.columns:
            ids = frame["ticket_id"].map(lambda x: str(x).strip().casefold())
            report.duplicate_ticket_ids = _duplicate_count(ids.tolist())
            if report.duplicate_ticket_ids:
                _add(
                    report,
                    "duplicate_ticket_ids",
                    "error",
                    f"Found {report.duplicate_ticket_ids} duplicate ticket IDs.",
                    count=report.duplicate_ticket_ids,
                )

        if "message" in frame.columns:
            messages = frame["message"].map(_normalize_message)
            report.duplicate_messages = _duplicate_count(messages.tolist())
            if report.duplicate_messages:
                _add(
                    report,
                    "duplicate_messages",
                    "warning",
                    f"Found {report.duplicate_messages} duplicate normalized messages.",
                    count=report.duplicate_messages,
                )

        if expected is not None and report.row_count != expected:
            _add(
                report,
                "row_count_differs",
                "warning",
                f"Expected {expected} rows; found {report.row_count}.",
            )

    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        _add(
            report,
            "csv_read_error",
            "error",
            f"Could not read CSV: {type(exc).__name__}: {exc}",
        )

    return report


def validate_datasets(
    dev_path: str | Path = "data/tickets_dev.csv",
    test_path: str | Path = "data/tickets_test.csv",
) -> PairValidationReport:
    """Validate both splits and detect ID or normalized-message overlap."""
    dev = validate_csv(dev_path, role="dev")
    test = validate_csv(test_path, role="test")

    dev_ids = {
        str(row.get("ticket_id", "")).strip().casefold()
        for row in dev._records
        if str(row.get("ticket_id", "")).strip()
    }
    test_ids = {
        str(row.get("ticket_id", "")).strip().casefold()
        for row in test._records
        if str(row.get("ticket_id", "")).strip()
    }
    dev_messages = {
        _normalize_message(row.get("message", ""))
        for row in dev._records
        if _normalize_message(row.get("message", ""))
    }
    test_messages = {
        _normalize_message(row.get("message", ""))
        for row in test._records
        if _normalize_message(row.get("message", ""))
    }

    overlap = {
        "ticket_ids": len(dev_ids & test_ids),
        "normalized_messages": len(dev_messages & test_messages),
    }
    issues: list[ValidationIssue] = []

    if overlap["ticket_ids"]:
        issues.append(
            ValidationIssue(
                "dev_test_ticket_id_overlap",
                "error",
                f"Development and test share {overlap['ticket_ids']} ticket IDs.",
                overlap["ticket_ids"],
            )
        )
    if overlap["normalized_messages"]:
        issues.append(
            ValidationIssue(
                "dev_test_message_overlap",
                "error",
                f"Development and test share {overlap['normalized_messages']} normalized messages.",
                overlap["normalized_messages"],
            )
        )

    return PairValidationReport(dev, test, overlap, issues)