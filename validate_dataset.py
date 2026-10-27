from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluation.dataset import validate_datasets


def main() -> int:
    report = validate_datasets()

    for dataset in (report.dev, report.test):
        status = "PASS" if dataset.is_valid else "FAIL"
        print(f"\n[{status}] {dataset.role}: {dataset.path}")
        print(f"  Rows: {dataset.row_count}; expected: {dataset.expected_rows}")
        print(f"  Columns: {dataset.columns}")
        print(f"  Duplicate IDs: {dataset.duplicate_ticket_ids}")
        print(f"  Duplicate messages: {dataset.duplicate_messages}")
        print(
            "  refund_label remains text: "
            f"{dataset.refund_labels_remain_text}"
        )
        for issue in dataset.issues:
            print(f"  {issue.severity.upper()} {issue.code}: {issue.message}")

    print(f"\nCross-split overlap: {report.overlap}")
    for issue in report.issues:
        print(f"  {issue.severity.upper()} {issue.code}: {issue.message}")

    print(
        f"\nOverall: {'PASS' if report.is_valid else 'FAIL'} "
        f"({report.error_count} errors, {report.warning_count} warnings)"
    )
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())