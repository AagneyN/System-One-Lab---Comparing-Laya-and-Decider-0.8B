import csv
from pathlib import Path

from evaluation.dataset import (
    EXPECTED_COLUMNS,
    read_dataset,
    validate_csv,
    validate_datasets,
)


def row(
    ticket_id="T-1",
    message="Please help",
    department_label="billing",
    urgency_label="low",
    refund_label="false",
    difficulty="straightforward",
):
    return {
        "ticket_id": ticket_id,
        "message": message,
        "department_label": department_label,
        "urgency_label": urgency_label,
        "refund_label": refund_label,
        "difficulty": difficulty,
    }


def write_csv(path: Path, rows, columns=EXPECTED_COLUMNS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def codes(report):
    return {issue.code for issue in report.issues}


def test_real_datasets_have_expected_schema_and_counts():
    report = validate_datasets("data/tickets_dev.csv", "data/tickets_test.csv")

    assert report.dev.columns == list(EXPECTED_COLUMNS)
    assert report.test.columns == list(EXPECTED_COLUMNS)
    assert report.dev.row_count == 350
    assert report.test.row_count == 150
    assert report.dev.is_valid
    assert report.test.is_valid
    assert report.overlap == {"ticket_ids": 0, "normalized_messages": 0}


def test_column_order_is_checked(tmp_path):
    path = tmp_path / "wrong_order.csv"
    wrong_order = ("message",) + EXPECTED_COLUMNS[:1] + EXPECTED_COLUMNS[2:]
    write_csv(path, [row()], columns=wrong_order)

    report = validate_csv(path, role="dev", expected_rows=1)

    assert "column_order_mismatch" in codes(report)
    assert not report.is_valid


def test_allowed_label_values_are_checked(tmp_path):
    path = tmp_path / "invalid_label.csv"
    write_csv(path, [row(department_label="legal")])

    report = validate_csv(path, role="dev", expected_rows=1)

    assert report.invalid_label_counts["department_label"] == 1
    assert "invalid_label_values" in codes(report)


def test_missing_and_blank_values_are_checked(tmp_path):
    path = tmp_path / "blank.csv"
    write_csv(path, [row(message="   ")])

    report = validate_csv(path, role="dev", expected_rows=1)

    assert report.blank_counts["message"] == 1
    assert "blank_values" in codes(report)


def test_duplicate_ticket_ids_and_messages_are_detected(tmp_path):
    path = tmp_path / "duplicates.csv"
    write_csv(
        path,
        [
            row(ticket_id="T-1", message="Repeated message"),
            row(ticket_id="t-1", message="  REPEATED   MESSAGE "),
        ],
    )

    report = validate_csv(path, role="dev", expected_rows=2)

    assert report.duplicate_ticket_ids == 1
    assert report.duplicate_messages == 1
    assert "duplicate_ticket_ids" in codes(report)
    assert "duplicate_messages" in codes(report)


def test_development_test_overlap_is_detected(tmp_path):
    dev_path = tmp_path / "dev.csv"
    test_path = tmp_path / "test.csv"
    write_csv(dev_path, [row(ticket_id="T-1", message="Same ticket text")])
    write_csv(test_path, [row(ticket_id="t-1", message="same ticket text")])

    report = validate_datasets(dev_path, test_path)

    assert report.overlap == {"ticket_ids": 1, "normalized_messages": 1}
    assert not report.is_valid


def test_refund_labels_remain_text(tmp_path):
    path = tmp_path / "refund.csv"
    write_csv(path, [row(refund_label="true"), row(refund_label="false")])

    frame = read_dataset(path)
    report = validate_csv(path, role="dev", expected_rows=2)

    assert frame["refund_label"].tolist() == ["true", "false"]
    assert all(isinstance(value, str) for value in frame["refund_label"])
    assert report.refund_labels_remain_text is True