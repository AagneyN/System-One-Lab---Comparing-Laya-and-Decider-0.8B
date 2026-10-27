import pandas as pd

from finetuning.runner import _rows_for_laya, _split_development
from models.tasks import TASK_QUESTIONS


def _frame():
    return pd.DataFrame(
        {
            "ticket_id": [f"T{i:03d}" for i in range(350)],
            "message": [f"Ticket message {i}" for i in range(350)],
            "department_label": ["billing", "technical", "sales", "account", "other"] * 70,
            "urgency_label": ["low", "medium", "high", "low", "medium"] * 70,
            "refund_label": ["true"] * 43 + ["false"] * 307,
            "difficulty": ["straightforward", "ambiguous", "multi_issue", "misleading", "out_of_scope", "incomplete"] * 58 + ["straightforward"] * 2,
        }
    )


def test_development_split_is_reproducible_ticket_disjoint_and_complete():
    frame = _frame()

    first = _split_development(frame, seed=123)
    second = _split_development(frame, seed=123)

    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 245,
        "calibration": 35,
        "validation": 70,
    }
    groups = [set(value) for value in first.values()]
    assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert set.union(*groups) == set(range(350))


def test_laya_training_rows_use_message_and_shared_schema_only():
    frame = _frame()
    row = _rows_for_laya(frame, [0])[0]

    assert row["state"] == {"ticket": "Ticket message 0"}
    assert row["questions"] == TASK_QUESTIONS
    assert row["expected"] == {
        "department": "billing",
        "urgency": "low",
        "refund": "true",
    }
    assert "difficulty" not in row["state"]


def test_development_split_keeps_refund_positive_records_in_each_partition():
    frame = _frame()
    splits = _split_development(frame, seed=20261027)

    for indices in splits.values():
        assert set(frame.iloc[indices]["refund_label"]) == {"true", "false"}
