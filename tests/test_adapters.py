from copy import deepcopy

from models.adapters import Decider08BAdapter, LayaBaseAdapter, LayaFineTunedAdapter
from models.tasks import TASK_LABELS


def fake_output(*, include_probabilities=True):
    answers = {}
    for task, labels in TASK_LABELS.items():
        answer = {"choice": labels[0]}
        if include_probabilities:
            answer["probabilities"] = {
                label: (1.0 if index == 0 else 0.0)
                for index, label in enumerate(labels)
            }
        answers[task] = answer
    return {"answers": answers}


class FakeLayaAgent:
    device = "cuda"
    dtype = "torch.float16"


class FakeLaya:
    def __init__(self, output):
        self.output = output

    def predict(self, state, questions, *, model):
        self.state = state
        self.questions = questions
        self.model = model
        return self.output

    def load(self, model):
        self.loaded_model = model
        return FakeLayaAgent()


class FakeDecider:
    def __init__(self, output):
        self.output = output

    def system_one(self, state, questions):
        self.state = state
        self.questions = questions
        return self.output


class FakeFineTunedLaya:
    device = "cuda"
    dtype = "torch.float16"

    def __init__(self, output):
        self.output = output

    def predict(self, state, questions):
        self.state = state
        self.questions = questions
        return self.output


def test_both_adapters_use_same_state_and_task_schema():
    message = "I was charged twice and would like my money back."
    laya_backend = FakeLaya(fake_output())
    decider_backend = FakeDecider(fake_output())

    laya_result = LayaBaseAdapter(laya_backend).predict(message)
    decider_result = Decider08BAdapter(decider_backend).predict(message)

    assert laya_backend.state == decider_backend.state == {"ticket": message}
    assert laya_backend.questions == decider_backend.questions
    assert laya_backend.model == "english"
    assert set(laya_result.predictions) == set(TASK_LABELS)
    assert set(decider_result.predictions) == set(TASK_LABELS)


def test_valid_probabilities_are_preserved():
    result = LayaBaseAdapter(FakeLaya(fake_output())).predict("Please help.")

    assert result.probabilities["department"] == {
        "billing": 1.0,
        "technical": 0.0,
        "sales": 0.0,
        "account": 0.0,
        "other": 0.0,
    }
    assert result.probability_notes == {}


def test_missing_probabilities_are_not_fabricated():
    result = Decider08BAdapter(
        FakeDecider(fake_output(include_probabilities=False))
    ).predict("Please help.")

    assert result.probabilities["refund"] is None
    assert "did not return" in result.probability_notes["refund"]


def test_invalid_probability_distribution_is_marked_unavailable():
    output = fake_output()
    output["answers"]["refund"]["probabilities"] = {
        "true": 0.2,
        "false": 0.2,
    }

    result = Decider08BAdapter(FakeDecider(output)).predict("Please help.")

    assert result.probabilities["refund"] is None
    assert "not a valid normalized" in result.probability_notes["refund"]


def test_finetuned_laya_adapter_uses_shared_task_schema_and_checkpoint_metadata():
    backend = FakeFineTunedLaya(fake_output())
    adapter = LayaFineTunedAdapter(
        backend,
        checkpoint="models/laya-finetuned/run-001",
    )

    result = adapter.predict("I cannot access my workspace settings.")

    assert backend.state == {"ticket": "I cannot access my workspace settings."}
    assert set(backend.questions) == set(TASK_LABELS)
    assert result.metadata["checkpoint"] == "models/laya-finetuned/run-001"
    assert result.metadata["device"] == "cuda"
