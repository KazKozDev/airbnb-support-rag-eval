"""The judge parser must survive reasoning-model output, not just clean JSON.

Regression guard for the bug where ~85% of judgments were silently dropped because
a reasoning judge wrapped its verdict in prose / thinking and the JSON failed to
parse."""
from eval.run_eval import _parse_judge


def test_plain_json():
    out = _parse_judge('{"faithfulness": 5, "correctness": 4, "rationale": "ok"}')
    assert out["faithfulness"] == 5 and out["correctness"] == 4


def test_fenced_json():
    out = _parse_judge('```json\n{"faithfulness": 3, "correctness": 2}\n```')
    assert out["faithfulness"] == 3 and out["correctness"] == 2


def test_json_after_reasoning_prose():
    text = ('Let me think. The answer is grounded but omits a detail, so I will '
            'score it. {"faithfulness": 5, "correctness": 3, "rationale": "partial"}')
    out = _parse_judge(text)
    assert out["faithfulness"] == 5 and out["correctness"] == 3


def test_unparseable_returns_none_scores():
    out = _parse_judge("I could not decide on a score for this one.")
    assert out["faithfulness"] is None and out["correctness"] is None
