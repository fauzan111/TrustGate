"""LLMJudge — an Evaluator that turns any JudgeModel into per-item pass scores.

It plugs into the same pipeline as the deterministic evaluators, emitting a ``judge_pass``
metric. The decision threshold is configurable and is what the calibration step tunes.
"""

from __future__ import annotations

from trustgate.evaluators.base import Evaluator
from trustgate.judge.base import JudgeModel
from trustgate.models import Item, Output, Score, TaskType


def _question_answer_reference(item: Item, output: Output) -> tuple[str, str, str | None]:
    if item.task_type is TaskType.GENERATION:
        q = str(item.input)
        a = str(output.prediction)
        ref = None if item.references is None else str(
            item.references[0] if isinstance(item.references, list) else item.references)
        return q, a, ref
    if item.task_type is TaskType.RETRIEVAL:
        q = str(item.input.get("question", ""))
        pred = output.prediction if isinstance(output.prediction, dict) else {}
        a = str(pred.get("answer", ""))
        ref = str(item.references.get("answer", "")) if isinstance(item.references, dict) else None
        return q, a, ref
    return "", "", None


class LLMJudge(Evaluator):
    def __init__(self, judge: JudgeModel, threshold: float = 0.5) -> None:
        self.judge = judge
        self.threshold = threshold
        self.name = f"llm_judge[{judge.name}]"

    def score(self, item: Item, output: Output) -> list[Score]:
        if item.task_type not in (TaskType.GENERATION, TaskType.RETRIEVAL):
            return []
        q, a, ref = _question_answer_reference(item, output)
        s = self.judge.score(q, a, ref)
        return [Score(item_id=item.id, evaluator=self.name, metric="judge_pass",
                      value=s, passed=s >= self.threshold)]


__all__ = ["LLMJudge"]
