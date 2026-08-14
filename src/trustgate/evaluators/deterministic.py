"""Deterministic evaluators — cheap, exact, no model calls."""

from __future__ import annotations

from trustgate.evaluators.base import Evaluator
from trustgate.models import Item, Output, Score


def _normalise(value: object) -> str:
    return str(value).strip().casefold()


class ExactMatch(Evaluator):
    """1.0 if the prediction exactly matches the reference (case/space-insensitive).

    ``references`` may be a single value or a list of accepted answers.
    """

    name = "exact_match"

    def score(self, item: Item, output: Output) -> list[Score]:
        refs = item.references
        accepted = refs if isinstance(refs, list) else [refs]
        pred = _normalise(output.prediction)
        passed = any(_normalise(r) == pred for r in accepted)
        return [
            Score(
                item_id=item.id,
                evaluator=self.name,
                metric="accuracy",
                value=1.0 if passed else 0.0,
                passed=passed,
            )
        ]
