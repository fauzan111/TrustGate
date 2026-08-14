"""The Evaluator contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trustgate.models import Item, Output, Score


class Evaluator(ABC):
    name: str

    @abstractmethod
    def score(self, item: Item, output: Output) -> list[Score]:  # pragma: no cover
        """Score one output against one item, returning one or more metrics."""
        raise NotImplementedError
