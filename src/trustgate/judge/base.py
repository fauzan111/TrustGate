"""The JudgeModel contract.

Any judge — a simulated one with known biases, a local Ollama model, or a hosted API —
implements these two methods so the Judge Lab experiments and the LLMJudge evaluator work
against all of them identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

Winner = Literal["a", "b"]


class JudgeModel(ABC):
    name: str

    @abstractmethod
    def score(self, question: str, answer: str, reference: str | None = None,
              *, trial: int = 0) -> float:
        """Pointwise quality score for one answer, in [0, 1].

        ``trial`` varies the internal seed so repeated calls can differ (used to measure
        self-consistency).
        """
        raise NotImplementedError

    @abstractmethod
    def compare(self, question: str, answer_a: str, answer_b: str, *, trial: int = 0) -> Winner:
        """Pairwise preference: return "a" or "b" for the better answer."""
        raise NotImplementedError


__all__ = ["JudgeModel", "Winner"]
