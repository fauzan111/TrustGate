"""A simulated judge with *known* biases.

This is the workhorse for developing and testing the Judge Lab. Because we set the biases
ourselves, we can prove that each probe detects the bias it targets and that bias
correction removes it — something a real, opaque model can never let us verify.

Biases you can dial in:
  * ``length_bias``   — preference for longer answers (the classic LLM-judge failure).
  * ``position_bias`` — preference for whichever answer is shown first (position "a").
  * ``noise``         — per-trial randomness, so repeated calls vary (self-consistency).
  * ``base_correct`` / ``base_wrong`` — scores anchored on true correctness.
"""

from __future__ import annotations

import hashlib

from trustgate.judge.base import JudgeModel, Winner


def _unit(*parts: object) -> float:
    """Deterministic pseudo-random value in [-0.5, 0.5) from arbitrary parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF - 0.5


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_len(text: str, scale: int = 200) -> float:
    return min(len(text) / scale, 1.0)


def _is_correct(answer: str, reference: str | None) -> bool:
    if reference is None:
        return False
    a, r = " ".join(answer.lower().split()), " ".join(reference.lower().split())
    return bool(a) and (a == r or (len(r) > 3 and r in a))


class SimulatedJudge(JudgeModel):
    def __init__(
        self,
        name: str = "sim-judge",
        *,
        length_bias: float = 0.0,
        position_bias: float = 0.0,
        noise: float = 0.0,
        base_correct: float = 0.8,
        base_wrong: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.length_bias = length_bias
        self.position_bias = position_bias
        self.noise = noise
        self.base_correct = base_correct
        self.base_wrong = base_wrong
        self.seed = seed

    def _utility(self, question: str, answer: str, reference: str | None, trial: int) -> float:
        base = self.base_correct if _is_correct(answer, reference) else self.base_wrong
        length_term = self.length_bias * _norm_len(answer)
        noise_term = self.noise * _unit(self.seed, question, answer, trial)
        return base + length_term + noise_term

    def score(self, question: str, answer: str, reference: str | None = None,
              *, trial: int = 0) -> float:
        return _clamp(self._utility(question, answer, reference, trial))

    def compare(self, question: str, answer_a: str, answer_b: str, *, trial: int = 0) -> Winner:
        # Pairwise correctness is unknown to the judge here (no reference given), so utility
        # is driven by length + position + noise — exactly the confounds we want to probe.
        # Noise is a function of *content* only (not the slot), so absent real bias the
        # decision is order-invariant and the position-swap probe reads ~0.
        ua = (self.length_bias * _norm_len(answer_a)
              + self.position_bias
              + self.noise * _unit(self.seed, question, answer_a, trial))
        ub = (self.length_bias * _norm_len(answer_b)
              + self.noise * _unit(self.seed, question, answer_b, trial))
        return "a" if ua >= ub else "b"


__all__ = ["SimulatedJudge"]
