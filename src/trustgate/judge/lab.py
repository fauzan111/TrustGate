"""Judge Lab — experiments that measure whether a judge can be trusted.

Every function returns plain numbers so they can be logged, asserted in tests, and put in
the decision report. The probes are deliberately constructed so an *unbiased* judge scores
near the neutral baseline and a *biased* judge deviates measurably.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from trustgate.judge.base import JudgeModel
from trustgate.stats import balanced_accuracy, cohen_kappa


# --------------------------------------------------------------------------- #
# Reliability: repeated-trial self-consistency
# --------------------------------------------------------------------------- #
@dataclass
class ConsistencyResult:
    score_std: float      # std of the continuous score across trials
    flip_rate: float      # fraction of trials whose binary pass differs from the majority


def self_consistency(judge: JudgeModel, question: str, answer: str,
                     reference: str | None = None, trials: int = 8,
                     threshold: float = 0.5) -> ConsistencyResult:
    scores = [judge.score(question, answer, reference, trial=t) for t in range(trials)]
    passes = [1 if s >= threshold else 0 for s in scores]
    majority = 1 if sum(passes) * 2 >= trials else 0
    flip_rate = sum(1 for p in passes if p != majority) / trials
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return ConsistencyResult(score_std=std, flip_rate=flip_rate)


# --------------------------------------------------------------------------- #
# Position bias: does swapping A/B order change the winner?
# --------------------------------------------------------------------------- #
def position_swap_rate(judge: JudgeModel, pairs: Sequence[tuple[str, str, str]]) -> float:
    """Fraction of (question, x, y) pairs whose *content* winner flips when order swaps.

    An order-invariant judge yields 0; a position-biased judge that always favors the
    first slot flips on every pair where x != y, approaching 1.
    """
    flips = 0
    total = 0
    for question, x, y in pairs:
        total += 1
        w1 = x if judge.compare(question, x, y) == "a" else y      # order (x, y)
        w2 = x if judge.compare(question, y, x) == "b" else y      # order (y, x)
        if w1 != w2:
            flips += 1
    return flips / total if total else 0.0


# --------------------------------------------------------------------------- #
# Length bias: among equally-correct answers, is the longer one preferred?
# --------------------------------------------------------------------------- #
def length_bias_rate(judge: JudgeModel,
                     pairs: Sequence[tuple[str, str, str]]) -> float:
    """Given (question, short, long) with both answers correct, fraction where the judge
    prefers the longer one. ~0.5 = unbiased; ->1.0 = strong length bias."""
    prefer_long = 0
    total = 0
    for question, short, long in pairs:
        total += 1
        # Average over both presentation orders to remove position as a confound.
        w_order1 = long if judge.compare(question, short, long) == "b" else short
        w_order2 = long if judge.compare(question, long, short) == "a" else short
        prefer_long += (w_order1 == long) + (w_order2 == long)
    return prefer_long / (2 * total) if total else 0.0


# --------------------------------------------------------------------------- #
# Judge <-> human agreement
# --------------------------------------------------------------------------- #
@dataclass
class AgreementResult:
    balanced_accuracy: float
    cohen_kappa: float
    n: int


def judge_human_agreement(judge_pass: Sequence[int], human_pass: Sequence[int]) -> AgreementResult:
    return AgreementResult(
        balanced_accuracy=balanced_accuracy(human_pass, judge_pass),
        cohen_kappa=cohen_kappa(human_pass, judge_pass),
        n=len(human_pass),
    )


# --------------------------------------------------------------------------- #
# Bias correction: tune the score->pass threshold against human labels
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    best_threshold: float
    agreement_before: float   # balanced accuracy at the default 0.5 threshold
    agreement_after: float    # balanced accuracy at the tuned threshold


def calibrate_threshold(judge_scores: Sequence[float], human_pass: Sequence[int],
                        default: float = 0.5) -> CalibrationResult:
    """Pick the threshold that maximizes judge<->human balanced accuracy.

    A simple, honest first calibrator; isotonic/Platt + PPI replace it in the sampling
    milestone. Never does worse than the default threshold.
    """
    def agree(th: float) -> float:
        pred = [1 if s >= th else 0 for s in judge_scores]
        return balanced_accuracy(human_pass, pred)

    before = agree(default)
    candidates = [i / 100 for i in range(0, 101, 5)]
    best_th, best_agree = default, before
    for th in candidates:
        a = agree(th)
        if a > best_agree:
            best_th, best_agree = th, a
    return CalibrationResult(best_threshold=best_th, agreement_before=before,
                             agreement_after=best_agree)


__all__ = [
    "ConsistencyResult", "self_consistency",
    "position_swap_rate", "length_bias_rate",
    "AgreementResult", "judge_human_agreement",
    "CalibrationResult", "calibrate_threshold",
]
