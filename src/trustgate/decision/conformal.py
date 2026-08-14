"""Risk-controlled release decisions via a confidence interval on the quality *drop*.

We frame shipping as a **non-inferiority test** around a single tolerance ``margin``.
Let ``delta = baseline_pass_rate - candidate_pass_rate`` (positive = a regression). We build
a ``(1 - alpha)`` confidence interval on ``delta`` and decide:

* **ship**        — ``ci_high(delta) <= margin``: confident the drop is within tolerance.
* **block**       — ``ci_low(delta)  >= margin``: confident the drop exceeds tolerance.
* **investigate** — the interval straddles ``margin``: collect more labels.

``alpha`` is the target error rate. With a valid interval, the probability of shipping a
true regression (delta >= margin) is bounded by ``alpha``. Two interval methods:

* ``normal``     — Wald interval on the difference of means (tight, asymptotic).
* ``hoeffding``  — distribution-free bound for bounded-[0,1] scores (conservative but with
                   a finite-sample guarantee — the "risk-controlled" flavor).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from trustgate.estimation.ppi import classical_mean
from trustgate.models import Decision, Verdict


@dataclass
class DiffEstimate:
    delta: float          # baseline_mean - candidate_mean
    ci_low: float
    ci_high: float
    method: str
    alpha: float
    n_candidate: int
    n_baseline: int


def _sample_var(xs: Sequence[float], mean: float) -> float:
    n = len(xs)
    return sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0


def difference_estimate(candidate: Sequence[float], baseline: Sequence[float],
                        alpha: float = 0.05, method: str = "normal") -> DiffEstimate:
    """CI on ``delta = mean(baseline) - mean(candidate)`` for per-item pass scores."""
    nc, nb = len(candidate), len(baseline)
    mc = sum(candidate) / nc if nc else 0.0
    mb = sum(baseline) / nb if nb else 0.0
    delta = mb - mc

    if nc == 0 or nb == 0:
        return DiffEstimate(delta, -1.0, 1.0, method, alpha, nc, nb)

    if method == "hoeffding":
        # Distribution-free: P(|dhat - delta| >= t) <= alpha for scores in [0, 1].
        hw = math.sqrt((1.0 / nc + 1.0 / nb) * math.log(2.0 / alpha) / 2.0)
    else:
        se = math.sqrt(_sample_var(candidate, mc) / nc + _sample_var(baseline, mb) / nb)
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        hw = z * se

    return DiffEstimate(delta, delta - hw, delta + hw, method, alpha, nc, nb)


def decide_from_diff(diff: DiffEstimate, margin: float = 0.05) -> Verdict:
    if diff.ci_high <= margin:
        return Verdict.SHIP
    if diff.ci_low >= margin:
        return Verdict.BLOCK
    return Verdict.INVESTIGATE


def release_decision(candidate: Sequence[float], baseline: Sequence[float],
                     margin: float = 0.05, alpha: float = 0.05,
                     method: str = "normal", metric: str = "pass_rate") -> Decision:
    """Full risk-controlled release decision from per-item pass scores."""
    diff = difference_estimate(candidate, baseline, alpha=alpha, method=method)
    verdict = decide_from_diff(diff, margin=margin)

    reasons = {
        Verdict.SHIP: f"drop CI upper {diff.ci_high:.3f} <= margin {margin}: within tolerance.",
        Verdict.BLOCK: f"drop CI lower {diff.ci_low:.3f} >= margin {margin}: regression.",
        Verdict.INVESTIGATE: f"drop CI [{diff.ci_low:.3f}, {diff.ci_high:.3f}] "
                             f"straddles margin {margin}: need more labels.",
    }
    return Decision(
        verdict=verdict, metric=metric,
        candidate=classical_mean(candidate, metric=metric),
        baseline=classical_mean(baseline, metric=metric),
        delta=diff.delta, delta_ci_low=diff.ci_low, delta_ci_high=diff.ci_high,
        margin=margin, alpha=alpha, method=method,
        reason=reasons[verdict] + f" (alpha={alpha}, method={method}).",
    )


__all__ = ["DiffEstimate", "difference_estimate", "decide_from_diff", "release_decision"]
