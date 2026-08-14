"""Prediction-Powered Inference (PPI) for a mean.

The problem: we have cheap judge scores ``f`` for *every* pool item, but expensive human
labels ``Y`` for only a few. We want a valid confidence interval on the true pass rate
``theta = E[Y]`` that is *tighter* than using the few labels alone.

Three estimators, for comparison:

* **naive**     — ``mean(f)`` over all items. Cheap but *biased*: if the judge is
                  systematically off, so is this estimate.
* **classical** — ``mean(Y)`` over the labeled items only. Unbiased but noisy (few labels).
* **ppi**       — corrects the judge's mean by its measured bias on the labeled set::

      theta_ppi = mean(f over all)  -  mean(f - Y over labeled)

  Unbiased like classical, but with variance ``var(f)/N + var(f - Y)/n``. When the judge
  correlates with the labels, ``var(f - Y)`` is small, so the interval is much tighter than
  classical for the same number of labels. (Angelopoulos et al., 2023.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from trustgate.models import Estimate


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pvar(xs: Sequence[float], mean: float) -> float:
    return sum((x - mean) ** 2 for x in xs) / len(xs) if xs else 0.0


def naive_mean(all_preds: Sequence[float], metric: str = "pass_rate") -> Estimate:
    """Judge-only mean. Biased; included as the 'ship and pray' baseline."""
    n = len(all_preds)
    mean = sum(all_preds) / n if n else 0.0
    se = math.sqrt(_pvar(all_preds, mean) / n) if n else 0.0
    return Estimate(metric=metric, mean=mean, ci_low=_clamp01(mean - 1.96 * se),
                    ci_high=_clamp01(mean + 1.96 * se), n=n, method="naive")


def classical_mean(labels: Sequence[float], metric: str = "pass_rate",
                   z: float = 1.96) -> Estimate:
    """Labels-only mean with a normal-approximation CI."""
    n = len(labels)
    if n == 0:
        return Estimate(metric=metric, mean=0.0, ci_low=0.0, ci_high=1.0, n=0, method="classical")
    mean = sum(labels) / n
    var = _pvar(labels, mean) * n / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return Estimate(metric=metric, mean=mean, ci_low=_clamp01(mean - z * se),
                    ci_high=_clamp01(mean + z * se), n=n, method="classical")


def ppi_mean(all_preds: Sequence[float], labeled_idx: Sequence[int],
             labels: Sequence[float], metric: str = "pass_rate", z: float = 1.96) -> Estimate:
    """PPI mean estimate + CI. ``labels[k]`` is the human label for item ``labeled_idx[k]``."""
    N = len(all_preds)
    n = len(labeled_idx)
    if n == 0 or N == 0:
        return naive_mean(all_preds, metric=metric)

    f_all_mean = sum(all_preds) / N
    rectifier = [all_preds[i] - labels[k] for k, i in enumerate(labeled_idx)]
    delta = sum(rectifier) / n
    theta = f_all_mean - delta

    var_f = _pvar(all_preds, f_all_mean)
    rect_mean = delta
    var_rect = (sum((r - rect_mean) ** 2 for r in rectifier) / (n - 1)) if n > 1 else 0.0
    se = math.sqrt(var_f / N + var_rect / n)

    return Estimate(metric=metric, mean=_clamp01(theta),
                    ci_low=_clamp01(theta - z * se), ci_high=_clamp01(theta + z * se),
                    n=n, method="ppi")


def ci_halfwidth(est: Estimate) -> float:
    return (est.ci_high - est.ci_low) / 2.0


__all__ = ["naive_mean", "classical_mean", "ppi_mean", "ci_halfwidth"]
