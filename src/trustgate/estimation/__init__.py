"""Estimation — turn per-item scores into an aggregate metric + uncertainty.

The walking skeleton uses a Wilson score interval for a proportion (accuracy), which is
well-behaved for small samples and needs no third-party dependency. Milestone 7-8 swaps
in Prediction-Powered Inference (PPI) for label-efficient calibrated estimates.
"""

from __future__ import annotations

import math

from trustgate.models import Estimate, Score


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    ``z=1.96`` ≈ 95% confidence. Returns (low, high), clamped to [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def estimate_accuracy(scores: list[Score], metric: str = "accuracy") -> Estimate:
    """Aggregate binary pass/fail scores into a calibrated-free point estimate + CI."""
    relevant = [s for s in scores if s.metric == metric]
    n = len(relevant)
    successes = sum(1 for s in relevant if s.value >= 0.5)
    mean = successes / n if n else 0.0
    low, high = wilson_interval(successes, n)
    return Estimate(metric=metric, mean=mean, ci_low=low, ci_high=high, n=n, method="wilson")


def estimate_mean(scores: list[Score], metric: str, z: float = 1.96) -> Estimate:
    """Normal-approximation mean + CI for a continuous metric in [0, 1].

    Used for retrieval metrics (precision, nDCG, groundedness) that are not binary.
    """
    values = [s.value for s in scores if s.metric == metric]
    n = len(values)
    if n == 0:
        return Estimate(metric=metric, mean=0.0, ci_low=0.0, ci_high=1.0, n=0, method="normal")
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    return Estimate(
        metric=metric, mean=mean,
        ci_low=max(0.0, mean - z * se), ci_high=min(1.0, mean + z * se),
        n=n, method="normal",
    )


def aggregate(scores: list[Score]) -> dict[str, Estimate]:
    """Estimate every metric present in ``scores`` (normal-approx CIs)."""
    metrics = sorted({s.metric for s in scores})
    return {m: estimate_mean(scores, m) for m in metrics}


from trustgate.estimation.ppi import (  # noqa: E402
    ci_halfwidth,
    classical_mean,
    naive_mean,
    ppi_mean,
)

__all__ = [
    "wilson_interval", "estimate_accuracy", "estimate_mean", "aggregate",
    "naive_mean", "classical_mean", "ppi_mean", "ci_halfwidth",
]
