"""The label-efficiency experiment — TrustGate's headline result.

Question: *how few human labels does a calibrated estimator need to measure the pass rate
to a target precision?* We answer it with a controlled simulation study: generate a pool
where we know every true label, then repeatedly reveal only a small labeled subset and
compare estimators + sampling strategies on

  * **CI half-width** (precision), and
  * **coverage** (does the 95% CI actually contain the true value — i.e. is it *valid*?).

Because the judge/label relationship is set by us, the benefit of PPI + uncertainty
sampling is measured, not assumed. Numbers here are simulation planning estimates; the same
machinery runs on real judge scores + human labels once the labeling sheet is filled.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from trustgate.estimation.ppi import ci_halfwidth, classical_mean, ppi_mean
from trustgate.sampling import sample_random, sample_stratified, sample_uncertainty


def synthetic_pool(n: int = 2000, true_rate: float = 0.8, separation: float = 0.5,
                   noise: float = 0.3, seed: int = 0) -> tuple[list[float], list[int], float]:
    """Build a pool of (judge_score, true_label).

    ``separation`` controls how well the judge score separates pass/fail (judge quality);
    ``noise`` adds spread. Returns (preds, labels, true_theta).
    """
    rng = random.Random(seed)
    preds: list[float] = []
    labels: list[int] = []
    for _ in range(n):
        y = 1 if rng.random() < true_rate else 0
        center = 0.5 + separation * (0.5 if y else -0.5)
        f = center + noise * (rng.random() - 0.5) * 2
        preds.append(max(0.0, min(1.0, f)))
        labels.append(y)
    return preds, labels, sum(labels) / n


@dataclass
class EfficiencyRow:
    strategy: str
    estimator: str
    budget: int
    mean_ci_halfwidth: float
    coverage: float           # fraction of repeats whose CI contains true_theta
    mean_abs_error: float
    repeats: int


def _score_bins(preds: list[float], bins: int = 5) -> list[int]:
    return [min(bins - 1, int(p * bins)) for p in preds]


def run_label_efficiency(preds: list[float], labels: list[int], true_theta: float,
                         budgets: list[int], repeats: int = 200,
                         seed: int = 0) -> list[EfficiencyRow]:
    """Compare {random, uncertainty, stratified} x {classical, ppi} across label budgets."""
    strata = _score_bins(preds)
    combos = [
        ("random", "classical"), ("random", "ppi"),
        ("uncertainty", "ppi"), ("stratified", "ppi"),
    ]
    rows: list[EfficiencyRow] = []

    for strategy, estimator in combos:
        for budget in budgets:
            halfwidths: list[float] = []
            covered = 0
            abs_errors: list[float] = []
            for r in range(repeats):
                if strategy == "random":
                    idx = sample_random(len(preds), budget, seed=seed + r)
                elif strategy == "uncertainty":
                    # Draw from the most-uncertain candidates. Note: this makes the labeled
                    # set non-uniform, which can *invalidate* PPI's coverage — the coverage
                    # column is exactly how we detect that failure mode.
                    idx = sample_uncertainty(preds, budget, candidate_factor=3, seed=seed + r)
                else:
                    idx = sample_stratified(strata, budget, seed=seed + r)
                y = [labels[i] for i in idx]

                if estimator == "classical":
                    est = classical_mean(y)
                else:
                    est = ppi_mean(preds, idx, y)

                halfwidths.append(ci_halfwidth(est))
                abs_errors.append(abs(est.mean - true_theta))
                if est.ci_low <= true_theta <= est.ci_high:
                    covered += 1

            rows.append(EfficiencyRow(
                strategy=strategy, estimator=estimator, budget=budget,
                mean_ci_halfwidth=statistics.mean(halfwidths),
                coverage=covered / repeats,
                mean_abs_error=statistics.mean(abs_errors),
                repeats=repeats,
            ))
    return rows


def labels_to_reach(rows: list[EfficiencyRow], target_halfwidth: float,
                    strategy: str, estimator: str) -> int | None:
    """Smallest budget for a (strategy, estimator) whose mean CI half-width <= target."""
    candidates = sorted(
        (row for row in rows if row.strategy == strategy and row.estimator == estimator),
        key=lambda r: r.budget,
    )
    for row in candidates:
        if row.mean_ci_halfwidth <= target_halfwidth:
            return row.budget
    return None


__all__ = ["EfficiencyRow", "synthetic_pool", "run_label_efficiency", "labels_to_reach"]
