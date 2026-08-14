"""W9-10 tests: the conformal release gate is correct and its error rate is bounded."""

from __future__ import annotations

import random

from trustgate.decision import difference_estimate, release_decision
from trustgate.decision.conformal import decide_from_diff
from trustgate.models import Verdict


def _bernoulli(n: int, p: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


def test_difference_estimate_centers_on_true_gap() -> None:
    base = [1.0] * 90 + [0.0] * 10      # 0.90
    cand = [1.0] * 80 + [0.0] * 20      # 0.80
    diff = difference_estimate(cand, base, alpha=0.05)
    assert abs(diff.delta - 0.10) < 1e-9
    assert diff.ci_low < 0.10 < diff.ci_high


def test_hoeffding_is_more_conservative_than_normal() -> None:
    base = _bernoulli(300, 0.9, 1)
    cand = _bernoulli(300, 0.8, 2)
    normal = difference_estimate(cand, base, alpha=0.05, method="normal")
    hoeff = difference_estimate(cand, base, alpha=0.05, method="hoeffding")
    normal_hw = (normal.ci_high - normal.ci_low) / 2
    hoeff_hw = (hoeff.ci_high - hoeff.ci_low) / 2
    assert hoeff_hw > normal_hw


def test_clear_regression_blocks() -> None:
    base = _bernoulli(400, 0.95, 1)
    cand = _bernoulli(400, 0.70, 2)
    d = release_decision(cand, base, margin=0.05, alpha=0.05)
    assert d.verdict is Verdict.BLOCK


def test_equivalent_candidate_ships() -> None:
    base = _bernoulli(800, 0.90, 1)
    cand = _bernoulli(800, 0.90, 7)
    d = release_decision(cand, base, margin=0.05, alpha=0.05)
    assert d.verdict is Verdict.SHIP


def test_small_sample_investigates() -> None:
    base = _bernoulli(25, 0.90, 1)
    cand = _bernoulli(25, 0.80, 2)
    d = release_decision(cand, base, margin=0.05, alpha=0.05)
    assert d.verdict is Verdict.INVESTIGATE


def test_false_block_rate_is_bounded_when_no_regression() -> None:
    # True delta = 0 (candidate == baseline). BLOCK should be rare.
    margin, alpha = 0.05, 0.05
    blocks = 0
    trials = 300
    for r in range(trials):
        base = _bernoulli(300, 0.85, seed=1000 + r)
        cand = _bernoulli(300, 0.85, seed=5000 + r)
        if decide_from_diff(difference_estimate(cand, base, alpha=alpha), margin=margin) is Verdict.BLOCK:
            blocks += 1
    # Blocking requires the whole CI above margin while the true drop is 0 -> essentially never.
    assert blocks / trials < 0.05


def test_false_ship_rate_is_bounded_under_real_regression() -> None:
    # True delta = 0.15 (>> margin 0.05). SHIP should be rare (bounded by alpha).
    margin, alpha = 0.05, 0.05
    ships = 0
    trials = 300
    for r in range(trials):
        base = _bernoulli(300, 0.90, seed=2000 + r)
        cand = _bernoulli(300, 0.75, seed=6000 + r)
        if decide_from_diff(difference_estimate(cand, base, alpha=alpha), margin=margin) is Verdict.SHIP:
            ships += 1
    assert ships / trials < 0.05
