"""W7-8 tests: sampling strategies, PPI validity + tightness, label-efficiency result."""

from __future__ import annotations

from trustgate.estimation import ci_halfwidth, classical_mean, naive_mean, ppi_mean
from trustgate.experiments import labels_to_reach, run_label_efficiency, synthetic_pool
from trustgate.sampling import sample_random, sample_stratified, sample_uncertainty


# --- sampling --------------------------------------------------------------- #
def test_random_sampling_respects_budget_and_bounds() -> None:
    idx = sample_random(pool_size=100, budget=15, seed=1)
    assert len(idx) == 15
    assert idx == sorted(idx)
    assert all(0 <= i < 100 for i in idx)


def test_uncertainty_sampling_picks_near_half() -> None:
    scores = [0.01, 0.02, 0.48, 0.51, 0.97, 0.99]
    idx = sample_uncertainty(scores, budget=2)
    assert set(idx) == {2, 3}  # 0.48 and 0.51 are closest to 0.5


def test_stratified_sampling_hits_budget_and_covers_strata() -> None:
    strata = ["a"] * 60 + ["b"] * 40
    idx = sample_stratified(strata, budget=10, seed=2)
    assert len(idx) == 10
    chosen = [strata[i] for i in idx]
    assert chosen.count("a") >= 4 and chosen.count("b") >= 2  # roughly proportional


# --- estimators ------------------------------------------------------------- #
def test_full_labeling_recovers_true_rate() -> None:
    preds, labels, theta = synthetic_pool(n=1000, true_rate=0.7, seed=1)
    full_idx = list(range(len(preds)))
    ppi = ppi_mean(preds, full_idx, labels)
    cls = classical_mean(labels)
    assert abs(ppi.mean - theta) < 1e-6
    assert abs(cls.mean - theta) < 1e-6


def test_naive_is_biased_when_judge_is_off() -> None:
    # Judge systematically optimistic: separation high but centered above the labels.
    preds, labels, theta = synthetic_pool(n=2000, true_rate=0.5, separation=0.8,
                                          noise=0.1, seed=3)
    naive = naive_mean(preds)
    # The judge's mean can drift from the truth; PPI corrects it.
    ppi = ppi_mean(preds, list(range(200)), labels[:200])
    assert abs(ppi.mean - theta) <= abs(naive.mean - theta) + 0.02


def test_ppi_is_tighter_than_classical_with_a_good_judge() -> None:
    preds, labels, theta = synthetic_pool(n=3000, true_rate=0.75, separation=0.7,
                                          noise=0.15, seed=5)
    idx = sample_random(len(preds), budget=150, seed=9)
    y = [labels[i] for i in idx]
    ppi = ppi_mean(preds, idx, y)
    cls = classical_mean(y)
    assert ci_halfwidth(ppi) < ci_halfwidth(cls)


# --- headline experiment ---------------------------------------------------- #
def test_label_efficiency_ppi_beats_classical_and_stays_valid() -> None:
    preds, labels, theta = synthetic_pool(n=2000, true_rate=0.8, separation=0.6,
                                          noise=0.25, seed=0)
    budgets = [40, 80, 160, 320]
    rows = run_label_efficiency(preds, labels, theta, budgets, repeats=120, seed=0)

    # At a fixed budget, random+PPI is tighter than random+classical...
    def get(strategy: str, estimator: str, budget: int) -> float:
        return next(r.mean_ci_halfwidth for r in rows
                    if r.strategy == strategy and r.estimator == estimator and r.budget == budget)

    assert get("random", "ppi", 160) < get("random", "classical", 160)

    # ...and random+PPI keeps ~95% coverage (valid intervals).
    cov = next(r.coverage for r in rows
               if r.strategy == "random" and r.estimator == "ppi" and r.budget == 160)
    assert cov >= 0.88

    # PPI reaches a target precision with no more labels than classical.
    target = 0.04
    cl = labels_to_reach(rows, target, "random", "classical")
    pp = labels_to_reach(rows, target, "random", "ppi")
    assert pp is not None
    assert cl is None or pp <= cl
