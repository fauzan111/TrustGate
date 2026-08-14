"""Week-0 walking-skeleton tests: prove the typed pipeline flows end to end."""

from __future__ import annotations

from trustgate.adapters import MockSUT
from trustgate.decision import decide
from trustgate.estimation import estimate_accuracy, wilson_interval
from trustgate.evaluators import ExactMatch
from trustgate.models import Dataset, Item, Output, Score, TaskType, Verdict
from trustgate.pipeline import gate


def _dataset(n: int = 100) -> Dataset:
    items = [
        Item(id=f"i{i:03d}", task_type=TaskType.GENERATION, input=f"q{i}", references="paris")
        for i in range(n)
    ]
    return Dataset(name="t", version="v1", items=items, license="CC0")


def test_dataset_hash_is_deterministic_and_content_addressed() -> None:
    a, b = _dataset(10), _dataset(10)
    assert a.content_hash == b.content_hash
    # Changing content changes the hash.
    b.items[0].references = "london"
    assert Dataset(name="t", version="v1", items=b.items).content_hash != a.content_hash


def test_exact_match_is_case_and_space_insensitive() -> None:
    item = Item(id="x", task_type=TaskType.GENERATION, input="q", references="Paris")
    good = Output(item_id="x", sut="s", prediction="  paris ")
    bad = Output(item_id="x", sut="s", prediction="London")
    assert ExactMatch().score(item, good)[0].passed is True
    assert ExactMatch().score(item, bad)[0].passed is False


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = wilson_interval(successes=90, n=100)
    assert 0.0 <= low < 0.90 < high <= 1.0


def test_pipeline_runs_end_to_end_and_returns_a_decision() -> None:
    ds = _dataset(100)
    _, est, decision = gate(MockSUT("baseline", quality=0.9, seed=1), ds, [ExactMatch()])
    assert est.n == 100
    assert 0.0 <= est.mean <= 1.0
    assert decision.metric == "accuracy"


def test_clear_regression_is_not_shipped() -> None:
    ds = _dataset(300)
    _, base_est, _ = gate(MockSUT("baseline", quality=0.95, seed=1), ds, [ExactMatch()])
    _, _, decision = gate(
        MockSUT("candidate", quality=0.70, seed=2), ds, [ExactMatch()],
        baseline=base_est, epsilon=0.02,
    )
    # A 25-point drop over 300 items must never be waved through.
    assert decision.verdict in (Verdict.BLOCK, Verdict.INVESTIGATE)
    assert decision.verdict is Verdict.BLOCK


def test_equivalent_candidate_ships_given_enough_samples() -> None:
    # Shipping an equivalent candidate requires enough samples that sampling noise is
    # smaller than the tolerance epsilon — otherwise the honest verdict is INVESTIGATE.
    # (This tension is exactly what the sampling/PPI milestone will make label-efficient.)
    ds = _dataset(1500)
    _, base_est, _ = gate(MockSUT("baseline", quality=0.90, seed=1), ds, [ExactMatch()])
    _, _, decision = gate(
        MockSUT("candidate", quality=0.90, seed=1), ds, [ExactMatch()],
        baseline=base_est, epsilon=0.05,
    )
    assert decision.verdict is Verdict.SHIP


def test_small_sample_with_tight_tolerance_defers_to_investigate() -> None:
    # The gate refuses to ship on noise: identical quality, tiny tolerance, small n.
    ds = _dataset(300)
    _, base_est, _ = gate(MockSUT("baseline", quality=0.90, seed=1), ds, [ExactMatch()])
    _, _, decision = gate(
        MockSUT("candidate", quality=0.90, seed=1), ds, [ExactMatch()],
        baseline=base_est, epsilon=0.02,
    )
    assert decision.verdict is Verdict.INVESTIGATE


def test_no_baseline_ships_when_confident() -> None:
    scores = [
        Score(item_id=f"i{i}", evaluator="exact_match", metric="accuracy", value=1.0)
        for i in range(50)
    ]
    d = decide(estimate_accuracy(scores), baseline=None)
    assert d.verdict is Verdict.SHIP
