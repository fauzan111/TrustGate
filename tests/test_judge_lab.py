"""W5-6 tests: the Judge Lab probes detect *known* injected biases, and calibration helps.

These use SimulatedJudge (biases we set ourselves) so every probe has a ground truth to
verify against — no model or network required.
"""

from __future__ import annotations

from trustgate.judge import (
    LLMJudge,
    SimulatedJudge,
    calibrate_threshold,
    judge_human_agreement,
    length_bias_rate,
    position_swap_rate,
    self_consistency,
)
from trustgate.models import Item, Output, TaskType
from trustgate.stats import balanced_accuracy, cohen_kappa


def test_simulated_judge_is_deterministic() -> None:
    j = SimulatedJudge(noise=0.3)
    assert j.score("q", "a", "a", trial=0) == j.score("q", "a", "a", trial=0)
    assert j.score("q", "a", "a", trial=0) != j.score("q", "a", "a", trial=1)


def test_length_probe_detects_injected_length_bias() -> None:
    pairs = [(f"q{i}", "short", "short plus a much much longer restatement of the answer")
             for i in range(30)]
    biased = length_bias_rate(SimulatedJudge(length_bias=0.6), pairs)
    unbiased = length_bias_rate(SimulatedJudge(length_bias=0.0, noise=0.2), pairs)
    assert biased > 0.85
    assert unbiased < 0.65
    assert biased > unbiased


def test_position_probe_detects_injected_position_bias() -> None:
    pairs = [(f"q{i}", f"x{i}", f"y{i}") for i in range(30)]
    biased = position_swap_rate(SimulatedJudge(position_bias=0.9), pairs)
    unbiased = position_swap_rate(SimulatedJudge(position_bias=0.0, noise=0.2), pairs)
    assert biased > 0.85
    assert unbiased < 0.15


def test_self_consistency_reflects_noise() -> None:
    quiet = self_consistency(SimulatedJudge(noise=0.0), "q", "a", "a", trials=8)
    loud = self_consistency(SimulatedJudge(noise=0.6), "q", "a", "a", trials=8)
    assert quiet.score_std == 0.0
    assert loud.score_std > 0.0


def test_stats_primitives() -> None:
    assert balanced_accuracy([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert balanced_accuracy([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) + 1.0) < 1e-9


def test_calibration_never_worse_and_can_recover_agreement() -> None:
    # Negatives cluster at 0.55, positives at 0.62: the default 0.5 threshold calls
    # everything positive (balanced acc 0.5); a tuned threshold separates them.
    scores = [0.55, 0.54, 0.56, 0.62, 0.63, 0.61]
    human = [0, 0, 0, 1, 1, 1]
    result = calibrate_threshold(scores, human)
    assert result.agreement_before <= result.agreement_after
    assert result.agreement_after == 1.0
    assert 0.56 < result.best_threshold <= 0.62


def test_judge_human_agreement_shape() -> None:
    r = judge_human_agreement([1, 0, 1, 0], [1, 0, 0, 0])
    assert r.n == 4
    assert 0.0 <= r.balanced_accuracy <= 1.0


def test_llm_judge_evaluator_plugs_into_scoring() -> None:
    judge = SimulatedJudge(noise=0.0, base_correct=0.8, base_wrong=0.2)
    ev = LLMJudge(judge, threshold=0.5)
    item = Item(id="g1", task_type=TaskType.GENERATION, input="Capital of France?",
                references=["Paris"])
    good = ev.score(item, Output(item_id="g1", sut="s", prediction="Paris"))[0]
    bad = ev.score(item, Output(item_id="g1", sut="s", prediction="Berlin"))[0]
    assert good.metric == "judge_pass" and good.passed is True
    assert bad.passed is False
