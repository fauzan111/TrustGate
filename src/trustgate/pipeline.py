"""The end-to-end pipeline: run → evaluate → estimate → decide.

This is the walking skeleton's spine. Each stage is deliberately small and typed so that
later milestones can swap richer implementations (judges, sampling, PPI, conformal
decisions) without changing the orchestration.
"""

from __future__ import annotations

import uuid

from trustgate.adapters.base import SUTAdapter
from trustgate.decision import decide
from trustgate.estimation import estimate_accuracy
from trustgate.evaluators.base import Evaluator
from trustgate.models import Dataset, Decision, Estimate, Run, Score, Split


def run_sut(adapter: SUTAdapter, dataset: Dataset, split: Split | None = None) -> Run:
    """Execute a SUT over (a split of) a dataset."""
    items = dataset.by_split(split) if split else dataset.items
    outputs = [adapter.predict(it) for it in items]
    return Run(
        id=uuid.uuid4().hex[:12],
        sut_name=adapter.name,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        outputs=outputs,
    )


def evaluate(run: Run, dataset: Dataset, evaluators: list[Evaluator]) -> list[Score]:
    """Score every output with every evaluator."""
    by_id = {it.id: it for it in dataset.items}
    scores: list[Score] = []
    for out in run.outputs:
        item = by_id[out.item_id]
        for ev in evaluators:
            scores.extend(ev.score(item, out))
    return scores


def gate(
    adapter: SUTAdapter,
    dataset: Dataset,
    evaluators: list[Evaluator],
    baseline: Estimate | None = None,
    epsilon: float = 0.02,
    split: Split | None = None,
) -> tuple[Run, Estimate, Decision]:
    """Full path: run the SUT, evaluate, estimate accuracy, and emit a release decision."""
    run = run_sut(adapter, dataset, split=split)
    scores = evaluate(run, dataset, evaluators)
    est = estimate_accuracy(scores)
    decision = decide(est, baseline=baseline, epsilon=epsilon)
    return run, est, decision


__all__ = ["run_sut", "evaluate", "gate"]
