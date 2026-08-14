"""Sampling strategies: choosing *which* items to spend human labels on.

Under a fixed label budget, the strategy decides which pool items a human annotates. The
label-efficiency experiment compares them: uncertainty sampling concentrates labels where
the judge is least sure, which — combined with PPI — reaches a target precision with fewer
labels than random labeling.

Each function returns a sorted list of selected pool indices.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable, Sequence


def sample_random(pool_size: int, budget: int, seed: int = 0) -> list[int]:
    budget = min(budget, pool_size)
    rng = random.Random(seed)
    return sorted(rng.sample(range(pool_size), budget))


def sample_uncertainty(scores: Sequence[float], budget: int,
                       candidate_factor: int = 1, seed: int = 0) -> list[int]:
    """Pick ``budget`` items near the 0.5 decision boundary (least judge confidence).

    With ``candidate_factor > 1`` the selection is drawn at random from the top
    ``budget * candidate_factor`` most-uncertain items, giving run-to-run variability (so
    coverage can be estimated) while staying focused on uncertain examples.
    """
    budget = min(budget, len(scores))
    order = sorted(range(len(scores)), key=lambda i: (abs(scores[i] - 0.5), i))
    if candidate_factor <= 1:
        return sorted(order[:budget])
    pool = order[: min(len(scores), budget * candidate_factor)]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, budget))


def sample_stratified(strata: Sequence[Hashable], budget: int, seed: int = 0) -> list[int]:
    """Allocate the budget across strata proportionally to stratum size, random within.

    ``strata[i]`` is the stratum key of pool item ``i`` (e.g. a slice tag or a score bin).
    """
    n = len(strata)
    budget = min(budget, n)
    rng = random.Random(seed)

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, key in enumerate(strata):
        groups[key].append(i)

    # Largest-remainder allocation so the totals sum exactly to the budget.
    alloc: dict[Hashable, int] = {}
    remainders: list[tuple[float, Hashable]] = []
    assigned = 0
    for key, idxs in groups.items():
        exact = budget * len(idxs) / n
        base = int(exact)
        alloc[key] = min(base, len(idxs))
        assigned += alloc[key]
        remainders.append((exact - base, key))

    for _, key in sorted(remainders, reverse=True):
        if assigned >= budget:
            break
        if alloc[key] < len(groups[key]):
            alloc[key] += 1
            assigned += 1

    selected: list[int] = []
    for key, idxs in groups.items():
        selected.extend(rng.sample(idxs, alloc[key]))
    return sorted(selected)


__all__ = ["sample_random", "sample_uncertainty", "sample_stratified"]
