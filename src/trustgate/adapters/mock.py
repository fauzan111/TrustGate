"""A deterministic mock SUT.

It answers each item *correctly* with probability ``quality`` and *wrongly* otherwise.
Because the coin flip is seeded from ``(item.id, seed)``, runs are fully reproducible —
which lets us simulate a controlled regression (e.g. quality 0.90 -> 0.80) end-to-end
before any real model is wired in.
"""

from __future__ import annotations

import hashlib

from trustgate.adapters.base import SUTAdapter
from trustgate.models import Item, Output


def _seeded_unit(item_id: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) from an item id + seed."""
    h = hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockSUT(SUTAdapter):
    def __init__(self, name: str, quality: float = 0.9, seed: int = 0) -> None:
        self.name = name
        self.quality = quality
        self.seed = seed

    def predict(self, item: Item) -> Output:
        correct = _seeded_unit(item.id, self.seed) < self.quality
        gold = item.references
        if correct:
            # A real SUT emits a single answer string, not the list of accepted answers.
            prediction = gold[0] if isinstance(gold, list) and gold else gold
        else:
            prediction = "no answer available"
        return Output(
            item_id=item.id,
            sut=self.name,
            prediction=prediction,
            latency_ms=1.0,
            cost_usd=0.0,
        )


class MockRAGSUT(SUTAdapter):
    """A mock retrieval SUT that emits ``{answer, cited_ids, retrieved_ids}``.

    With probability ``quality`` it produces a *good* output (gold answer, correct
    citations, gold docs ranked first). Otherwise it degrades: a wrong answer and a
    distractor citation, with gold docs pushed down the ranking. Deterministic per item.
    """

    def __init__(self, name: str, quality: float = 0.9, seed: int = 0) -> None:
        self.name = name
        self.quality = quality
        self.seed = seed

    def predict(self, item: Item) -> Output:
        corpus_ids = [d["id"] for d in item.input["corpus"]]
        support = list(item.references.get("supporting_ids", []))
        gold_answer = item.references.get("answer", "")
        good = _seeded_unit(item.id, self.seed) < self.quality

        if good:
            answer = gold_answer
            cited = support[:] if support else []
            ranked = support + [c for c in corpus_ids if c not in support]
        else:
            answer = "no relevant information found"
            distractor = next((c for c in corpus_ids if c not in support), None)
            cited = [distractor] if distractor else []
            ranked = [c for c in corpus_ids if c not in support] + support

        return Output(
            item_id=item.id,
            sut=self.name,
            prediction={"answer": answer, "cited_ids": cited, "retrieved_ids": ranked},
            latency_ms=1.0,
            cost_usd=0.0,
        )


class MockAgentSUT(SUTAdapter):
    """A mock agent that emits ``{final_answer, calls}`` trajectories.

    A *good* run executes the gold required calls (and escalates when it should); a *bad*
    run skips required calls and, on safety items, performs the consequential action
    instead of escalating — exactly the unsafe behavior the Safety evaluator must catch.
    """

    def __init__(self, name: str, quality: float = 0.9, seed: int = 0) -> None:
        self.name = name
        self.quality = quality
        self.seed = seed

    def predict(self, item: Item) -> Output:
        required = item.references.get("required_calls", []) if isinstance(item.references, dict) else []
        tools = [t["name"] for t in item.input.get("tools", [])]
        good = _seeded_unit(item.id, self.seed) < self.quality

        if good:
            calls = [{"tool": rc["tool"], "args": rc.get("args_contains", "")} for rc in required]
            final = "task completed"
        else:
            # Skip required calls; on escalate items, do the risky action instead.
            action_tool = next((t for t in tools if t != "escalate"), None)
            calls = [{"tool": action_tool, "args": "executed without approval"}] if action_tool else []
            final = "no answer available"

        return Output(item_id=item.id, sut=self.name,
                      prediction={"final_answer": final, "calls": calls},
                      latency_ms=1.0, cost_usd=0.0)
