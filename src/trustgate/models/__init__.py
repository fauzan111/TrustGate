"""Core typed domain model for TrustGate.

Everything in TrustGate flows through these Pydantic types so that every component
(adapters, evaluators, sampling, estimation, decision) shares one contract.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """The five task shapes TrustGate can evaluate under one contract."""

    GENERATION = "generation"          # free-form text
    STRUCTURED = "structured"          # JSON-schema-validated output
    RETRIEVAL = "retrieval"            # RAG: answer + retrieved context / citations
    TOOL_TRAJECTORY = "tool_trajectory"  # agent: sequence of steps / tool calls
    PAIRWISE = "pairwise"              # preference between two candidates


class Split(str, Enum):
    """Dev items are used while building; hidden items are reserved for final,
    contamination-safe scoring."""

    DEV = "dev"
    HIDDEN = "hidden"


class Item(BaseModel):
    """A single test case."""

    id: str
    task_type: TaskType
    input: Any                                   # str or dict, depending on task_type
    references: Any | None = None                # gold answer(s) / accepted set
    tags: list[str] = Field(default_factory=list)  # slice labels, e.g. ["multi_hop"]
    split: Split = Split.DEV
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """An immutable, content-hashed, versioned collection of items."""

    name: str
    version: str
    items: list[Item]
    license: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content_hash(self) -> str:
        """Deterministic hash over item content — the reproducibility anchor.

        Excludes ``created_at`` so identical items always hash identically.
        """
        payload = [
            it.model_dump(mode="json", exclude={"metadata"}) for it in self.items
        ]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def by_split(self, split: Split) -> list[Item]:
        return [it for it in self.items if it.split == split]


class Output(BaseModel):
    """What a SUT produced for one item."""

    item_id: str
    sut: str
    prediction: Any                      # str or dict
    trace: list[dict[str, Any]] | None = None   # tool-trajectory steps, if any
    latency_ms: float | None = None
    cost_usd: float | None = None


class Run(BaseModel):
    """One SUT executed over one dataset version → a set of outputs."""

    id: str
    sut_name: str
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    outputs: list[Output]

    def output_for(self, item_id: str) -> Output | None:
        return next((o for o in self.outputs if o.item_id == item_id), None)


class Score(BaseModel):
    """One evaluator's verdict on one item, for one metric."""

    item_id: str
    evaluator: str
    metric: str
    value: float                 # continuous score; for pass/fail use 1.0 / 0.0
    passed: bool | None = None


class Estimate(BaseModel):
    """An aggregate metric with an uncertainty interval and its sample size."""

    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n: int
    method: str = "wilson"


class Verdict(str, Enum):
    SHIP = "ship"
    INVESTIGATE = "investigate"
    BLOCK = "block"


class Decision(BaseModel):
    """The release-gate output: candidate vs. baseline."""

    verdict: Verdict
    metric: str
    candidate: Estimate
    baseline: Estimate | None = None
    delta: float | None = None                 # baseline_mean - candidate_mean (drop)
    delta_ci_low: float | None = None
    delta_ci_high: float | None = None
    margin: float | None = None                # tolerated drop
    alpha: float | None = None                 # target false-decision rate
    method: str | None = None                  # "normal" | "hoeffding"
    reason: str = ""


__all__ = [
    "TaskType", "Split", "Item", "Dataset", "Output", "Run",
    "Score", "Estimate", "Verdict", "Decision",
]
