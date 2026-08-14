"""W3-4 tests: retrieval evaluators, immutable registry, contamination checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from trustgate.adapters import MockRAGSUT
from trustgate.estimation import aggregate
from trustgate.evaluators import default_retrieval_evaluators
from trustgate.models import Dataset, Item, Split, TaskType
from trustgate.pipeline import evaluate, run_sut
from trustgate.registry import RegistryStore, check_split_leakage, load_seed
from trustgate.registry.store import ImmutabilityError

SEED_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "evalmix" / "seed"


def _retrieval_item() -> Item:
    return Item(
        id="ret-x",
        task_type=TaskType.RETRIEVAL,
        input={"question": "What is the warranty?",
               "corpus": [{"id": "d1", "text": "The warranty lasts 24 months for the motor."},
                          {"id": "d2", "text": "Shipping is free."}]},
        references={"answer": "24 months", "supporting_ids": ["d1"]},
    )


def _score_map(scores) -> dict[str, float]:
    return {s.metric: s.value for s in scores}


def test_perfect_retrieval_output_scores_top() -> None:
    item = _retrieval_item()
    out = MockRAGSUT("perfect", quality=1.0, seed=1).predict(item)
    scores = _score_map(
        [s for ev in default_retrieval_evaluators(k=2) for s in ev.score(item, out)]
    )
    assert scores["answer_correctness"] == 1.0
    assert scores["citation_precision"] == 1.0
    assert scores["citation_recall"] == 1.0
    assert scores["recall@2"] == 1.0
    assert scores["ndcg@2"] == 1.0
    assert scores["groundedness_lexical"] > 0.0


def test_degraded_retrieval_output_scores_worse() -> None:
    item = _retrieval_item()
    out = MockRAGSUT("bad", quality=0.0, seed=1).predict(item)
    scores = _score_map(
        [s for ev in default_retrieval_evaluators(k=2) for s in ev.score(item, out)]
    )
    assert scores["answer_correctness"] == 0.0
    assert scores["citation_precision"] == 0.0      # cited a distractor
    assert scores["recall@2"] <= 1.0
    assert scores["ndcg@2"] < 1.0                   # gold pushed down the ranking


def test_rag_bank_mean_drops_with_quality() -> None:
    ds = load_seed(SEED_DIR)
    retrieval = Dataset(name="r", version="v1",
                        items=[it for it in ds.items if it.task_type is TaskType.RETRIEVAL])
    evs = default_retrieval_evaluators(k=3)

    good = aggregate(evaluate(run_sut(MockRAGSUT("g", quality=1.0, seed=1), retrieval),
                              retrieval, evs))
    bad = aggregate(evaluate(run_sut(MockRAGSUT("b", quality=0.0, seed=1), retrieval),
                             retrieval, evs))
    assert good["citation_precision"].mean > bad["citation_precision"].mean
    assert good["ndcg@3"].mean > bad["ndcg@3"].mean


def test_registry_roundtrip_and_immutability(tmp_path: Path) -> None:
    db = tmp_path / "reg.sqlite"
    ds = load_seed(SEED_DIR, name="evalmix-seed", version="v1")
    with RegistryStore(db) as store:
        h1 = store.save_dataset(ds)
        # Idempotent: saving identical content again is fine.
        assert store.save_dataset(ds) == h1
        loaded = store.load_dataset("evalmix-seed", "v1")
        assert loaded.content_hash == ds.content_hash
        assert len(loaded.items) == len(ds.items)

        # Mutating content under the same version must be refused.
        mutated_items = list(ds.items)
        mutated_items[0] = mutated_items[0].model_copy(update={"references": ["CHANGED"]})
        mutated = Dataset(name="evalmix-seed", version="v1", items=mutated_items)
        with pytest.raises(ImmutabilityError):
            store.save_dataset(mutated)


def test_seed_set_has_no_split_leakage() -> None:
    ds = load_seed(SEED_DIR)
    assert check_split_leakage(ds, n=8, threshold=0.5) == []


def test_contamination_detects_near_duplicate_across_splits() -> None:
    shared = {"question": "What is the return window for online orders?",
              "corpus": [{"id": "d1", "text": "Online orders may be returned within 30 days of delivery."}]}
    refs = {"answer": "30 days", "supporting_ids": ["d1"]}
    ds = Dataset(name="c", version="v1", items=[
        Item(id="dev1", task_type=TaskType.RETRIEVAL, input=shared, references=refs, split=Split.DEV),
        Item(id="hid1", task_type=TaskType.RETRIEVAL, input=shared, references=refs, split=Split.HIDDEN),
    ])
    hits = check_split_leakage(ds, n=8, threshold=0.5)
    assert any(h.hidden_id == "hid1" and h.dev_id == "dev1" for h in hits)
