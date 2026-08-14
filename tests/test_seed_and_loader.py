"""W1-2 tests: seed cases load, validate, and are well-formed per the frozen schema."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from trustgate.models import Item, TaskType
from trustgate.registry import load_jsonl, load_seed
from trustgate.registry.loader import LoadError
from trustgate.tasks import SchemaError, validate_item

SEED_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "evalmix" / "seed"


def test_seed_directory_loads_and_validates() -> None:
    ds = load_seed(SEED_DIR)
    assert len(ds.items) == 30
    counts = Counter(it.task_type.value for it in ds.items)
    assert counts["generation"] == 10
    assert counts["retrieval"] == 10
    assert counts["tool_trajectory"] == 10


def test_seed_hash_is_stable_across_loads() -> None:
    assert load_seed(SEED_DIR).content_hash == load_seed(SEED_DIR).content_hash


def test_every_seed_item_has_slice_tags_and_expected() -> None:
    ds = load_seed(SEED_DIR)
    for it in ds.items:
        assert it.tags, f"{it.id} has no slice tags"
        assert "expected" in it.metadata, f"{it.id} missing metadata.expected"


def test_hidden_split_is_reserved() -> None:
    ds = load_seed(SEED_DIR)
    hidden = [it for it in ds.items if it.split.value == "hidden"]
    assert len(hidden) >= 3  # some items withheld for contamination-safe final scoring


def test_retrieval_supporting_ids_reference_real_docs() -> None:
    ds = load_seed(SEED_DIR)
    for it in ds.items:
        if it.task_type is TaskType.RETRIEVAL:
            corpus_ids = {d["id"] for d in it.input["corpus"]}
            for sid in it.references["supporting_ids"]:
                assert sid in corpus_ids


def test_bad_retrieval_item_is_rejected() -> None:
    bad = Item(
        id="bad-ret",
        task_type=TaskType.RETRIEVAL,
        input={"question": "q", "corpus": [{"id": "d1", "text": "t"}]},
        references={"answer": "a", "supporting_ids": ["d_missing"]},  # id not in corpus
    )
    with pytest.raises(SchemaError):
        validate_item(bad)


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = tmp_path / "dupes.jsonl"
    p.write_text(
        '{"id":"x","task_type":"generation","input":"q","references":["a"],"tags":["t"],"metadata":{}}\n'
        '{"id":"x","task_type":"generation","input":"q2","references":["b"],"tags":["t"],"metadata":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(LoadError, match="duplicate item id"):
        load_jsonl(p)


def test_loader_rejects_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text('{"id": "x", not valid json\n', encoding="utf-8")
    with pytest.raises(LoadError, match="invalid JSON"):
        load_jsonl(p)
