"""Load seed cases from JSONL into a validated :class:`~trustgate.models.Dataset`."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from trustgate.models import Dataset, Item
from trustgate.tasks import SchemaError, validate_item


class LoadError(ValueError):
    """Raised when a JSONL file cannot be parsed into valid items."""


def _load_items(path: Path) -> list[Item]:
    items: list[Item] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LoadError(f"{path.name}:{lineno}: invalid JSON: {exc}") from exc
            try:
                item = Item.model_validate(obj)
            except ValidationError as exc:
                raise LoadError(f"{path.name}:{lineno}: does not match Item schema:\n{exc}") from exc
            try:
                validate_item(item)
            except SchemaError as exc:
                raise LoadError(f"{path.name}:{lineno}: {exc}") from exc
            items.append(item)
    return items


def _check_unique_ids(items: list[Item]) -> None:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise LoadError(f"duplicate item id: {it.id}")
        seen.add(it.id)


def load_jsonl(path: str | Path, name: str = "dataset", version: str = "v1",
               license: str = "unknown") -> Dataset:
    """Load a single JSONL file into a validated Dataset."""
    items = _load_items(Path(path))
    _check_unique_ids(items)
    return Dataset(name=name, version=version, items=items, license=license)


def load_seed(directory: str | Path, name: str = "evalmix-seed", version: str = "v1",
              license: str = "CC0") -> Dataset:
    """Load and merge every ``*.jsonl`` file in a directory into one Dataset."""
    directory = Path(directory)
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise LoadError(f"no .jsonl files found in {directory}")
    items: list[Item] = []
    for f in files:
        items.extend(_load_items(f))
    _check_unique_ids(items)
    return Dataset(name=name, version=version, items=items, license=license)


__all__ = ["LoadError", "load_jsonl", "load_seed"]
