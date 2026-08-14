"""Load human (or reference) pass labels from a labeling sheet.

The labeling sheet (see ``trustgate labeling-template``) has one row per item with an
``adjudicated_pass`` column. This loader turns a filled sheet into ``{item_id: 0/1}`` that
the Judge Lab consumes for judge↔human agreement and threshold calibration — so *real*
human labels drop in wherever the ground-truth oracle is used today.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_labels(csv_path: str | Path, column: str = "adjudicated_pass") -> dict[str, int]:
    """Return ``{item_id: pass}`` for rows whose ``column`` is a filled 0/1."""
    labels: dict[str, int] = {}
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get(column) or "").strip()
            if raw in {"0", "1"}:
                labels[row["item_id"]] = int(raw)
    return labels


__all__ = ["load_labels"]
