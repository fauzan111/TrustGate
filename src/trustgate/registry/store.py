"""SQLite-backed registry: immutable, content-addressed datasets + runs + scores.

Persistence uses the Python stdlib ``sqlite3`` only (no heavyweight deps). Datasets are
**immutable per (name, version)**: re-saving the same content is idempotent, but attempting
to overwrite a version with *different* content raises :class:`ImmutabilityError`. That is
the reproducibility guarantee the release gate relies on.

(Parquet artifact export is a later optimization; JSON payloads in SQLite are enough at
seed/flagship scale and keep the repo dependency-light.)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from trustgate.models import Dataset, Run, Score


class ImmutabilityError(RuntimeError):
    """Raised when overwriting an existing dataset version with different content."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    hash TEXT NOT NULL,
    license TEXT,
    created_at TEXT,
    payload TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    sut_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    passed INTEGER
);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
"""


class RegistryStore:
    def __init__(self, db_path: str | Path = "trustgate.sqlite") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RegistryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- datasets -------------------------------------------------------------
    def save_dataset(self, dataset: Dataset) -> str:
        """Persist a dataset immutably. Returns its content hash."""
        row = self._conn.execute(
            "SELECT hash FROM datasets WHERE name=? AND version=?",
            (dataset.name, dataset.version),
        ).fetchone()
        new_hash = dataset.content_hash
        if row is not None:
            if row[0] != new_hash:
                raise ImmutabilityError(
                    f"{dataset.name} {dataset.version} already exists with hash {row[0]}; "
                    f"refusing to overwrite with different content (hash {new_hash}). "
                    f"Bump the version."
                )
            return new_hash  # idempotent: identical content already stored
        self._conn.execute(
            "INSERT INTO datasets (name, version, hash, license, created_at, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (dataset.name, dataset.version, new_hash, dataset.license,
             dataset.created_at.isoformat(), dataset.model_dump_json()),
        )
        self._conn.commit()
        return new_hash

    def load_dataset(self, name: str, version: str) -> Dataset:
        row = self._conn.execute(
            "SELECT payload FROM datasets WHERE name=? AND version=?", (name, version),
        ).fetchone()
        if row is None:
            raise KeyError(f"dataset {name} {version} not found")
        return Dataset.model_validate_json(row[0])

    def list_datasets(self) -> list[tuple[str, str, str, int]]:
        """Return (name, version, hash, item_count) for every stored dataset."""
        out = []
        for name, version, h, payload in self._conn.execute(
            "SELECT name, version, hash, payload FROM datasets ORDER BY name, version"
        ):
            n_items = Dataset.model_validate_json(payload).items.__len__()
            out.append((name, version, h, n_items))
        return out

    # --- runs & scores --------------------------------------------------------
    def save_run(self, run: Run, scores: list[Score] | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs "
            "(id, sut_name, dataset_name, dataset_version, dataset_hash, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run.id, run.sut_name, run.dataset_name, run.dataset_version,
             run.dataset_hash, run.model_dump_json()),
        )
        if scores:
            self._conn.executemany(
                "INSERT INTO scores (run_id, item_id, evaluator, metric, value, passed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(run.id, s.item_id, s.evaluator, s.metric, s.value,
                  None if s.passed is None else int(s.passed)) for s in scores],
            )
        self._conn.commit()

    def load_run(self, run_id: str) -> Run:
        row = self._conn.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} not found")
        return Run.model_validate_json(row[0])


__all__ = ["RegistryStore", "ImmutabilityError"]
