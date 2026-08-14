"""Registry — loading, validating, and (later) persisting datasets.

Milestone W1-2 ships the JSONL loader with schema validation and duplicate-id detection.
Milestone W3-4 adds SQLite/Parquet persistence, immutable versioning, and contamination
checks on top of this same interface.
"""

from .contamination import check_external, check_split_leakage
from .loader import load_jsonl, load_seed
from .store import ImmutabilityError, RegistryStore

__all__ = [
    "load_jsonl", "load_seed",
    "RegistryStore", "ImmutabilityError",
    "check_split_leakage", "check_external",
]
