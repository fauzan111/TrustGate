"""Contamination checks via word n-gram overlap.

Two uses:

* **Intra-dataset leakage** — a hidden item that is a near-duplicate of a dev item makes
  the hidden split useless. We flag hidden↔dev pairs whose n-gram overlap exceeds a
  threshold.
* **External contamination** — an item that overlaps heavily with a known training corpus
  (passed as text) is likely memorized rather than genuinely solved.

Overlap is the containment coefficient |A ∩ B| / min(|A|, |B|) over word n-grams — robust
to one text being much longer than the other.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from trustgate.models import Dataset, Item, Split

_WORD = re.compile(r"[a-z0-9]+")


def item_text(item: Item) -> str:
    """Flatten an item's input + references into one lowercase string for comparison."""
    parts = [json.dumps(item.input, sort_keys=True), json.dumps(item.references, sort_keys=True)]
    return " ".join(parts).lower()


def ngrams(text: str, n: int) -> set[str]:
    words = _WORD.findall(text)
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def overlap(a: str, b: str, n: int) -> float:
    """Containment coefficient of word n-grams between two texts, in [0, 1]."""
    ga, gb = ngrams(a, n), ngrams(b, n)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / min(len(ga), len(gb))


@dataclass
class ContaminationHit:
    hidden_id: str
    dev_id: str
    overlap: float


def check_split_leakage(dataset: Dataset, n: int = 8, threshold: float = 0.5) -> list[ContaminationHit]:
    """Flag hidden items that overlap a dev item above ``threshold``."""
    dev = [(it.id, item_text(it)) for it in dataset.by_split(Split.DEV)]
    hidden = [(it.id, item_text(it)) for it in dataset.by_split(Split.HIDDEN)]
    hits: list[ContaminationHit] = []
    for hid, htext in hidden:
        for did, dtext in dev:
            ov = overlap(htext, dtext, n)
            if ov >= threshold:
                hits.append(ContaminationHit(hidden_id=hid, dev_id=did, overlap=ov))
    return sorted(hits, key=lambda h: -h.overlap)


def check_external(dataset: Dataset, corpus_text: str, n: int = 8,
                   threshold: float = 0.5) -> list[tuple[str, float]]:
    """Flag items whose text overlaps a known external corpus above ``threshold``."""
    hits = []
    for it in dataset.items:
        ov = overlap(item_text(it), corpus_text.lower(), n)
        if ov >= threshold:
            hits.append((it.id, ov))
    return sorted(hits, key=lambda t: -t[1])


__all__ = ["item_text", "ngrams", "overlap", "ContaminationHit",
           "check_split_leakage", "check_external"]
