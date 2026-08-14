"""Retrieval / RAG evaluators (deterministic).

They read a retrieval :class:`~trustgate.models.Output` whose ``prediction`` is a dict:

    {"answer": str, "cited_ids": [str, ...], "retrieved_ids": [str, ...]}  # ranked

and score it against the item's gold ``{"answer", "supporting_ids"}`` over the item's
in-line corpus. The groundedness metric here is a **lexical proxy** — it is replaced by
the LLM-judge groundedness check in the Judge Lab milestone (W5-6).
"""

from __future__ import annotations

import math
import re

from trustgate.evaluators.base import Evaluator
from trustgate.models import Item, Output, Score

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(str(text).lower()) if len(w) > 2}


def _pred(output: Output) -> dict:
    p = output.prediction
    return p if isinstance(p, dict) else {}


def _norm(x: object) -> str:
    return " ".join(str(x).lower().split())


class RetrievalAnswer(Evaluator):
    """Answer correctness: normalized match (either direction) to the gold answer."""

    name = "retrieval_answer"

    def score(self, item: Item, output: Output) -> list[Score]:
        gold = _norm(item.references.get("answer", ""))
        pred = _norm(_pred(output).get("answer", ""))
        ok = bool(pred) and (gold == pred or (len(gold) > 3 and gold in pred))
        return [Score(item_id=item.id, evaluator=self.name, metric="answer_correctness",
                      value=1.0 if ok else 0.0, passed=ok)]


class CitationPrecision(Evaluator):
    """Fraction of cited ids that are in the gold supporting set."""

    name = "citation_precision"

    def score(self, item: Item, output: Output) -> list[Score]:
        support = set(item.references.get("supporting_ids", []))
        cited = list(_pred(output).get("cited_ids", []))
        if not cited:
            value = 1.0 if not support else 0.0  # no citations is correct only if none needed
        else:
            value = sum(1 for c in cited if c in support) / len(cited)
        return [Score(item_id=item.id, evaluator=self.name, metric="citation_precision",
                      value=value)]


class CitationRecall(Evaluator):
    """Fraction of gold supporting ids that were cited."""

    name = "citation_recall"

    def score(self, item: Item, output: Output) -> list[Score]:
        support = set(item.references.get("supporting_ids", []))
        cited = set(_pred(output).get("cited_ids", []))
        value = 1.0 if not support else len(cited & support) / len(support)
        return [Score(item_id=item.id, evaluator=self.name, metric="citation_recall",
                      value=value)]


class RecallAtK(Evaluator):
    """Whether the gold supporting docs appear within the top-k retrieved ids."""

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.name = f"recall@{k}"

    def score(self, item: Item, output: Output) -> list[Score]:
        support = set(item.references.get("supporting_ids", []))
        topk = set(_pred(output).get("retrieved_ids", [])[: self.k])
        value = 1.0 if not support else len(support & topk) / len(support)
        return [Score(item_id=item.id, evaluator=self.name, metric=f"recall@{self.k}",
                      value=value)]


class NDCGAtK(Evaluator):
    """Normalized DCG@k with binary relevance from the gold supporting set."""

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.name = f"ndcg@{k}"

    def score(self, item: Item, output: Output) -> list[Score]:
        support = set(item.references.get("supporting_ids", []))
        ranked = list(_pred(output).get("retrieved_ids", []))[: self.k]
        if not support:
            value = 1.0
        else:
            dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(ranked) if doc in support)
            ideal_hits = min(len(support), self.k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            value = dcg / idcg if idcg > 0 else 0.0
        return [Score(item_id=item.id, evaluator=self.name, metric=f"ndcg@{self.k}",
                      value=value)]


class GroundednessLexical(Evaluator):
    """Lexical proxy for groundedness: overlap of answer tokens with cited context text.

    A weak stand-in until the LLM-judge groundedness check lands in W5-6.
    """

    name = "groundedness_lexical"

    def score(self, item: Item, output: Output) -> list[Score]:
        corpus = {d["id"]: d["text"] for d in item.input.get("corpus", [])}
        cited = _pred(output).get("cited_ids", [])
        answer_toks = _tokens(_pred(output).get("answer", ""))
        context_toks: set[str] = set()
        for cid in cited:
            context_toks |= _tokens(corpus.get(cid, ""))
        if not answer_toks:
            value = 0.0
        else:
            value = len(answer_toks & context_toks) / len(answer_toks)
        return [Score(item_id=item.id, evaluator=self.name, metric="groundedness_lexical",
                      value=value)]


def default_retrieval_evaluators(k: int = 3) -> list[Evaluator]:
    """The standard deterministic RAG evaluator bank."""
    return [
        RetrievalAnswer(), CitationPrecision(), CitationRecall(),
        RecallAtK(k), NDCGAtK(k), GroundednessLexical(),
    ]


__all__ = [
    "RetrievalAnswer", "CitationPrecision", "CitationRecall",
    "RecallAtK", "NDCGAtK", "GroundednessLexical", "default_retrieval_evaluators",
]
