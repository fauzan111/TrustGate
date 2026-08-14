"""Evaluator bank.

Every evaluator implements ``Evaluator.score(item, output) -> list[Score]`` so the
pipeline can treat deterministic checks, retrieval metrics, and LLM judges uniformly.
The walking skeleton ships the deterministic :class:`ExactMatch`.
"""

from .base import Evaluator
from .deterministic import ExactMatch
from .retrieval import (
    CitationPrecision,
    CitationRecall,
    GroundednessLexical,
    NDCGAtK,
    RecallAtK,
    RetrievalAnswer,
    default_retrieval_evaluators,
)
from .trajectory import (
    Efficiency,
    GoalCompletion,
    Safety,
    ToolCallCorrectness,
    TrajectoryPass,
    default_trajectory_evaluators,
)

__all__ = [
    "Evaluator", "ExactMatch",
    "RetrievalAnswer", "CitationPrecision", "CitationRecall",
    "RecallAtK", "NDCGAtK", "GroundednessLexical", "default_retrieval_evaluators",
    "GoalCompletion", "ToolCallCorrectness", "Efficiency", "Safety", "TrajectoryPass",
    "default_trajectory_evaluators",
]
