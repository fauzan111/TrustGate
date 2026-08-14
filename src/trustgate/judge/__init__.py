"""Judge Lab — LLM-as-judge with bias probes, agreement, and calibration.

Everything is built against the :class:`JudgeModel` contract, so the default
:class:`SimulatedJudge` (with known biases, for development/testing) and the
:class:`OllamaJudge` (a real local model) are interchangeable.
"""

from .base import JudgeModel, Winner
from .lab import (
    AgreementResult,
    CalibrationResult,
    ConsistencyResult,
    calibrate_threshold,
    judge_human_agreement,
    length_bias_rate,
    position_swap_rate,
    self_consistency,
)
from .llm_judge import LLMJudge
from .ollama import OllamaJudge
from .simulated import SimulatedJudge

__all__ = [
    "JudgeModel", "Winner",
    "SimulatedJudge", "OllamaJudge", "LLMJudge",
    "self_consistency", "ConsistencyResult",
    "position_swap_rate", "length_bias_rate",
    "judge_human_agreement", "AgreementResult",
    "calibrate_threshold", "CalibrationResult",
]
