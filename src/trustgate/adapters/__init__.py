"""SUT (System Under Test) adapters.

Every backend the user wants to evaluate — a hosted endpoint, a local Ollama model,
or an offline file of pre-computed outputs — implements the same ``SUTAdapter`` contract.
The walking skeleton ships :class:`MockSUT`; real adapters are added in later milestones.
"""

from .base import SUTAdapter
from .mock import MockAgentSUT, MockRAGSUT, MockSUT

__all__ = ["SUTAdapter", "MockSUT", "MockRAGSUT", "MockAgentSUT"]
