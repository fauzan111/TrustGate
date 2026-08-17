"""Ollama-backed judge (same interface as SimulatedJudge).

This talks to a local Ollama server over HTTP (default http://localhost:11434) using only
the stdlib, so it needs no extra Python dependency. It is not exercised by the test suite
(no server in CI); the moment you run e.g. ``ollama pull llama3.1`` it works with zero
changes elsewhere — the Judge Lab and LLMJudge already speak the JudgeModel contract.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from trustgate.judge.base import JudgeModel, Winner

_SCORE_PROMPT = """You are a strict evaluator. Rate how well the ANSWER responds to the \
QUESTION on a scale from 0.0 (poor) to 1.0 (excellent). {ref_clause}
Return ONLY a number between 0 and 1.

QUESTION: {question}
ANSWER: {answer}
SCORE:"""

_COMPARE_PROMPT = """You are a strict evaluator. Which answer is better for the QUESTION?
Reply with exactly "A" or "B".

QUESTION: {question}
ANSWER A: {answer_a}
ANSWER B: {answer_b}
BETTER:"""


class OllamaJudge(JudgeModel):
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434",
                 temperature: float = 0.0, num_ctx: int | None = None) -> None:
        import os
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        # Cap the context window. Ollama otherwise defaults to the model's full context
        # (128k for llama3.x), whose KV cache needs many gigabytes and fails to load on a
        # typical laptop. Judge prompts are short, so a few thousand tokens is ample.
        self.num_ctx = num_ctx if num_ctx is not None else int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

    def _generate(self, prompt: str, trial: int) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": trial, "num_ctx": self.num_ctx},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            # Generous timeout: the model can take tens of seconds to load on the first call.
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read()).get("response", "")
        except urllib.error.HTTPError as exc:  # pragma: no cover - needs server
            detail = exc.read().decode(errors="replace")[:400]
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for model '{self.model}': {detail}. "
                f"A memory error usually means the model is too big; try a smaller one."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - needs server
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} (model '{self.model}'). "
                f"Is the server running and the model pulled? Original error: {exc}"
            ) from exc

    def score(self, question: str, answer: str, reference: str | None = None,
              *, trial: int = 0) -> float:  # pragma: no cover - needs server
        ref_clause = f"Reference answer: {reference}" if reference else ""
        text = self._generate(_SCORE_PROMPT.format(
            question=question, answer=answer, ref_clause=ref_clause), trial)
        m = re.search(r"[01](?:\.\d+)?|0?\.\d+", text)
        return max(0.0, min(1.0, float(m.group()))) if m else 0.0

    def compare(self, question: str, answer_a: str, answer_b: str,
                *, trial: int = 0) -> Winner:  # pragma: no cover - needs server
        text = self._generate(_COMPARE_PROMPT.format(
            question=question, answer_a=answer_a, answer_b=answer_b), trial).strip().upper()
        return "b" if text.startswith("B") else "a"


__all__ = ["OllamaJudge"]
