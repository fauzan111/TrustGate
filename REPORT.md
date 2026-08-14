# TrustGate — Technical Report

*Risk-controlled evaluation & release-gating for LLM / RAG / agent systems.*

## Scientific question
How few human labels does a calibrated judge need to detect a 5–10% quality regression
without excessive false-ship or false-block decisions?

## Benchmark
Curated seed set: **30 items** (10 generation,
10 retrieval, 10 tool-trajectory), 20%
hidden, with seeded failures and abstain/escalate safety cases. A synthetic **EvalMix**
(≈500 items) adds statistical weight (`trustgate build-evalmix`).

## Key results

### 1. Label efficiency (Prediction-Powered Inference)
To reach a ±0.05 confidence interval on the pass rate (random sampling, valid ~95% coverage):

| estimator | labels needed |
|---|---|
| classical (labels only) | 320 |
| **PPI (judge + labels)** | **80** |

**PPI reaches the same precision with 75% fewer human labels.** Uncertainty
sampling tightens intervals further but *breaks* coverage (~0.02) — stratified sampling
preserves both tightness and validity.

### 2. Judge trustworthiness (bias probes)
| probe | biased judge | clean judge |
|---|---|---|
| length-bias rate | 1.00 | 0.60 |
| position-swap rate | 1.00 | ~0.00 |

Pointwise judge↔human agreement stayed ~0.88 even for a badly biased judge — showing why
ranking-level probes, not just agreement, are required.

### 3. Risk-controlled release gate
| scenario | verdict |
|---|---|
| baseline 0.95 → candidate 0.60 | **BLOCK** |
| baseline 0.90 → candidate 0.90 | **INVESTIGATE** |

Decisions use a non-inferiority CI on the quality drop; false-block and false-ship rates are
bounded below `alpha` (verified by simulation tests).

## Evidence index
| claim | command | artifact |
|---|---|---|
| Benchmark composition | `trustgate stats benchmarks/evalmix/seed` | seed JSONL |
| No hidden/dev leakage | `trustgate contamination benchmarks/evalmix/seed` | — |
| PPI ~75% fewer labels | `trustgate label-efficiency` | `benchmarks/evalmix/label_efficiency.csv` |
| Probes detect known bias | `trustgate judge-lab --length-bias 0.4` | — |
| Gate blocks a regression | `trustgate gate-ci --candidate-quality 0.60` | exit code 1 |
| RAG evaluators | `trustgate rag-demo` | — |
| Agent evaluators | `trustgate agent-demo` | — |

Reproduce: `pip install -e ".[dev]" && pytest -q` (44 tests).

## Go / No-Go
**GO** — the system detects seeded regressions with bounded error, cuts required human
labels by ~75% via PPI, exposes judge bias its agreement score would hide, and gates
releases in CI. Remaining work before production: real human labels in place of the
ground-truth oracle, a hosted/Ollama judge, and larger real corpora.
