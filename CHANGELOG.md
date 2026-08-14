# Changelog

## 0.1.0 — first flagship release

The full evaluation-and-release-gating pipeline, buildable and testable offline (no GPU, no
model server). 49 tests passing.

### Added
- **Typed core** — `Item / Dataset / Run / Output / Score / Estimate / Decision` (Pydantic).
- **Pipeline** — `run → evaluate → estimate → decide`, with a mock SUT and Wilson CIs.
- **EvalMix seed set** — 30 hand-authored cases (generation / retrieval / tool-trajectory),
  20% hidden, with seeded failures and abstain/escalate safety cases, plus a
  human-first scoring **RUBRIC** and a blind-labeling sheet generator.
- **Registry** — immutable, content-addressed SQLite persistence; word n-gram
  contamination checks for hidden↔dev leakage.
- **Evaluators** — deterministic (exact match); retrieval (answer, citation precision/recall,
  Recall@k, nDCG@k, lexical groundedness); tool-trajectory (goal, tool-call correctness,
  efficiency, safety gate).
- **Judge Lab** — `JudgeModel` interface with `SimulatedJudge` (known biases) and
  `OllamaJudge` (drop-in local model); probes for length bias, position bias,
  self-consistency; judge↔human agreement; threshold calibration.
- **Sampling + PPI** — random / stratified / uncertainty sampling; naive / classical /
  Prediction-Powered-Inference estimators; the label-efficiency experiment
  (**~50–75% fewer labels** for equal precision; documents the uncertainty-sampling
  coverage failure mode).
- **Conformal release gate** — non-inferiority decision on the quality drop (`normal` and
  distribution-free `hoeffding` CIs) with a bounded error rate; a **GitHub Action** that
  fails a PR on a regression.
- **EvalMix-500 builder**, **technical report generator** with an evidence index, and CLI:
  `demo, validate, stats, labeling-template, contamination, ingest, rag-demo, agent-demo,
  judge-lab, label-efficiency, gate-ci, build-evalmix, report`.

### Known limitations
- "Human" labels in demos use a ground-truth oracle until the labeling sheet is filled.
- Label-efficiency numbers are from a controlled simulation study.
- Groundedness is a lexical proxy pending the LLM-judge groundedness check.
