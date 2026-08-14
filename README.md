# TrustGate

**Risk-controlled evaluation & release-gating for LLM / RAG / agent systems.**

Register any endpoint (or upload traces), give it a versioned test set, and get back
**calibrated quality estimates with confidence intervals**, per-slice failure analysis,
and a statistically-powered **ship / investigate / block** decision — using as **few human
labels as possible**.

> Research question: *How many human labels does a calibrated judge need to detect a seeded
> 5–10% regression without excessive false-ship or false-block decisions?*

See [`DESIGN.md`](DESIGN.md) for the full architecture and 12-week roadmap.

## Status

**Milestone 0 — walking skeleton (done).** A fully typed pipeline flows end to end:
`run → evaluate → estimate → decide`, with a mock SUT, deterministic evaluator, Wilson
confidence intervals, and a CI-bound-aware release gate.

**Milestone W1–2 — frozen interface + seed set (done).** The human scoring
[`RUBRIC.md`](benchmarks/evalmix/RUBRIC.md) is written *before any LLM judge*. 30 validated
seed cases (10 generation / 10 retrieval / 10 tool-trajectory, with abstention, distractor,
safety, and seeded-failure items) load through a schema-validating registry. A
blind-labeling sheet is generated for two raters + adjudication.

```bash
trustgate validate benchmarks/evalmix/seed        # schema-check every seed case
trustgate stats    benchmarks/evalmix/seed        # composition by type / split / slice
trustgate labeling-template benchmarks/evalmix/seed --out sheet.csv
```

**Milestone W3–4 — registry, contamination, RAG evaluators (done).** An immutable,
content-addressed SQLite registry (re-saving different content under the same version is
refused); word n-gram contamination checks for hidden↔dev split leakage; and the
deterministic RAG evaluator bank (answer correctness, citation precision/recall, Recall@k,
nDCG@k, lexical groundedness proxy).

```bash
trustgate contamination benchmarks/evalmix/seed   # flag hidden/dev leakage
trustgate ingest        benchmarks/evalmix/seed --db trustgate.sqlite   # immutable store
trustgate rag-demo --quality 1.0                  # run the RAG evaluator bank
```

**Milestone W5–6 — Judge Lab (done).** LLM-as-judge behind a pluggable `JudgeModel`
interface, with a `SimulatedJudge` (known, dial-in biases) for development/testing and an
`OllamaJudge` that drops in unchanged once a model is pulled locally. The lab measures
**length-bias** and **position-swap** rates, **self-consistency**, **judge↔human
agreement** (balanced accuracy, Cohen's κ), and **threshold calibration**.

```bash
trustgate judge-lab --length-bias 0.4                 # probe catches the length bias -> 1.00
trustgate judge-lab --position-bias 0.9 --length-bias 0  # position-swap rate -> 1.00
```

No Ollama required to build or test any of this — the simulated judge with *known* biases
is what lets the probes be verified. To use a real judge later:
`ollama pull llama3.1`, then construct `OllamaJudge()` in place of `SimulatedJudge()`.

**Milestone W7–8 — sampling + PPI + label-efficiency (done).** The statistical core.
Sampling strategies (random / stratified / uncertainty) and three estimators — naive
(biased), classical (labels-only), and **Prediction-Powered Inference (PPI)** — feed a
controlled label-efficiency experiment.

```bash
trustgate label-efficiency        # simulation: CI width & coverage vs label budget
```

Headline results (2000-item simulation, 200 repeats):

| strategy | estimator | labels | CI half-width | coverage |
|---|---|---:|---:|---:|
| random | classical | 320 | 0.044 | 0.97 |
| random | **ppi** | 160 | 0.043 | 0.97 |
| stratified | ppi | 160 | 0.043 | 0.99 |
| uncertainty | ppi | 160 | 0.062 | **0.02 ⚠** |

* **PPI reaches the same precision as classical with ~50% fewer human labels**, keeping
  valid ~95% coverage.
* **Uncertainty sampling breaks PPI's coverage** (it collapses to ~0.02) — a real failure
  mode most eval repos ignore. Stratified sampling keeps both tightness *and* validity.

**Milestone W9–10 — conformal gate + GitHub Action (done).** A risk-controlled release
decision built as a **non-inferiority test** on the quality *drop*: `ship` when the drop's
CI is below the tolerance, `block` when it's above, `investigate` when it straddles. Two CI
methods — `normal` (tight) and `hoeffding` (distribution-free, finite-sample guarantee) —
with a target error rate `alpha`. Shipped as a **GitHub Action** that fails a PR on a
regression.

```bash
trustgate gate-ci --baseline-quality 0.95 --candidate-quality 0.60   # BLOCK, exit 1
trustgate gate-ci --baseline-quality 0.90 --candidate-quality 0.90   # SHIP,  exit 0
```

The `.github/actions/trustgate` composite action + `.github/workflows/eval-gate.yml` wire
this into CI: evaluations gate merges just like unit tests. Simulation tests confirm the
false-block and false-ship rates stay bounded below `alpha`.

**Milestone W11–12 — agent evaluators, EvalMix-500, report, release (done).**
Tool-trajectory evaluators (goal completion, tool-call correctness, efficiency, and a hard
**safety gate**); a synthetic **EvalMix-500** builder; and a one-command **technical report**
with an evidence index and go/no-go verdict. Tagged `v0.1.0`.

```bash
trustgate agent-demo --quality 1.0            # tool-trajectory evaluator bank
trustgate build-evalmix                       # scale to ~500 items
trustgate report --out REPORT.md              # technical report + evidence index
```

**All 12 milestones complete — 49 tests passing, fully offline.**

## Quickstart

```bash
# from the trustgate/ directory
pip install -e .          # installs pydantic + typer
trustgate demo            # baseline 0.90 vs candidate 0.80 -> a release verdict
```

Example output:

```
Dataset:   demo-capitals v1  (hash=6d2a7179b96f568a, n=50)
Baseline:  0.840  CI=[0.715, 0.917]
Candidate: 0.860  CI=[0.738, 0.930]
VERDICT:   INVESTIGATE
Reason:    Interval straddles threshold: collect more labels. (delta=+0.020, eps=0.02).
```

At n=50 the sampling noise is wider than the tolerance, so the honest verdict is
*investigate* — the gate refuses to ship (or block) on noise. Give it a bigger sample and
a real regression and it will **BLOCK**:

```bash
trustgate demo --n 400 --baseline-quality 0.95 --candidate-quality 0.75
# VERDICT: BLOCK — Candidate upper bound < baseline - eps: regression detected. (delta=-0.193)
```

## Run the tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Layout

```
src/trustgate/
  models/       typed domain: Item, Dataset, Run, Score, Estimate, Decision
  adapters/     SUT connectors (MockSUT now; ollama / http / trace-import next)
  evaluators/   ExactMatch now; embedding / retrieval / judge next
  estimation/   Wilson CI now; Prediction-Powered Inference (PPI) next
  decision/     CI-bound gate now; conformal risk control next
  pipeline.py   run → evaluate → estimate → decide
  cli.py        `trustgate demo`
```

License: MIT (see `LICENSE`).
