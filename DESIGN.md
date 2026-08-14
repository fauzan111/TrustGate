# TrustGate — Design Document

**Risk-Controlled Evaluation & Release-Gating for LLM / RAG / Agent systems.**

TrustGate is a standalone, black-box service. A user registers any endpoint (or uploads
traces), uploads a versioned test set, and receives **calibrated** quality estimates with
**confidence intervals**, per-slice failure analysis, and a statistically-powered
**ship / investigate / block** decision — using as **few human labels as possible**.

Research question (the headline result):
> *How many human labels does a calibrated judge need to detect a seeded 5–10% regression,
> without excessive false-ship or false-block decisions?*

---

## 1. Core concepts (the domain model)

| Concept | Meaning |
|---|---|
| **SUT** (System Under Test) | A registered endpoint or trace-producer being evaluated. |
| **Item** | One test case: `{id, task_type, input, references, tags(slices), split}`. |
| **Dataset** | Immutable, content-hashed, versioned collection of Items + license + lineage. |
| **Run** | Executing a SUT over a Dataset version → produces **Outputs** (answers / trajectories). |
| **Evaluator** | Scores Outputs → per-item, per-metric **Scores**. |
| **Judge** | An LLM-as-judge Evaluator, wrapped with bias controls. |
| **LabelSet** | Human ground-truth labels used to *calibrate* cheap judge scores. |
| **Estimate** | Aggregate metric + confidence interval, calibrated with human labels. |
| **Decision** | ship / investigate / block, comparing candidate vs. baseline, with bounded error. |

### Task types (one typed contract for all)
1. **free-form generation**
2. **structured output** (JSON-schema validated)
3. **retrieval / RAG** (context + citations)
4. **tool trajectory** (agent: sequence of steps + tool calls)
5. **pairwise preference**

---

## 2. Architecture (components)

```
                 ┌──────────────┐
   test set  ───▶│   REGISTRY   │  datasets · SUTs · runs · evaluators · labels
                 │ (SQLite +    │  immutable versions via content hash
                 │  Parquet)    │  contamination checks · dev/hidden splits
                 └──────┬───────┘
                        │
      ┌─────────────────┼───────────────────────────────┐
      ▼                 ▼                                ▼
┌───────────┐   ┌───────────────┐               ┌────────────────┐
│ ADAPTERS  │   │ EVALUATOR BANK│               │  JUDGE LAB     │
│ call SUT, │──▶│ deterministic │◀──────────────│ rubric versions│
│ collect   │   │ embedding/NLP │   judge type  │ position swap  │
│ outputs/  │   │ retrieval     │               │ repeat trials  │
│ traces    │   │ llm-judge     │               │ length-bias    │
└───────────┘   │ human-review  │               │ bias correction│
                └──────┬────────┘               └────────────────┘
                       │ per-item scores
                       ▼
              ┌──────────────────┐        ┌──────────────────┐
              │ SAMPLING ENGINE  │◀──────▶│  CALIBRATION &   │
              │ random/stratified│ labels │  ESTIMATION      │
              │ /uncertainty     │ budget │ CIs · PPI ·      │
              └────────┬─────────┘        │ isotonic/Platt   │
                       │                  └────────┬─────────┘
                       ▼                           ▼
                 ┌────────────────────────────────────────┐
                 │            DECISION ENGINE              │
                 │ conformal risk control · power analysis │
                 │ → ship / investigate / block + evidence │
                 └───────────────────┬────────────────────┘
                                     │
          ┌──────────────┬───────────┴──────────┬───────────────┐
          ▼              ▼                      ▼               ▼
     FastAPI API   Typer CLI          Streamlit dashboard   GitHub Action
                                                            (PR release gate)
```

### Component responsibilities

1. **Registry** — storage + versioning. SQLite for metadata, Parquet for item/score
   artifacts. Every Dataset/Run gets a content hash → reproducibility. Handles dev/hidden
   splits and contamination checks (n-gram overlap between test items and known corpora).
2. **Adapters** — connect to a SUT. Ship three: `openai_compatible` (HTTP), `local`
   (Ollama), `trace_import` (offline JSONL of pre-computed outputs/trajectories). All
   behind one `SUTAdapter` interface.
3. **Evaluator bank** — all evaluators implement one `Evaluator.score(item, output) ->
   list[Score]` contract:
   - *deterministic*: exact-match, regex, JSON-schema validity, numeric tolerance.
   - *embedding/NLP*: semantic similarity, ROUGE/BLEU where relevant.
   - *retrieval*: Recall@k, nDCG, citation precision, faithfulness/groundedness.
   - *llm-judge*: rubric-scored (delegates to Judge Lab).
   - *human-review*: pulls from LabelSet.
4. **Judge Lab** — the differentiator. Wraps any judge model with: rubric versioning,
   position-swap (A/B order flip), repeated trials (self-consistency), length-bias probe,
   self-preference probe, and **bias correction** + measured judge-vs-human agreement
   (Cohen's κ / balanced accuracy).
5. **Sampling engine** — chooses *which* items to send for human labeling under a fixed
   budget: `random`, `stratified` (by slice), `uncertainty` (active — label where the
   judge is least confident). This produces the label-efficiency curve.
6. **Calibration & estimation** — the statistical spine:
   - Calibrate judge scores against human labels (Platt / isotonic).
   - **Prediction-Powered Inference (PPI)**: combine many cheap judge scores + few human
     labels into a *valid, tighter* confidence interval than labels alone. This is the
     modern, rigorous, label-efficient method and your core research lever.
   - Report estimate ± CI, calibration error (ECE), estimator bias.
7. **Decision engine** — compares candidate vs. baseline:
   - **Conformal risk control**: bound the false-decision rate (distribution-free).
   - Power analysis: given effect size (e.g. 5–10% drop) + variance → min samples/labels.
   - Emits `ship / investigate / block` + an evidence bundle.
8. **API / CLI / Dashboard / GitHub Action** — surfaces. The Action is what makes it
   *usable by others*: one YAML step fails a PR when quality regresses.

---

## 3. The benchmark: EvalMix-500

A reusable, licence-clean benchmark you author (portfolio gold — citable, hard to fake):

- **150** open-generation · **150** RAG (over a *fresh* neutral corpus) · **100** synthetic
  tool trajectories · **100** adversarial/shift cases.
- **20% held out hidden** for contamination-safe final scoring.
- **Seeded known failures** so detection power is measurable: swapped citations, retrieval
  omissions, verbosity inflation, schema errors, subtle tool misuse, unjustified refusals.

Start with a **60-item seed** (20 generation / 20 retrieval / 20 tool-trajectory) in weeks 1–2,
scale to 500 in week 11.

---

## 4. Tech stack (all free / local-friendly, no training GPU)

| Layer | Choice |
|---|---|
| Language / packaging | Python 3.11, `uv` (or Poetry) |
| Schemas / config | Pydantic v2 |
| API | FastAPI + Uvicorn |
| CLI | Typer |
| Storage | SQLite (metadata) + DuckDB/Parquet (artifacts) |
| Embeddings | sentence-transformers (local) |
| RAG metrics | RAGAS / DeepEval (optional, behind adapters) |
| Agent/trajectory eval | Inspect AI |
| Judge models | Ollama (local, default) + optional hosted (OpenAI/Anthropic) |
| Statistics | numpy, scipy, statsmodels, `ppi-py`, scikit-learn (isotonic/Platt) |
| Dashboard | Streamlit (v1) → Next.js (later, optional) |
| CI | GitHub Actions (composite action) |
| Quality | pytest, ruff, mypy |

---

## 5. Repository structure

```
trustgate/
  pyproject.toml
  README.md
  DESIGN.md
  src/trustgate/
    config.py                 # pydantic settings
    models/                   # Item, Dataset, Run, Score, LabelSet, Estimate, Decision
    registry/                 # sqlite + parquet, versioning, contamination checks
    adapters/                 # openai_compatible, local(ollama), trace_import
    tasks/                    # task-type validators
    evaluators/               # base, deterministic, embedding, retrieval, judge, human
    judge_lab/                # bias probes, judge calibration, agreement
    sampling/                 # random, stratified, uncertainty
    estimation/               # calibration, PPI, confidence intervals
    decision/                 # conformal risk control, power, ship/block
    api/                      # fastapi routes
    report/                   # markdown/html report + evidence index
    cli.py                    # typer entrypoint
  dashboard/                  # streamlit app
  benchmarks/evalmix/         # dataset build scripts + dataset cards
  examples/                   # runnable end-to-end demos
  tests/
  .github/
    workflows/ci.yml
    actions/trustgate/action.yml   # the reusable release-gate action
```

---

## 6. Milestone roadmap (12 weeks, ~12–15 h/week)

| Weeks | Milestone | "Done" acceptance test |
|---|---|---|
| **0** | **Walking skeleton** | `trustgate run` executes a mock SUT over 5 items with a deterministic evaluator and prints a naive estimate + decision. Pipeline typed end-to-end. |
| **1–2** | Evaluator interface frozen + 60 seed cases | Human rubric written *before* any LLM judge; 2 blind raters; disagreements adjudicated. |
| **3–4** | Registry + deterministic/NLP/retrieval evaluators | Dataset versioning + contamination check; retrieval metrics (Recall@k, nDCG, citation precision) on RAG items. |
| **5–6** | Judge Lab | Judge with position-swap + repeat trials + length-bias probe; measured judge↔human agreement + bias correction. |
| **7–8** | Sampling + calibration/estimation | random vs stratified vs uncertainty under fixed budgets; calibrated estimates + CIs via PPI; the **label-efficiency curve**. |
| **9–10** | Conformal decision engine + GitHub Action | ship/investigate/block with bounded error; Action fails a PR on a seeded regression. |
| **11** | Scale to EvalMix-500 + agent trajectories | full benchmark; end-state + trajectory metrics; slice failure report. |
| **12** | Report + demo + tagged release | technical report ending in go/no-go; 3-min demo; evidence index (every claim → command + artifact). |

---

## 7. First concrete steps (this week)

1. Scaffold repo + `pyproject.toml`; create the Pydantic models (`Item`, `Dataset`,
   `Run`, `Score`, `Estimate`, `Decision`).
2. Build the **walking skeleton**: mock SUT → deterministic evaluator → naive mean
   estimate → threshold decision. Get `pytest` green.
3. Write the **human rubric**, then hand-author the 60 seed cases as JSONL.
4. Label blind, adjudicate, stand up deterministic baselines. *Only then* add judges.

---

## 8. Portfolio-credibility rules (keep these)

- Only **new** public/synthetic data — no energy/insurance/invoice/support/sentiment domains.
- One scientific question, one benchmark, one decision report.
- Separate hidden final tests; publish contamination checks.
- Report negative results, uncertainty, cost, latency, failure slices.
- Put a measured number on the CV **only after** a tagged release reproduces it.
- Add an **evidence index** so a reviewer verifies your strongest claim in < 5 minutes.
