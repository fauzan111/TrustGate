# TrustGate — 3-minute demo

A guided tour of the evaluation-and-release-gating pipeline. Everything below runs offline
(no GPU, no API keys). Reproduce it end-to-end with:

```bash
pip install -e ".[dev]"
bash scripts/demo.sh
```

## Getting a shareable recording

Two ready-made artifacts are in the repo, both generated from a real run (no manual typing):

| file | how to (re)generate | use |
|---|---|---|
| `trustgate.cast` | `python scripts/make_cast.py` | upload to asciinema.org |
| `docs/trustgate.gif` | `python scripts/cast_to_gif.py trustgate.cast docs/trustgate.gif <font.ttf>` | embed in README (works on Windows) |

The GIF renderer is pure Python + Pillow (no Rust/agg/Docker needed) and uses any
monospace `.ttf` (e.g. JetBrains Mono). It's already embedded at the top of `README.md`.

**Share it as a link** (from Linux / macOS / WSL — the asciinema CLI needs a Unix pty and
does *not* run on native Windows):

```bash
pip install asciinema
asciinema upload trustgate.cast          # prints a shareable asciinema.org URL
```

**Or embed a GIF** (works anywhere, incl. Windows) using `agg`, asciinema's standalone
GIF generator (a single Rust binary — no pty needed):

```bash
agg trustgate.cast docs/trustgate.gif
```

Then paste the asciinema URL into the badge slot in `README.md`, or reference the GIF.

**Prefer a live recording?** Use WSL: `wsl`, then install the project and run
`asciinema rec -c "bash scripts/demo.sh"`.

---

## What you'll see

### 1. Benchmark composition
30 curated seed cases across three task types, 20% held out as a hidden split, with slice
tags (safety, abstain, distractor, multi-hop, …) reported separately.

```text
$ trustgate stats benchmarks/evalmix/seed
Dataset evalmix-seed v1  (n=30, hash=ca245da97e7b3909)
By task type:  generation 10 · retrieval 10 · tool_trajectory 10
By split:      dev 24 · hidden 6
```

### 2. Contamination check
Word n-gram overlap between the hidden and dev splits — proving the hidden set isn't a
near-duplicate of development data.

```text
$ trustgate contamination benchmarks/evalmix/seed
CLEAN - no hidden/dev overlap >= 0.5 (n=8).
```

### 3. Judge Lab — bias probes
Inject a *known* length bias into the judge; the probe detects it (rate → 1.00). This is
why the lab uses a simulated judge with known biases: you can prove the probe works.

```text
$ trustgate judge-lab --length-bias 0.4
  length-bias rate      1.00   (0.5 = none, ->1.0 = prefers longer)
  position-swap rate    0.00   (0.0 = none, ->1.0 = order decides)
```

Swap to a real local model with one flag once it's pulled: `trustgate judge-lab --judge ollama`.
Use real human labels with `--labels <sheet.csv>`.

### 4. Label efficiency — Prediction-Powered Inference (the headline)
```text
$ trustgate label-efficiency
Labels to reach +/-0.05 CI (random sampling, valid coverage):
  classical: 320     ppi: 160
  -> PPI reaches the same precision with 50% fewer human labels.
```

### 5. Risk-controlled release gate
A real regression is **BLOCKED** (exit code 1 → the PR fails); an equivalent candidate ships.

```text
$ trustgate gate-ci --baseline-quality 0.95 --candidate-quality 0.60
drop delta     +0.345  CI=[0.271, 0.419]
VERDICT: BLOCK
Reason:  drop CI lower 0.271 >= margin 0.05: regression. (alpha=0.05, method=normal).
```

---

Full technical report with an evidence index and go/no-go verdict: `trustgate report`.
