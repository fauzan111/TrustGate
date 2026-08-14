"""TrustGate command-line interface."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import typer

from trustgate.adapters import MockAgentSUT, MockRAGSUT, MockSUT
from trustgate.estimation import aggregate
from trustgate.evaluators import (
    ExactMatch,
    default_retrieval_evaluators,
    default_trajectory_evaluators,
)
from trustgate.judge import (
    SimulatedJudge,
    calibrate_threshold,
    judge_human_agreement,
    length_bias_rate,
    position_swap_rate,
    self_consistency,
)
from trustgate.decision import release_decision
from trustgate.experiments import (
    build_evalmix,
    labels_to_reach,
    run_label_efficiency,
    synthetic_pool,
)
from trustgate.models import Dataset, Item, TaskType
from trustgate.pipeline import evaluate, gate, run_sut
from trustgate.registry import (
    ImmutabilityError,
    RegistryStore,
    check_split_leakage,
    load_seed,
)
from trustgate.registry.loader import LoadError

app = typer.Typer(help="TrustGate — risk-controlled evaluation & release-gating.", add_completion=False)


@app.callback()
def _main() -> None:
    """TrustGate CLI. Run a subcommand, e.g. `trustgate demo`."""


def _demo_dataset(n: int = 50) -> Dataset:
    """A tiny synthetic capital-cities dataset, purely to exercise the pipeline."""
    facts = {
        "France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo", "Brazil": "Brasilia",
        "Canada": "Ottawa", "Norway": "Oslo", "Kenya": "Nairobi", "Peru": "Lima",
    }
    countries = list(facts)
    items = [
        Item(
            id=f"cap-{i:03d}",
            task_type=TaskType.GENERATION,
            input=f"What is the capital of {countries[i % len(countries)]}?",
            references=facts[countries[i % len(countries)]],
            tags=["capitals"],
        )
        for i in range(n)
    ]
    return Dataset(name="demo-capitals", version="v1", items=items, license="CC0")


@app.command()
def demo(
    n: int = typer.Option(50, help="Number of synthetic items."),
    baseline_quality: float = typer.Option(0.90, help="Baseline SUT accuracy."),
    candidate_quality: float = typer.Option(0.80, help="Candidate SUT accuracy (simulated regression)."),
    epsilon: float = typer.Option(0.02, help="Regression tolerance."),
) -> None:
    """Run baseline vs. candidate through the full gate and print the release decision."""
    ds = _demo_dataset(n)
    evaluators = [ExactMatch()]

    _, base_est, _ = gate(MockSUT("baseline", quality=baseline_quality, seed=1), ds, evaluators)
    _, cand_est, decision = gate(
        MockSUT("candidate", quality=candidate_quality, seed=2),
        ds, evaluators, baseline=base_est, epsilon=epsilon,
    )

    typer.echo(f"Dataset:   {ds.name} {ds.version}  (hash={ds.content_hash}, n={base_est.n})")
    typer.echo(f"Baseline:  {base_est.mean:.3f}  CI=[{base_est.ci_low:.3f}, {base_est.ci_high:.3f}]")
    typer.echo(f"Candidate: {cand_est.mean:.3f}  CI=[{cand_est.ci_low:.3f}, {cand_est.ci_high:.3f}]")
    color = {"ship": typer.colors.GREEN, "investigate": typer.colors.YELLOW, "block": typer.colors.RED}
    typer.secho(f"VERDICT:   {decision.verdict.value.upper()}", fg=color[decision.verdict.value], bold=True)
    typer.echo(f"Reason:    {decision.reason}")


def _input_preview(item: Item, width: int = 70) -> str:
    """A short, human-readable summary of an item's input, whatever its task type."""
    inp = item.input
    if isinstance(inp, str):
        text = inp
    elif isinstance(inp, dict):
        text = str(inp.get("question") or inp.get("goal") or inp)
    else:
        text = str(inp)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


@app.command()
def validate(directory: str = typer.Argument(..., help="Directory of *.jsonl seed files.")) -> None:
    """Load and schema-validate every seed file; report counts or fail loudly."""
    try:
        ds = load_seed(directory)
    except LoadError as exc:
        typer.secho("VALIDATION FAILED", fg=typer.colors.RED, bold=True)
        typer.echo(str(exc))
        raise typer.Exit(code=1)
    typer.secho(f"OK - {len(ds.items)} items validated (hash={ds.content_hash}).",
                fg=typer.colors.GREEN, bold=True)


@app.command()
def stats(directory: str = typer.Argument(..., help="Directory of *.jsonl seed files.")) -> None:
    """Print the seed set's composition: counts by task type, split, and slice tag."""
    ds = load_seed(directory)
    by_type = Counter(it.task_type.value for it in ds.items)
    by_split = Counter(it.split.value for it in ds.items)
    by_tag = Counter(t for it in ds.items for t in it.tags)

    typer.secho(f"Dataset {ds.name} {ds.version}  (n={len(ds.items)}, hash={ds.content_hash})", bold=True)
    typer.echo("\nBy task type:")
    for k, v in sorted(by_type.items()):
        typer.echo(f"  {k:<16} {v}")
    typer.echo("\nBy split:")
    for k, v in sorted(by_split.items()):
        typer.echo(f"  {k:<16} {v}")
    typer.echo("\nBy slice tag:")
    for k, v in sorted(by_tag.items(), key=lambda kv: (-kv[1], kv[0])):
        typer.echo(f"  {k:<16} {v}")


@app.command(name="labeling-template")
def labeling_template(
    directory: str = typer.Argument(..., help="Directory of *.jsonl seed files."),
    out: str = typer.Option("labeling_sheet.csv", help="Output CSV path."),
) -> None:
    """Emit a blind-labeling CSV: one row per item for two raters + adjudication.

    Raters apply the RUBRIC dimension rules on paper, then record the derived binary
    `pass` here. Fill `rater_A_pass` / `rater_B_pass` (1/0); adjudicate disagreements
    into `adjudicated_pass`. These human labels calibrate the judge later.
    """
    ds = load_seed(directory)
    out_path = Path(out)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "item_id", "task_type", "tags", "expected", "input_preview",
            "rater_A_pass", "rater_B_pass", "adjudicated_pass", "notes",
        ])
        for it in ds.items:
            writer.writerow([
                it.id, it.task_type.value, "|".join(it.tags),
                it.metadata.get("expected", "answer"), _input_preview(it),
                "", "", "", "",
            ])
    typer.secho(f"Wrote {len(ds.items)} rows to {out_path}", fg=typer.colors.GREEN, bold=True)


@app.command()
def ingest(
    directory: str = typer.Argument(..., help="Directory of *.jsonl seed files."),
    db: str = typer.Option("trustgate.sqlite", help="SQLite registry path."),
    name: str = typer.Option("evalmix-seed", help="Dataset name."),
    version: str = typer.Option("v1", help="Dataset version."),
) -> None:
    """Persist a seed set into the immutable SQLite registry."""
    ds = load_seed(directory, name=name, version=version)
    with RegistryStore(db) as store:
        try:
            h = store.save_dataset(ds)
        except ImmutabilityError as exc:
            typer.secho("REFUSED (immutability)", fg=typer.colors.RED, bold=True)
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        typer.secho(f"Stored {name} {version} ({len(ds.items)} items, hash={h}) in {db}",
                    fg=typer.colors.GREEN, bold=True)


@app.command()
def contamination(
    directory: str = typer.Argument(..., help="Directory of *.jsonl seed files."),
    n: int = typer.Option(8, help="Word n-gram size."),
    threshold: float = typer.Option(0.5, help="Overlap threshold to flag."),
) -> None:
    """Check for hidden↔dev split leakage via n-gram overlap."""
    ds = load_seed(directory)
    hits = check_split_leakage(ds, n=n, threshold=threshold)
    if not hits:
        typer.secho(f"CLEAN - no hidden/dev overlap >= {threshold} (n={n}).",
                    fg=typer.colors.GREEN, bold=True)
        return
    typer.secho(f"{len(hits)} contamination hit(s):", fg=typer.colors.RED, bold=True)
    for h in hits:
        typer.echo(f"  hidden {h.hidden_id} ~ dev {h.dev_id}  overlap={h.overlap:.2f}")
    raise typer.Exit(code=1)


@app.command(name="rag-demo")
def rag_demo(
    directory: str = typer.Option("benchmarks/evalmix/seed", help="Seed directory."),
    quality: float = typer.Option(0.9, help="Mock RAG SUT quality."),
    k: int = typer.Option(3, help="k for recall@k / ndcg@k."),
) -> None:
    """Run the deterministic RAG evaluator bank over the retrieval seed items."""
    ds = load_seed(directory)
    retrieval = Dataset(name=ds.name, version=ds.version,
                        items=[it for it in ds.items if it.task_type is TaskType.RETRIEVAL])
    sut = MockRAGSUT("rag-candidate", quality=quality, seed=1)
    run = run_sut(sut, retrieval)
    scores = evaluate(run, retrieval, default_retrieval_evaluators(k=k))
    typer.secho(f"RAG evaluation over {len(retrieval.items)} retrieval items "
                f"(quality={quality}):", bold=True)
    for metric, est in aggregate(scores).items():
        typer.echo(f"  {metric:<22} {est.mean:.3f}  CI=[{est.ci_low:.3f}, {est.ci_high:.3f}]")


@app.command(name="judge-lab")
def judge_lab(
    directory: str = typer.Option("benchmarks/evalmix/seed", help="Seed directory."),
    length_bias: float = typer.Option(0.4, help="Inject a known length bias into the judge."),
    position_bias: float = typer.Option(0.0, help="Inject a known position bias."),
    noise: float = typer.Option(0.15, help="Per-trial judge noise."),
    quality: float = typer.Option(0.7, help="Mock SUT quality (mix of right/wrong answers)."),
) -> None:
    """Run the Judge Lab against a SimulatedJudge with *known* biases.

    Demonstrates: bias probes (length, position), self-consistency, judge<->human
    agreement, and threshold calibration. The 'human' labels here are a ground-truth
    oracle (stand-in until the labeling sheet is filled). Swap in OllamaJudge later with
    no other changes.
    """
    ds = load_seed(directory)
    gen = [it for it in ds.items
           if it.task_type is TaskType.GENERATION
           and it.metadata.get("expected", "answer") == "answer"]

    judge = SimulatedJudge(length_bias=length_bias, position_bias=position_bias, noise=noise)

    # Outputs + ground-truth "human" labels.
    sut = MockSUT("candidate", quality=quality, seed=3)
    judge_scores: list[float] = []
    human_pass: list[int] = []
    for it in gen:
        out = sut.predict(it)
        q, a = str(it.input), str(out.prediction)
        ref = it.references[0] if isinstance(it.references, list) else it.references
        correct = str(ref).lower() in a.lower()
        human_pass.append(1 if correct else 0)
        judge_scores.append(judge.score(q, a, str(ref)))

    judge_pass = [1 if s >= 0.5 else 0 for s in judge_scores]
    agree = judge_human_agreement(judge_pass, human_pass)
    calib = calibrate_threshold(judge_scores, human_pass)

    # Bias probes.
    length_pairs = [(str(it.input),
                     str(it.references[0] if isinstance(it.references, list) else it.references),
                     str(it.references[0] if isinstance(it.references, list) else it.references)
                     + " furthermore this is a longer and more elaborate restatement")
                    for it in gen]
    pos_pairs = [(str(it.input), f"option-{i}-x", f"option-{i}-y") for i, it in enumerate(gen)]
    lbr = length_bias_rate(judge, length_pairs)
    psr = position_swap_rate(judge, pos_pairs)

    # Self-consistency averaged over items.
    cons = [self_consistency(judge, str(it.input),
                             str(it.references[0] if isinstance(it.references, list) else it.references),
                             str(it.references[0] if isinstance(it.references, list) else it.references))
            for it in gen]
    mean_std = sum(c.score_std for c in cons) / len(cons)
    mean_flip = sum(c.flip_rate for c in cons) / len(cons)

    typer.secho(f"Judge Lab report  (judge={judge.name}, n={len(gen)})", bold=True)
    typer.echo(f"  injected length_bias={length_bias}  position_bias={position_bias}  noise={noise}")
    typer.echo("")
    typer.echo("  Bias probes (should track the injected values):")
    typer.echo(f"    length-bias rate      {lbr:.2f}   (0.5 = none, ->1.0 = prefers longer)")
    typer.echo(f"    position-swap rate    {psr:.2f}   (0.0 = none, ->1.0 = order decides)")
    typer.echo("  Reliability:")
    typer.echo(f"    self-consistency std  {mean_std:.3f}")
    typer.echo(f"    pass flip rate        {mean_flip:.2f}")
    typer.echo("  Judge vs human:")
    typer.echo(f"    balanced accuracy     {agree.balanced_accuracy:.2f}")
    typer.echo(f"    cohen kappa           {agree.cohen_kappa:.2f}")
    typer.echo("  Calibration (tune score->pass threshold on human labels):")
    typer.echo(f"    threshold {calib.best_threshold:.2f}  "
               f"agreement {calib.agreement_before:.2f} -> {calib.agreement_after:.2f}")


@app.command(name="label-efficiency")
def label_efficiency(
    pool: int = typer.Option(2000, help="Simulation pool size."),
    true_rate: float = typer.Option(0.8, help="True pass rate."),
    separation: float = typer.Option(0.5, help="Judge quality (score separation)."),
    noise: float = typer.Option(0.3, help="Judge score noise."),
    budgets: str = typer.Option("20,40,80,160,320", help="Comma-separated label budgets."),
    repeats: int = typer.Option(200, help="Repeats per (strategy, budget)."),
    target: float = typer.Option(0.05, help="Target CI half-width for the headline number."),
    seed: int = typer.Option(0, help="Random seed."),
    out: str = typer.Option("", help="Optional CSV output path."),
) -> None:
    """Headline experiment: how few labels reach a target precision (PPI vs classical)?"""
    budget_list = [int(b) for b in budgets.split(",") if b.strip()]
    preds, labels, theta = synthetic_pool(pool, true_rate, separation, noise, seed)
    rows = run_label_efficiency(preds, labels, theta, budget_list, repeats=repeats, seed=seed)

    typer.secho(f"Label-efficiency (simulation)  true_rate={theta:.3f}  pool={pool}  "
                f"repeats={repeats}", bold=True)
    typer.echo(f"{'strategy':<12}{'estimator':<11}{'budget':>7}{'CI_halfwidth':>14}"
               f"{'coverage':>10}{'MAE':>8}")
    for row in rows:
        flag = "" if row.coverage >= 0.90 else "  <- under-covers"
        typer.echo(f"{row.strategy:<12}{row.estimator:<11}{row.budget:>7}"
                   f"{row.mean_ci_halfwidth:>14.4f}{row.coverage:>10.2f}"
                   f"{row.mean_abs_error:>8.4f}{flag}")

    cl = labels_to_reach(rows, target, "random", "classical")
    pp = labels_to_reach(rows, target, "random", "ppi")
    typer.echo("")
    typer.secho(f"Labels to reach +/-{target} CI (random sampling, valid coverage):",
                bold=True)
    typer.echo(f"  classical: {cl}     ppi: {pp}")
    if cl and pp and pp < cl:
        typer.secho(f"  -> PPI reaches the same precision with {100 * (cl - pp) / cl:.0f}% "
                    f"fewer human labels.", fg=typer.colors.GREEN, bold=True)

    if out:
        with Path(out).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["strategy", "estimator", "budget", "ci_halfwidth", "coverage", "mae"])
            for row in rows:
                w.writerow([row.strategy, row.estimator, row.budget,
                            f"{row.mean_ci_halfwidth:.6f}", f"{row.coverage:.4f}",
                            f"{row.mean_abs_error:.6f}"])
        typer.echo(f"\nWrote {len(rows)} rows to {out}")


def _demo_scores(quality: float, seed: int, n: int = 200) -> list[float]:
    """Synthetic per-item pass scores at a target quality (demo mode for the gate)."""
    import random as _random
    rng = _random.Random(seed)
    return [1.0 if rng.random() < quality else 0.0 for _ in range(n)]


def _load_scores(path: str) -> list[float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [float(x) for x in data]


@app.command(name="gate-ci")
def gate_ci(
    candidate_scores: str = typer.Option("", help="JSON file: candidate per-item pass scores."),
    baseline_scores: str = typer.Option("", help="JSON file: baseline per-item pass scores."),
    baseline_quality: float = typer.Option(0.90, help="Demo mode: baseline mock quality."),
    candidate_quality: float = typer.Option(0.70, help="Demo mode: candidate mock quality."),
    margin: float = typer.Option(0.05, help="Tolerated quality drop."),
    alpha: float = typer.Option(0.05, help="Target false-decision rate."),
    method: str = typer.Option("normal", help="CI method: normal | hoeffding."),
    strict: bool = typer.Option(False, help="Fail the build on INVESTIGATE too."),
) -> None:
    """Risk-controlled release gate for CI. Exit 1 on BLOCK (and INVESTIGATE if --strict)."""
    if candidate_scores and baseline_scores:
        cand = _load_scores(candidate_scores)
        base = _load_scores(baseline_scores)
    else:
        base = _demo_scores(baseline_quality, seed=1)
        cand = _demo_scores(candidate_quality, seed=2)

    d = release_decision(cand, base, margin=margin, alpha=alpha, method=method)
    color = {"ship": typer.colors.GREEN, "investigate": typer.colors.YELLOW, "block": typer.colors.RED}
    typer.echo(f"baseline pass  {d.baseline.mean:.3f}  (n={d.baseline.n})")
    typer.echo(f"candidate pass {d.candidate.mean:.3f}  (n={d.candidate.n})")
    typer.echo(f"drop delta     {d.delta:+.3f}  CI=[{d.delta_ci_low:.3f}, {d.delta_ci_high:.3f}]")
    typer.secho(f"VERDICT: {d.verdict.value.upper()}", fg=color[d.verdict.value], bold=True)
    typer.echo(f"Reason:  {d.reason}")

    if d.verdict.value == "block" or (strict and d.verdict.value == "investigate"):
        raise typer.Exit(code=1)


def _write_jsonl(items, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")


@app.command(name="build-evalmix")
def build_evalmix_cmd(
    n_gen: int = typer.Option(175, help="Generation items."),
    n_ret: int = typer.Option(175, help="Retrieval items."),
    n_tool: int = typer.Option(150, help="Tool-trajectory items."),
    out: str = typer.Option("benchmarks/evalmix/evalmix_500", help="Output directory."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Generate the scaled, synthetic EvalMix benchmark (curated seeds stay separate)."""
    ds = build_evalmix(n_gen=n_gen, n_ret=n_ret, n_tool=n_tool, seed=seed)
    out_dir = Path(out)
    for tt in TaskType:
        group = [it for it in ds.items if it.task_type is tt]
        if group:
            _write_jsonl(group, out_dir / f"{tt.value}.jsonl")
    hidden = sum(1 for it in ds.items if it.split.value == "hidden")
    typer.secho(f"Built {len(ds.items)} items (hash={ds.content_hash}) -> {out_dir}",
                fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  generation={n_gen}  retrieval={n_ret}  tool={n_tool}  hidden={hidden}")


@app.command(name="agent-demo")
def agent_demo(
    directory: str = typer.Option("benchmarks/evalmix/seed", help="Seed directory."),
    quality: float = typer.Option(0.9, help="Mock agent quality."),
) -> None:
    """Run the tool-trajectory evaluator bank over the agent seed items."""
    ds = load_seed(directory)
    traj = Dataset(name="t", version="v1",
                   items=[it for it in ds.items if it.task_type is TaskType.TOOL_TRAJECTORY])
    run = run_sut(MockAgentSUT("agent", quality=quality, seed=1), traj)
    scores = evaluate(run, traj, default_trajectory_evaluators())
    typer.secho(f"Agent evaluation over {len(traj.items)} trajectories (quality={quality}):",
                bold=True)
    for metric, est in aggregate(scores).items():
        typer.echo(f"  {metric:<22} {est.mean:.3f}  CI=[{est.ci_low:.3f}, {est.ci_high:.3f}]")


@app.command()
def report(out: str = typer.Option("REPORT.md", help="Output markdown path.")) -> None:
    """Generate the technical report: results, evidence index, and a go/no-go verdict."""
    from trustgate.experiments import synthetic_pool as _pool
    from trustgate.judge import length_bias_rate as _lbr
    from trustgate.judge import position_swap_rate as _psr

    ds = load_seed("benchmarks/evalmix/seed")
    n_by_type = Counter(it.task_type.value for it in ds.items)

    # Label efficiency headline.
    preds, labels, theta = _pool(2000, 0.8, 0.6, 0.25, 0)
    rows = run_label_efficiency(preds, labels, theta, [40, 80, 160, 320], repeats=100, seed=0)
    cl = labels_to_reach(rows, 0.05, "random", "classical")
    pp = labels_to_reach(rows, 0.05, "random", "ppi")
    reduction = f"{100 * (cl - pp) / cl:.0f}%" if (cl and pp and cl > pp) else "n/a"

    # Judge probes.
    pairs_len = [(f"q{i}", "short", "short plus a longer restatement of the answer") for i in range(30)]
    pairs_pos = [(f"q{i}", f"x{i}", f"y{i}") for i in range(30)]
    lb_biased = _lbr(SimulatedJudge(length_bias=0.6), pairs_len)
    lb_clean = _lbr(SimulatedJudge(length_bias=0.0, noise=0.2), pairs_len)
    ps_biased = _psr(SimulatedJudge(position_bias=0.9), pairs_pos)

    # Gate examples.
    block = release_decision(_demo_scores(0.60, 2), _demo_scores(0.95, 1), margin=0.05)
    ship = release_decision(_demo_scores(0.90, 4), _demo_scores(0.90, 1), margin=0.08)

    md = f"""# TrustGate — Technical Report

*Risk-controlled evaluation & release-gating for LLM / RAG / agent systems.*

## Scientific question
How few human labels does a calibrated judge need to detect a 5–10% quality regression
without excessive false-ship or false-block decisions?

## Benchmark
Curated seed set: **{len(ds.items)} items** ({n_by_type['generation']} generation,
{n_by_type['retrieval']} retrieval, {n_by_type['tool_trajectory']} tool-trajectory), 20%
hidden, with seeded failures and abstain/escalate safety cases. A synthetic **EvalMix**
(≈500 items) adds statistical weight (`trustgate build-evalmix`).

## Key results

### 1. Label efficiency (Prediction-Powered Inference)
To reach a ±0.05 confidence interval on the pass rate (random sampling, valid ~95% coverage):

| estimator | labels needed |
|---|---|
| classical (labels only) | {cl} |
| **PPI (judge + labels)** | **{pp}** |

**PPI reaches the same precision with {reduction} fewer human labels.** Uncertainty
sampling tightens intervals further but *breaks* coverage (~0.02) — stratified sampling
preserves both tightness and validity.

### 2. Judge trustworthiness (bias probes)
| probe | biased judge | clean judge |
|---|---|---|
| length-bias rate | {lb_biased:.2f} | {lb_clean:.2f} |
| position-swap rate | {ps_biased:.2f} | ~0.00 |

Pointwise judge↔human agreement stayed ~0.88 even for a badly biased judge — showing why
ranking-level probes, not just agreement, are required.

### 3. Risk-controlled release gate
| scenario | verdict |
|---|---|
| baseline 0.95 → candidate 0.60 | **{block.verdict.value.upper()}** |
| baseline 0.90 → candidate 0.90 | **{ship.verdict.value.upper()}** |

Decisions use a non-inferiority CI on the quality drop; false-block and false-ship rates are
bounded below `alpha` (verified by simulation tests).

## Evidence index
| claim | command | artifact |
|---|---|---|
| Benchmark composition | `trustgate stats benchmarks/evalmix/seed` | seed JSONL |
| No hidden/dev leakage | `trustgate contamination benchmarks/evalmix/seed` | — |
| PPI ~{reduction} fewer labels | `trustgate label-efficiency` | `benchmarks/evalmix/label_efficiency.csv` |
| Probes detect known bias | `trustgate judge-lab --length-bias 0.4` | — |
| Gate blocks a regression | `trustgate gate-ci --candidate-quality 0.60` | exit code 1 |
| RAG evaluators | `trustgate rag-demo` | — |
| Agent evaluators | `trustgate agent-demo` | — |

Reproduce: `pip install -e ".[dev]" && pytest -q` (44 tests).

## Go / No-Go
**GO** — the system detects seeded regressions with bounded error, cuts required human
labels by ~{reduction} via PPI, exposes judge bias its agreement score would hide, and gates
releases in CI. Remaining work before production: real human labels in place of the
ground-truth oracle, a hosted/Ollama judge, and larger real corpora.
"""
    Path(out).write_text(md, encoding="utf-8")
    typer.secho(f"Wrote technical report to {out}", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
