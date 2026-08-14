#!/usr/bin/env bash
# TrustGate — 3-minute guided demo.
# Record it with:  asciinema rec trustgate.cast -c "bash scripts/demo.sh"
# then upload:     asciinema upload trustgate.cast
set -e

banner() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; sleep 1; }

banner "1/8  What's in the benchmark? (10 gen / 10 retrieval / 10 tool, 20% hidden)"
trustgate stats benchmarks/evalmix/seed

banner "2/8  Is the hidden split contaminated by the dev split?"
trustgate contamination benchmarks/evalmix/seed

banner "3/8  RAG evaluators (answer, citations, Recall@k, nDCG, groundedness)"
trustgate rag-demo --quality 1.0

banner "4/8  Agent tool-trajectory evaluators (note the safety gate)"
trustgate agent-demo --quality 0.4

banner "5/8  Judge Lab: inject a KNOWN length bias -> the probe catches it (rate -> 1.0)"
trustgate judge-lab --length-bias 0.4

banner "6/8  The headline: Prediction-Powered Inference needs far fewer human labels"
trustgate label-efficiency

banner "7/8  Risk-controlled release gate — a real regression is BLOCKED (exit 1)"
trustgate gate-ci --baseline-quality 0.95 --candidate-quality 0.60 || echo "(exit code 1 => PR would fail)"

banner "8/8  ...and an equivalent candidate SHIPS (exit 0)"
trustgate gate-ci --baseline-quality 0.90 --candidate-quality 0.90

banner "Done. Full technical report:  trustgate report"
