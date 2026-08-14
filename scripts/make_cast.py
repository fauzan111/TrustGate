"""Generate a real asciicast v2 file from the TrustGate demo — Windows-friendly.

asciinema's own recorder needs a Unix pty and won't run on native Windows. This script
runs the demo commands, captures their genuine output, and writes a valid `trustgate.cast`
(asciicast v2). You can then, from any machine:

    asciinema upload trustgate.cast          # share on asciinema.org
    agg trustgate.cast trustgate.gif         # or convert to a GIF for the README

Usage:  python scripts/make_cast.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WIDTH, HEIGHT = 100, 34
PROMPT = "[1;32mtrustgate[0m$ "   # green prompt

STEPS: list[tuple[str, list[str]]] = [
    ("Benchmark composition", ["trustgate", "stats", "benchmarks/evalmix/seed"]),
    ("Contamination check", ["trustgate", "contamination", "benchmarks/evalmix/seed"]),
    ("RAG evaluators", ["trustgate", "rag-demo", "--quality", "1.0"]),
    ("Agent evaluators (safety gate)", ["trustgate", "agent-demo", "--quality", "0.4"]),
    ("Judge Lab: known length bias -> probe catches it", ["trustgate", "judge-lab", "--length-bias", "0.4"]),
    ("Label efficiency (PPI)", ["trustgate", "label-efficiency"]),
    ("Release gate: regression -> BLOCK (exit 1)",
     ["trustgate", "gate-ci", "--baseline-quality", "0.95", "--candidate-quality", "0.60"]),
    ("Release gate: equivalent -> SHIP (exit 0)",
     ["trustgate", "gate-ci", "--baseline-quality", "0.90", "--candidate-quality", "0.90"]),
]


def _crlf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    events: list[list] = []
    t = 0.0

    def emit(text: str, dt: float) -> None:
        nonlocal t
        t += dt
        events.append([round(t, 3), "o", text])

    emit("[1;36m# TrustGate — evaluation & release-gating demo[0m\r\n\r\n", 0.5)

    for title, cmd in STEPS:
        emit(f"[1;36m# {title}[0m\r\n", 0.8)
        emit(PROMPT, 0.3)
        # simulate typing the command
        typed = " ".join(cmd)
        for ch in typed:
            emit(ch, 0.02)
        emit("\r\n", 0.3)
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        out = (result.stdout or "") + (result.stderr or "")
        emit(_crlf(out) + "\r\n", 0.5)

    emit("[1;32m# Full report:[0m trustgate report\r\n", 1.0)

    header = {"version": 2, "width": WIDTH, "height": HEIGHT,
              "title": "TrustGate demo", "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"}}
    out_path = repo / "trustgate.cast"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(header) + "\n")
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    print(f"Wrote {out_path}  ({len(events)} events, duration ~{t:.0f}s)")
    print("Next: asciinema upload trustgate.cast   (or: agg trustgate.cast trustgate.gif)")
    sys.exit(0)


if __name__ == "__main__":
    main()
