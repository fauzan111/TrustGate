"""Decision engine — the release gate.

Walking-skeleton policy (upgraded to conformal risk control in milestone 9-10):

* **ship**        — candidate is not meaningfully worse: its lower CI bound is at or above
                    the baseline mean minus a tolerance ``epsilon``.
* **block**       — candidate is clearly worse: its upper CI bound is below the baseline
                    mean minus ``epsilon`` (a real, detected regression).
* **investigate** — the interval straddles the threshold; more labels are needed.

Using CI bounds (not just point estimates) is what makes the gate robust to noise.
"""

from __future__ import annotations

from trustgate.models import Decision, Estimate, Verdict


def decide(
    candidate: Estimate,
    baseline: Estimate | None = None,
    epsilon: float = 0.02,
) -> Decision:
    """Compare a candidate estimate against a baseline and emit a release verdict."""
    if baseline is None:
        # No baseline: ship only if we are confident quality is decent.
        verdict = Verdict.SHIP if candidate.ci_low >= 0.5 else Verdict.INVESTIGATE
        return Decision(
            verdict=verdict,
            metric=candidate.metric,
            candidate=candidate,
            reason=f"No baseline; candidate CI=[{candidate.ci_low:.3f}, {candidate.ci_high:.3f}].",
        )

    threshold = baseline.mean - epsilon
    delta = candidate.mean - baseline.mean

    if candidate.ci_low >= threshold:
        verdict, reason = Verdict.SHIP, "Candidate lower bound >= baseline - eps: no regression."
    elif candidate.ci_high < threshold:
        verdict, reason = Verdict.BLOCK, "Candidate upper bound < baseline - eps: regression detected."
    else:
        verdict, reason = Verdict.INVESTIGATE, "Interval straddles threshold: collect more labels."

    return Decision(
        verdict=verdict,
        metric=candidate.metric,
        candidate=candidate,
        baseline=baseline,
        delta=delta,
        reason=f"{reason} (delta={delta:+.3f}, eps={epsilon}).",
    )


from trustgate.decision.conformal import (  # noqa: E402
    DiffEstimate,
    decide_from_diff,
    difference_estimate,
    release_decision,
)

__all__ = [
    "decide",
    "DiffEstimate", "difference_estimate", "decide_from_diff", "release_decision",
]
