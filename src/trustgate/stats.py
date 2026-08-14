"""Small, dependency-free statistics used across the Judge Lab.

Kept in stdlib so the core stays installable without numpy/scikit-learn. The heavier
stats extras (PPI, isotonic calibration) arrive with the sampling milestone.
"""

from __future__ import annotations

from collections.abc import Sequence


def _confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean of per-class recall over the classes actually present in ``y_true``.

    Robust to class imbalance and to a class being absent (that class is skipped).
    """
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    recalls = []
    if tp + fn > 0:
        recalls.append(tp / (tp + fn))   # recall of positive class
    if tn + fp > 0:
        recalls.append(tn / (tn + fp))   # recall of negative class
    return sum(recalls) / len(recalls) if recalls else 0.0


def cohen_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Cohen's kappa for two binary labelings. 1 = perfect, 0 = chance, <0 = worse."""
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    p_true1 = sum(y_true) / n
    p_pred1 = sum(y_pred) / n
    pe = p_true1 * p_pred1 + (1 - p_true1) * (1 - p_pred1)
    if 1 - pe == 0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


__all__ = ["balanced_accuracy", "cohen_kappa"]
