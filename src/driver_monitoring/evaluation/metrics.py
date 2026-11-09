"""Detection metrics from proposal Table 3.1."""

from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall, F1, false-negative rate and specificity.

    Positive class = vigilance decline (1). Ratios with an empty denominator
    are NaN (e.g. recall for a subject with no decline windows).
    """
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall) if tp else (0.0 if tp + fn else float("nan"))
    return {
        "accuracy": ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": ratio(fn, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(y_true),
    }


def summarise_folds(fold_metrics: list[dict[str, float]],
                    keys=("accuracy", "precision", "recall", "f1", "fnr", "specificity")) -> dict:
    """Mean and standard deviation across folds, ignoring NaN folds."""
    out = {}
    for k in keys:
        v = np.array([m[k] for m in fold_metrics], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"mean": float(v.mean()) if len(v) else float("nan"),
                  "std": float(v.std()) if len(v) else float("nan"), "folds": int(len(v))}
    return out
