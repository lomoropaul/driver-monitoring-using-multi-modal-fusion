"""Significance tests from proposal section 3.9.

* McNemar's test on paired predictions for the same test windows
* Wilcoxon signed-rank test on per-fold metrics
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, wilcoxon


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test: do models A and B make different numbers of errors?

    ``b`` = windows only A got right, ``c`` = windows only B got right.
    Uses the exact binomial form, valid for any discordant count.
    """
    a_ok, b_ok = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_a_correct": b, "only_b_correct": c, "p_value": float(p)}


def wilcoxon_folds(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired per-fold values (NaN folds dropped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 2 or np.allclose(a, b):
        return {"n_folds": int(len(a)), "median_diff": float(np.median(a - b)) if len(a) else float("nan"),
                "p_value": 1.0}
    res = wilcoxon(a, b)
    return {"n_folds": int(len(a)), "median_diff": float(np.median(a - b)), "p_value": float(res.pvalue)}
