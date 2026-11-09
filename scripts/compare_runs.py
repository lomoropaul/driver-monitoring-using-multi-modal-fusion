"""Compare all finished runs and test hypothesis H1 (proposal sections 1.3.1 and 3.9).

    .venv\\Scripts\\python.exe scripts\\compare_runs.py

Prints a results table, then compares the best fusion model with the best
single-modality baseline (both chosen by pooled accuracy):

* McNemar's test on all test windows (accuracy) and on decline windows only
  (false negatives), using paired predictions
* Wilcoxon signed-rank test on per-fold accuracy and false-negative rate

Writes runs/comparison.json and runs/comparison.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from driver_monitoring.evaluation.stats import mcnemar, wilcoxon_folds

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ORDER = ["eeg_baseline", "face_baseline", "early_fusion", "late_fusion", "hybrid_fusion"]
KEY = ["subject", "test", "window"]


def main() -> None:
    runs = [r for r in ORDER if (RUNS / r / "summary.json").exists()]
    table = []
    for r in runs:
        s = json.loads((RUNS / r / "summary.json").read_text())
        p, f = s["pooled"], s["per_fold"]
        table.append({"run": r, "params": s["parameters"], **{k: p[k] for k in
                      ("accuracy", "precision", "recall", "f1", "fnr", "specificity")},
                      "fold_acc": f"{f['accuracy']['mean']:.3f} ± {f['accuracy']['std']:.3f}"})
    df = pd.DataFrame(table).set_index("run")
    print(df.round(3).to_string())

    singles = [r for r in runs if r.endswith("baseline")]
    fusions = [r for r in runs if r.endswith("fusion")]
    result = {"table": df.reset_index().to_dict("records")}
    if singles and fusions:
        best_s = max(singles, key=lambda r: df.loc[r, "accuracy"])
        best_f = max(fusions, key=lambda r: df.loc[r, "accuracy"])
        a = pd.read_csv(RUNS / best_f / "predictions.csv")
        b = pd.read_csv(RUNS / best_s / "predictions.csv")
        m = a.merge(b, on=KEY + ["label"], suffixes=("_f", "_s"))
        y, pf, ps = m["label"].to_numpy(), (m["prob_decline_f"] >= .5).to_numpy(), (m["prob_decline_s"] >= .5).to_numpy()
        pos = y == 1
        fa = pd.read_csv(RUNS / best_f / "fold_metrics.csv").set_index("subject")
        fb = pd.read_csv(RUNS / best_s / "fold_metrics.csv").set_index("subject").loc[fa.index]
        h1 = {
            "fusion": best_f, "single": best_s,
            "accuracy_above_90pct": bool(df.loc[best_f, "accuracy"] > 0.90),
            "mcnemar_all_windows": mcnemar(y, pf, ps),
            "mcnemar_decline_windows": mcnemar(y[pos], pf[pos], ps[pos]),
            "wilcoxon_fold_accuracy": wilcoxon_folds(fa["accuracy"], fb["accuracy"]),
            "wilcoxon_fold_fnr": wilcoxon_folds(fa["fnr"], fb["fnr"]),
        }
        result["h1"] = h1
        print(f"\nH1: best fusion ({best_f}) vs best single modality ({best_s})")
        print(f"  accuracy {df.loc[best_f, 'accuracy']:.3f} vs {df.loc[best_s, 'accuracy']:.3f}"
              f"  (above 90%: {h1['accuracy_above_90pct']})")
        print(f"  FNR      {df.loc[best_f, 'fnr']:.3f} vs {df.loc[best_s, 'fnr']:.3f}")
        for k in ("mcnemar_all_windows", "mcnemar_decline_windows"):
            v = h1[k]
            print(f"  McNemar ({k.split('_', 1)[1].replace('_', ' ')}): only fusion right {v['only_a_correct']}, "
                  f"only single right {v['only_b_correct']}, p = {v['p_value']:.2g}")
        for k in ("wilcoxon_fold_accuracy", "wilcoxon_fold_fnr"):
            v = h1[k]
            print(f"  Wilcoxon ({k.split('_', 2)[2]}): median fold difference {v['median_diff']:+.3f} "
                  f"over {v['n_folds']} folds, p = {v['p_value']:.2g}")

    (RUNS / "comparison.json").write_text(json.dumps(result, indent=2, default=str))
    try:
        text = df.round(3).to_markdown()
    except ImportError:  # to_markdown needs the optional 'tabulate' package
        text = df.round(3).to_string()
    (RUNS / "comparison.md").write_text(text)
    print(f"\nSaved {RUNS / 'comparison.json'}")


if __name__ == "__main__":
    main()
