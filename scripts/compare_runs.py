"""Compare all finished runs and test hypotheses H1 and H2 (proposal sections 1.3.1 and 3.9).

    .venv\\Scripts\\python.exe scripts\\compare_runs.py
    .venv\\Scripts\\python.exe scripts\\compare_runs.py --no-latency   # skip the timing runs

Prints a results table with model size (MB) and CPU latency per window, then:

H1: the best fusion model against the best single-modality baseline (both
chosen by pooled accuracy):
* McNemar's test on all test windows (accuracy) and on decline windows only
  (false negatives), using paired predictions
* Wilcoxon signed-rank test on per-fold accuracy and false-negative rate

H2: the distilled Student against the Teacher (size reduction, retention of
accuracy and recall) and against the same Student trained without
distillation (the ablation that isolates the effect of distillation).

Latency here is measured on this PC's CPU with one thread and batch size 1;
the Jetson figures come from the deployment step.

Writes runs/comparison.json and runs/comparison.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from driver_monitoring.evaluation.efficiency import example_window, latency_ms, size_mb
from driver_monitoring.evaluation.stats import mcnemar, wilcoxon_folds
from driver_monitoring.models.branches import build_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ORDER = ["eeg_baseline", "face_baseline", "early_fusion", "late_fusion", "hybrid_fusion",
         "teacher", "student_scratch", "student_kd", "student_mid_scratch", "student_mid_kd"]
STUDENTS = ["student", "student_mid"]
KEY = ["subject", "test", "window"]


def _model(summary: dict):
    cfg = summary["config"]
    if "model_kwargs" not in cfg:
        return None  # late fusion has no single network
    return build_model(cfg["model"], **cfg["model_kwargs"])


def paired(run_a: str, run_b: str) -> dict:
    """McNemar and Wilcoxon tests of run A against run B on the same test windows and folds."""
    a = pd.read_csv(RUNS / run_a / "predictions.csv")
    b = pd.read_csv(RUNS / run_b / "predictions.csv")
    m = a.merge(b, on=KEY + ["label"], suffixes=("_a", "_b"))
    y, pa, pb = m["label"].to_numpy(), (m["prob_decline_a"] >= .5).to_numpy(), (m["prob_decline_b"] >= .5).to_numpy()
    pos = y == 1
    fa = pd.read_csv(RUNS / run_a / "fold_metrics.csv").set_index("subject")
    fb = pd.read_csv(RUNS / run_b / "fold_metrics.csv").set_index("subject").loc[fa.index]
    return {
        "mcnemar_all_windows": mcnemar(y, pa, pb),
        "mcnemar_decline_windows": mcnemar(y[pos], pa[pos], pb[pos]),
        "wilcoxon_fold_accuracy": wilcoxon_folds(fa["accuracy"], fb["accuracy"]),
        "wilcoxon_fold_fnr": wilcoxon_folds(fa["fnr"], fb["fnr"]),
    }


def print_paired(name_a: str, name_b: str, r: dict) -> None:
    for k in ("mcnemar_all_windows", "mcnemar_decline_windows"):
        v = r[k]
        print(f"  McNemar ({k.split('_', 1)[1].replace('_', ' ')}): only {name_a} right {v['only_a_correct']}, "
              f"only {name_b} right {v['only_b_correct']}, p = {v['p_value']:.2g}")
    for k in ("wilcoxon_fold_accuracy", "wilcoxon_fold_fnr"):
        v = r[k]
        print(f"  Wilcoxon ({k.split('_', 2)[2]}): median fold difference {v['median_diff']:+.3f} "
              f"over {v['n_folds']} folds, p = {v['p_value']:.2g}")


def h2_report(df: pd.DataFrame, runs: list[str], name: str) -> dict:
    """Distilled Student ``name`` against the Teacher and against the same Student without distillation."""
    kd_run, scratch_run = f"{name}_kd", f"{name}_scratch"
    t, s = df.loc["teacher"], df.loc[kd_run]
    h2 = {
        "parameter_reduction": float(1 - s["params"] / t["params"]),
        "size_reduction": float(1 - s["size_mb"] / t["size_mb"]),
        "at_least_70pct_smaller": bool(1 - s["params"] / t["params"] >= 0.70
                                       and 1 - s["size_mb"] / t["size_mb"] >= 0.70),
        "accuracy_retention": float(s["accuracy"] / t["accuracy"]),
        "recall_retention": float(s["recall"] / t["recall"]),
        "student_vs_teacher": paired(kd_run, "teacher"),
    }
    if "cpu_mean_ms" in df:
        h2["cpu_speedup"] = float(t["cpu_mean_ms"] / s["cpu_mean_ms"])
    print(f"\nH2: distilled {name} vs Teacher")
    print(f"  parameters {int(s['params']):,} vs {int(t['params']):,}  ({h2['parameter_reduction']:.1%} fewer); "
          f"size {s['size_mb']:.3f} vs {t['size_mb']:.3f} MB  ({h2['size_reduction']:.1%} smaller)")
    print(f"  retention: accuracy {h2['accuracy_retention']:.1%}, recall {h2['recall_retention']:.1%}")
    if "cpu_speedup" in h2:
        print(f"  CPU latency {s['cpu_mean_ms']:.1f} vs {t['cpu_mean_ms']:.1f} ms per window "
              f"({h2['cpu_speedup']:.1f}x faster)")
    print_paired("Student", "Teacher", h2["student_vs_teacher"])
    if scratch_run in runs:
        h2["kd_vs_scratch"] = paired(kd_run, scratch_run)
        print(f"  Ablation, same Student with vs without distillation: accuracy "
              f"{s['accuracy']:.3f} vs {df.loc[scratch_run, 'accuracy']:.3f}, "
              f"FNR {s['fnr']:.3f} vs {df.loc[scratch_run, 'fnr']:.3f}")
        print_paired("KD", "scratch", h2["kd_vs_scratch"])
    kd = json.loads((RUNS / kd_run / "summary.json").read_text())
    h2["chosen_per_fold"] = kd.get("chosen_per_fold")
    h2["grid_test_pooled"] = {k: {m: v[m] for m in ("accuracy", "recall", "fnr")}
                              for k, v in kd.get("grid_test_pooled", {}).items()}
    return h2


def main() -> None:
    global RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--runs", type=Path, default=RUNS, help="folder holding the runs (e.g. on Google Drive)")
    args = ap.parse_args()
    RUNS = args.runs

    runs = [r for r in ORDER if (RUNS / r / "summary.json").exists()]
    table, example = [], example_window()
    for r in runs:
        s = json.loads((RUNS / r / "summary.json").read_text())
        p, f = s["pooled"], s["per_fold"]
        model = _model(s)
        row = {"run": r, "params": s["parameters"],
               **{k: p[k] for k in ("accuracy", "precision", "recall", "f1", "fnr", "specificity")},
               "fold_acc": f"{f['accuracy']['mean']:.3f} +/- {f['accuracy']['std']:.3f}",
               "size_mb": size_mb(model) if model is not None else np.nan}
        if not args.no_latency:
            row.update({f"cpu_{k}": v for k, v in latency_ms(model, example).items()} if model is not None
                       else {"cpu_mean_ms": np.nan, "cpu_p95_ms": np.nan})
        table.append(row)
    df = pd.DataFrame(table).set_index("run")
    print(df.round(3).to_string())

    result = {"table": df.reset_index().to_dict("records")}
    singles = [r for r in runs if r.endswith("baseline")]
    fusions = [r for r in runs if r.endswith("fusion")]
    if singles and fusions:
        best_s = max(singles, key=lambda r: df.loc[r, "accuracy"])
        best_f = max(fusions, key=lambda r: df.loc[r, "accuracy"])
        h1 = {"fusion": best_f, "single": best_s,
              "accuracy_above_90pct": bool(df.loc[best_f, "accuracy"] > 0.90), **paired(best_f, best_s)}
        result["h1"] = h1
        print(f"\nH1: best fusion ({best_f}) vs best single modality ({best_s})")
        print(f"  accuracy {df.loc[best_f, 'accuracy']:.3f} vs {df.loc[best_s, 'accuracy']:.3f}"
              f"  (above 90%: {h1['accuracy_above_90pct']})")
        print(f"  FNR      {df.loc[best_f, 'fnr']:.3f} vs {df.loc[best_s, 'fnr']:.3f}")
        print_paired("fusion", "single", h1)

    h2 = {name: h2_report(df, runs, name) for name in STUDENTS if {"teacher", f"{name}_kd"} <= set(runs)}
    if h2:
        result["h2"] = h2

    (RUNS / "comparison.json").write_text(json.dumps(result, indent=2, default=str))
    try:
        text = df.round(3).to_markdown()
    except ImportError:  # to_markdown needs the optional 'tabulate' package
        text = df.round(3).to_string()
    (RUNS / "comparison.md").write_text(text)
    print(f"\nSaved {RUNS / 'comparison.json'}")


if __name__ == "__main__":
    main()
