"""Print the 14 DROZY drivers: their sessions and labels, and each model's accuracy on each driver.

    .venv\\Scripts\\python.exe scripts\\driver_results.py
    .venv\\Scripts\\python.exe scripts\\driver_results.py --metric recall

Reads data/raw/drozy/KSS.txt and runs/<model>/fold_metrics.csv (each driver is
tested by the model trained without them).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"eeg_baseline": "EEG only", "face_baseline": "Face only", "early_fusion": "Early",
          "late_fusion": "Late", "hybrid_fusion": "Hybrid", "teacher": "Teacher", "student_kd": "Student"}
LOST = {(9, 1), (10, 2), (12, 2), (12, 3), (13, 3), (7, 1)}  # missing from DROZY (7-1 never happened)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "recall", "specificity", "f1", "fnr"])
    ap.add_argument("--decline-min", type=int, default=7, help="KSS at or above this is labelled drowsy")
    args = ap.parse_args()

    kss = np.loadtxt(ROOT / "data" / "raw" / "drozy" / "KSS.txt", dtype=int)
    folds = {m: pd.read_csv(ROOT / "runs" / m / "fold_metrics.csv").set_index("subject")
             for m in MODELS if (ROOT / "runs" / m / "fold_metrics.csv").exists()}
    any_run = next(iter(folds.values()))

    print("DRIVERS: sleepiness rating (KSS 1-9) per test; '*' = drowsy (KSS >= %d); '-' = lost\n" % args.decline_min)
    print(f"{'Driver':<8}{'Test 1':>8}{'Test 2':>8}{'Test 3':>8}{'Windows':>10}{'Drowsy':>9}")
    for s in range(1, 15):
        cells = []
        for t in range(1, 4):
            if (s, t) in LOST:
                cells.append("-")
            else:
                k = kss[s - 1, t - 1]
                cells.append(f"{k}{'*' if k >= args.decline_min else ' '}")
        n = int(any_run.loc[s, "n"])
        drowsy = int(any_run.loc[s, "tp"] + any_run.loc[s, "fn"])
        print(f"{s:<8}{cells[0]:>8}{cells[1]:>8}{cells[2]:>8}{n:>10}{drowsy:>9}")

    print(f"\n{args.metric.upper()} ON EACH DRIVER (model trained without that driver)\n")
    table = pd.DataFrame({MODELS[m]: f[args.metric] for m, f in folds.items()})
    table.index.name = "Driver"
    table.loc["Mean"] = table.mean()
    print(table.round(3).to_string(na_rep="  n/a"))
    if args.metric in ("recall", "fnr"):
        print("\nn/a: drivers 3, 12 and 13 have no drowsy windows, so recall and FNR are undefined.")
    best = table.drop(index="Mean").mean(axis=1)
    print(f"\nEasiest driver: {best.idxmax()} (mean {best.max():.3f})   Hardest driver: {best.idxmin()} (mean {best.min():.3f})")


if __name__ == "__main__":
    main()
