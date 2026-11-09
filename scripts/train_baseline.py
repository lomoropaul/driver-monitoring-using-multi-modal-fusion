"""Train a single-modality baseline with leave-one-subject-out cross-validation.

    .venv\\Scripts\\python.exe scripts\\train_baseline.py --model eeg
    .venv\\Scripts\\python.exe scripts\\train_baseline.py --model face
    .venv\\Scripts\\python.exe scripts\\train_baseline.py --model eeg --folds 1 2 --epochs 3   # quick check

Results go to runs/<model>_baseline/: predictions.csv, fold_metrics.csv,
summary.json and one model file per fold.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from driver_monitoring.data.prepare import load_prepared
from driver_monitoring.training.data import build_inputs
from driver_monitoring.training.loso import TrainConfig, run_loso

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["eeg", "face"], required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--folds", type=int, nargs="*", help="test subjects to run (default: all)")
    ap.add_argument("--out", type=Path, help="output folder (default: runs/<model>_baseline)")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    cfg = yaml.safe_load(args.config.read_text())
    processed = ROOT / cfg["data"]["processed"]
    face = processed / "drozy_face" if args.model == "face" else None
    ds = load_prepared(processed / "drozy", face)
    inputs = build_inputs(ds)

    if args.model == "eeg":
        kwargs = {"n_channels": inputs["eeg"].shape[1], "n_samples": inputs["eeg"].shape[2]}
    else:
        kwargs = {"n_feats": inputs["face_feats"].shape[1]}
    tc = TrainConfig(model=args.model, epochs=args.epochs, patience=args.patience,
                     seed=cfg["seed"], model_kwargs=kwargs)
    out = args.out or ROOT / "runs" / f"{args.model}_baseline"
    summary = run_loso(tc, inputs, out, args.folds)

    print("\nPooled over all test windows:")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in summary["pooled"].items()}, indent=2))
    print("\nMean ± std across folds:")
    for k, v in summary["per_fold"].items():
        print(f"  {k:<12} {v['mean']:.3f} ± {v['std']:.3f}  ({v['folds']} folds)")
    print(f"\nSaved to {out}  ({summary['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
