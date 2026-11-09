"""Train an EEG + face fusion model with leave-one-subject-out cross-validation.

Run the two baselines first (scripts/train_baseline.py); fusion starts each
branch from the baseline trained on the same fold, and late fusion reuses
them directly.

    .venv\\Scripts\\python.exe scripts\\train_fusion.py --strategy late      # minutes
    .venv\\Scripts\\python.exe scripts\\train_fusion.py --strategy early
    .venv\\Scripts\\python.exe scripts\\train_fusion.py --strategy hybrid

Results go to runs/<strategy>_fusion/.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from driver_monitoring.data.prepare import load_prepared
from driver_monitoring.training.data import branch_kwargs, build_inputs
from driver_monitoring.training.late_fusion import run_late_fusion
from driver_monitoring.training.loso import TrainConfig, run_loso

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["early", "late", "hybrid"], required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4, help="lower than baselines: branches start pre-trained")
    ap.add_argument("--modality-dropout", type=float, default=0.15)
    ap.add_argument("--no-init", action="store_true", help="train branches from scratch")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    cfg = yaml.safe_load(args.config.read_text())
    processed = ROOT / cfg["data"]["processed"]
    inputs = build_inputs(load_prepared(processed / "drozy", processed / "drozy_face"))
    eeg_kw, face_kw = branch_kwargs(inputs)
    runs = ROOT / "runs"
    out = args.out or runs / f"{args.strategy}_fusion"

    if args.strategy == "late":
        summary = run_late_fusion(inputs, runs / "eeg_baseline", runs / "face_baseline", out,
                                  eeg_kw, face_kw, folds=args.folds)
    else:
        tc = TrainConfig(model=args.strategy, epochs=args.epochs, patience=args.patience, lr=args.lr,
                         seed=cfg["seed"],
                         model_kwargs={"eeg_kwargs": eeg_kw, "face_kwargs": face_kw,
                                       "modality_dropout": args.modality_dropout},
                         init_from={} if args.no_init else {"eeg": str(runs / "eeg_baseline"),
                                                            "face": str(runs / "face_baseline")})
        summary = run_loso(tc, inputs, out, args.folds)

    print("\nPooled over all test windows:")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in summary["pooled"].items()}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
