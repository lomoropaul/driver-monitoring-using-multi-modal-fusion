"""Teacher, Student and distillation runs with leave-one-subject-out CV (proposal 3.7.3-3.7.4, 3.8 step 4).

    .venv\\Scripts\\python.exe scripts\\train_distill.py --stage teacher   # scaled-up early fusion
    .venv\\Scripts\\python.exe scripts\\train_distill.py --stage student   # Student with KD, (T, alpha) grid
    .venv\\Scripts\\python.exe scripts\\train_distill.py --stage scratch   # same Student without KD (ablation)
    .venv\\Scripts\\python.exe scripts\\train_distill.py --stage student --student student_mid   # larger Student

Sizes come from the ``distillation`` section of configs/default.yaml.
Results go to runs/teacher, runs/<student>_kd and runs/<student>_scratch.
The Student stage needs the Teacher run for the same folds.

Each finished fold is saved on its own, so an interrupted run resumes where it
stopped, and several processes can split the folds (``--folds 1 2 3 4 5
--threads 4``) and share an output folder. Running a stage again over all
folds afterwards just loads them and writes the merged summary.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from driver_monitoring.data.prepare import load_prepared
from driver_monitoring.models.branches import build_model, count_parameters
from driver_monitoring.training.data import branch_kwargs, build_inputs
from driver_monitoring.training.distill import run_distillation
from driver_monitoring.training.loso import TrainConfig, run_loso

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("teacher", "student", "scratch")


def model_spec(size: dict, eeg_kw: dict, face_kw: dict, modality_dropout: float) -> tuple[str, dict]:
    return size["strategy"], {"eeg_kwargs": {**eeg_kw, **size["eeg"]}, "face_kwargs": {**face_kw, **size["face"]},
                              "hidden": size["hidden"], "modality_dropout": modality_dropout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--student", default="student", help="Student size from the config (student, student_mid)")
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cuda or cpu")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="folder for all runs (e.g. on Google Drive)")
    ap.add_argument("--data", type=Path, help="processed data folder (default: from the config)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modality-dropout", type=float, default=0.15)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(args.config.read_text())
    kd_cfg = cfg["distillation"]
    processed = args.data or ROOT / cfg["data"]["processed"]
    inputs = build_inputs(load_prepared(processed / "drozy", processed / "drozy_face"))
    eeg_kw, face_kw = branch_kwargs(inputs)
    t_name, t_kw = model_spec(kd_cfg["teacher"], eeg_kw, face_kw, args.modality_dropout)
    s_name, s_kw = model_spec(kd_cfg[args.student], eeg_kw, face_kw, args.modality_dropout)
    n_t, n_s = count_parameters(build_model(t_name, **t_kw)), count_parameters(build_model(s_name, **s_kw))
    print(f"Teacher {n_t:,} parameters, Student {n_s:,} ({1 - n_s / n_t:.1%} smaller)")

    out_name = {"teacher": "teacher", "student": f"{args.student}_kd", "scratch": f"{args.student}_scratch"}
    out = args.out or args.runs / out_name[args.stage]
    name, kw = (t_name, t_kw) if args.stage == "teacher" else (s_name, s_kw)
    tc = TrainConfig(model=name, epochs=args.epochs, patience=args.patience, lr=args.lr,
                     seed=cfg["seed"], model_kwargs=kw, device=args.device)
    if args.stage == "student":
        summary = run_distillation(tc, inputs, args.runs / "teacher", t_name, t_kw, out,
                                   kd_cfg["temperatures"], kd_cfg["alphas"], args.folds)
    else:
        summary = run_loso(tc, inputs, out, args.folds)

    print("\nPooled over all test windows:")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in summary["pooled"].items()}, indent=2))
    print(f"\nSaved to {out}  ({summary['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
