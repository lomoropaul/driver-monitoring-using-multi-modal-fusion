"""Test a trained model on one DROZY driver and see what it predicts, window by window.

    .venv\\Scripts\\python.exe scripts\\test_model.py --subject 7
    .venv\\Scripts\\python.exe scripts\\test_model.py --subject 7 --model student_kd
    .venv\\Scripts\\python.exe scripts\\test_model.py --subject 4 --test 3 --model early_fusion

The model used is the one from the fold in which this driver was held out, so
the driver was never seen in training or validation: this is a fair test.

For each session it prints the sleepiness rating (KSS), the true label, how many
4-second windows the model flagged as vigilance decline, and the accuracy, then
saves a plot of the predicted decline probability over time to
runs/tests/S<subject>_<model>.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from driver_monitoring.data.prepare import load_prepared
from driver_monitoring.models.branches import build_model
from driver_monitoring.training.data import build_inputs
from driver_monitoring.training.loso import _predict

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["teacher", "early_fusion", "hybrid_fusion", "eeg_baseline", "face_baseline", "student_kd",
          "student_scratch", "student_mid_kd", "student_mid_scratch"]


def session_kss(folder: Path) -> dict:
    """{(subject, test): KSS} from the prepared session files."""
    out = {}
    for f in folder.glob("*.npz"):
        d = np.load(f)
        out[(int(d["subject"]), int(d["test"]))] = int(d["kss"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, required=True, help="DROZY driver, 1-14")
    ap.add_argument("--test", type=int, choices=[1, 2, 3], help="one session only (default: all of this driver's)")
    ap.add_argument("--model", default="teacher", choices=MODELS)
    ap.add_argument("--threshold", type=float, default=0.5, help="decline probability above which a window is flagged")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    run = ROOT / "runs" / args.model
    ckpt = run / f"fold_subject{args.subject}.pt"
    if not ckpt.exists():
        raise SystemExit(f"No model for subject {args.subject} in {run}. Train it first (see README).")
    cfg = json.loads((run / "summary.json").read_text())["config"]
    model = build_model(cfg["model"], **cfg["model_kwargs"])
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))

    processed = ROOT / yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["data"]["processed"]
    print("Loading data...")
    inputs = build_inputs(load_prepared(processed / "drozy", processed / "drozy_face"))
    kss = session_kss(processed / "drozy")
    sel = inputs["subject"] == args.subject
    if args.test:
        sel &= inputs["test"] == args.test
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        raise SystemExit(f"No clean windows for subject {args.subject}" + (f", test {args.test}" if args.test else ""))
    prob = _predict(model, inputs, idx)
    flagged = prob >= args.threshold

    print(f"\nModel: {args.model} (trained without driver {args.subject})")
    print(f"{'Session':<9}{'KSS':>4}  {'True state':<12}{'Windows':>8}  {'Flagged decline':>16}  {'Accuracy':>9}")
    sessions = sorted(set(inputs["test"][idx].tolist()))
    for t in sessions:
        m = inputs["test"][idx] == t
        y = inputs["label"][idx][m]
        state = "DECLINE" if y[0] == 1 else "alert"
        acc = (flagged[m] == y).mean()
        print(f"{args.subject}-{t:<7}{kss.get((args.subject, t), '?'):>4}  {state:<12}{m.sum():>8}  "
              f"{flagged[m].sum():>6} ({flagged[m].mean():>5.1%})  {acc:>9.1%}")
    y = inputs["label"][idx]
    print(f"{'All':<9}{'':>4}  {'':<12}{len(idx):>8}  {flagged.sum():>6} ({flagged.mean():>5.1%})  "
          f"{(flagged == y).mean():>9.1%}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(sessions), 1, figsize=(10, 2.1 * len(sessions)), squeeze=False, sharex=True)
    for ax, t in zip(axes[:, 0], sessions):
        m = inputs["test"][idx] == t
        minutes = inputs["window"][idx][m] * 4 / 60
        decline = inputs["label"][idx][m][0] == 1
        ax.fill_between(minutes, 0, prob[m], color="#c0185a" if decline else "#004a93", alpha=0.35, step="mid")
        ax.plot(minutes, prob[m], color="#c0185a" if decline else "#004a93", lw=1)
        ax.axhline(args.threshold, color="black", ls="--", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(decline)")
        ax.set_title(f"Session {args.subject}-{t}: KSS {kss.get((args.subject, t), '?')}, truly "
                     f"{'DROWSY' if decline else 'alert'}; flagged {flagged[m].mean():.0%} of windows", fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("Minutes into session")
    fig.tight_layout()
    out = ROOT / "runs" / "tests" / f"S{args.subject}{'-' + str(args.test) if args.test else ''}_{args.model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
