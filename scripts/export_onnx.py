"""Export a trained model to ONNX, the format used on the Jetson Orin Nano (ONNX Runtime or TensorRT).

    .venv\\Scripts\\python.exe scripts\\export_onnx.py                       # distilled Student, fold of driver 7
    .venv\\Scripts\\python.exe scripts\\export_onnx.py --model teacher --subject 7

The model of the fold in which --subject was held out is exported, so a demo
on that driver uses a model that never saw them. The export is checked against
PyTorch on random inputs and saved to exports/<model>_S<subject>.onnx.

On the Jetson the same file can be run with ONNX Runtime, or compiled with
TensorRT:  trtexec --onnx=<file>.onnx --fp16 --saveEngine=<file>.engine
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from driver_monitoring.models.branches import build_model

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("eeg", "crops", "face_feats")


class OnnxWrapper(nn.Module):
    """Takes the three input tensors directly (ONNX has no dict inputs) and returns P(decline)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, eeg, crops, face_feats):
        logits = self.model({"eeg": eeg, "crops": crops, "face_feats": face_feats})
        return torch.softmax(logits, dim=1)[:, 1:]


def example_inputs(model_kwargs: dict, batch: int = 1):
    eeg_kw, face_kw = model_kwargs.get("eeg_kwargs", {}), model_kwargs.get("face_kwargs", {})
    return (torch.randn(batch, eeg_kw.get("n_channels", 5), eeg_kw.get("n_samples", 512)),
            torch.rand(batch, 10, 64, 64),
            torch.randn(batch, face_kw.get("n_feats", 10)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="student_kd", help="run folder in runs/ (a fusion model)")
    ap.add_argument("--subject", type=int, default=7, help="use the model trained without this driver")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    run = ROOT / "runs" / args.model
    cfg = json.loads((run / "summary.json").read_text())["config"]
    if cfg["model"] not in ("early", "hybrid"):
        raise SystemExit("Export a fusion model (early or hybrid); single-modality models take one input.")
    model = build_model(cfg["model"], **cfg["model_kwargs"])
    model.load_state_dict(torch.load(run / f"fold_subject{args.subject}.pt", weights_only=True, map_location="cpu"))
    wrapped = OnnxWrapper(model).eval()

    out = args.out or ROOT / "exports" / f"{args.model}_S{args.subject}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    x = example_inputs(cfg["model_kwargs"])
    kw = dict(input_names=list(INPUTS), output_names=["p_decline"], opset_version=17,
              dynamic_axes={n: {0: "batch"} for n in [*INPUTS, "p_decline"]})
    try:
        torch.onnx.export(wrapped, x, str(out), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(wrapped, x, str(out), **kw)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(out)))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    test = example_inputs(cfg["model_kwargs"], batch=4)
    got = sess.run(None, {n: t.numpy() for n, t in zip(INPUTS, test)})[0]
    with torch.no_grad():
        ref = wrapped(*test).numpy()
    diff = float(np.abs(got - ref).max())
    print(f"Exported {args.model} (fold of driver {args.subject}) to {out}")
    print(f"  size {out.stat().st_size / 1000:.0f} KB, max difference vs PyTorch {diff:.1e}")
    if diff > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
