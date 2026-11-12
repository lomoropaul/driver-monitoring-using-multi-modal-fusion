"""The in-vehicle program: run the exported Student on a stream of merged EEG + face windows.

    .venv\\Scripts\\python.exe scripts\\edge_demo.py --subject 7 --test 3              # replay as fast as possible
    .venv\\Scripts\\python.exe scripts\\edge_demo.py --subject 7 --test 3 --realtime   # one window every 4 s, like a car

This is the loop a Jetson Orin Nano would run. Here the "sensors" are a recorded
DROZY session replayed window by window (each window = 4 s of 5-channel EEG +
10 face frames); in a vehicle the same windows would come from the EEG cap and
the infrared camera. For every window it:

1. runs the ONNX model and times it,
2. averages the decline probability over the last minute (15 windows),
3. sets the alarm level: OK below 0.5, WARNING from 0.5, STOP from 0.8.

ONNX Runtime uses the GPU (TensorRT or CUDA) when available, as on a Jetson,
and the CPU otherwise. The model is exported with scripts/export_onnx.py, from
the fold in which this driver was held out, so the driver is unseen.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from driver_monitoring.data.prepare import load_prepared
from driver_monitoring.training.data import build_inputs

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [(0.8, "STOP", "Stop now: pull over safely and rest"), (0.5, "WARNING", "Signs of drowsiness: take a break soon"),
          (0.0, "OK", "Driver attentive")]


def level(p: float) -> tuple[str, str]:
    for threshold, name, text in LEVELS:
        if p >= threshold:
            return name, text
    return LEVELS[-1][1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, default=7)
    ap.add_argument("--test", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--model", default="student_kd", help="exports/<model>_S<subject>.onnx")
    ap.add_argument("--realtime", action="store_true", help="wait 4 s between windows, as in a vehicle")
    ap.add_argument("--smooth", type=int, default=15, help="windows averaged for the alarm (15 = 1 minute)")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (1 approximates a small edge CPU)")
    args = ap.parse_args()

    import onnxruntime as ort
    path = ROOT / "exports" / f"{args.model}_S{args.subject}.onnx"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/export_onnx.py --model {args.model} --subject {args.subject}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    wanted = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in wanted if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(path), opts, providers=providers)

    processed = ROOT / yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["data"]["processed"]
    inputs = build_inputs(load_prepared(processed / "drozy", processed / "drozy_face"))
    idx = np.flatnonzero((inputs["subject"] == args.subject) & (inputs["test"] == args.test))
    if len(idx) == 0:
        raise SystemExit(f"No windows for session {args.subject}-{args.test}")
    truth = "DECLINE (drowsy)" if inputs["label"][idx[0]] == 1 else "alert"

    print(f"Model {path.name} ({path.stat().st_size / 1000:.0f} KB) on {sess.get_providers()[0]}")
    print(f"Session {args.subject}-{args.test}: {len(idx)} windows of 4 s, true state {truth}\n")
    print(f"{'time':>6}  {'p(window)':>9}  {'p(1 min)':>8}  {'level':<8} {'ms':>6}")
    recent, lat, first = deque(maxlen=args.smooth), [], {}
    feed = lambda i: {"eeg": inputs["eeg"][i:i + 1].astype(np.float32),
                      "crops": (inputs["crops"][i:i + 1] / 255.0).astype(np.float32),
                      "face_feats": inputs["face_feats"][i:i + 1].astype(np.float32)}
    sess.run(None, feed(idx[0]))  # warm-up
    prev = None
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        p = float(sess.run(None, feed(i))[0][0, 0])
        lat.append((time.perf_counter() - t0) * 1000)
        recent.append(p)
        ps = float(np.mean(recent))
        name, text = level(ps)
        secs = int(inputs["window"][i]) * 4
        first.setdefault(name, secs)
        if name != prev or k % 15 == 0:
            note = f"  <- {text}" if name != prev else ""
            print(f"{secs // 60:02d}:{secs % 60:02d}  {p:>9.2f}  {ps:>8.2f}  {name:<8} {lat[-1]:>6.2f}{note}")
            prev = name
        if args.realtime:
            time.sleep(max(0.0, 4 - (time.perf_counter() - t0)))

    lat = np.array(lat)
    print(f"\nLatency per window: mean {lat.mean():.2f} ms, 95th percentile {np.percentile(lat, 95):.2f} ms "
          f"(limit 100 ms, window length 4000 ms)")
    for name in ("WARNING", "STOP"):
        if name in first:
            print(f"First {name}: {first[name] // 60:02d}:{first[name] % 60:02d} into the session")
    if not ({"WARNING", "STOP"} & first.keys()):
        print("No alarm raised during this session.")


if __name__ == "__main__":
    main()
