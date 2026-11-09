"""Check that every library the project needs is installed and working.

Run:  .venv\\Scripts\\python.exe scripts\\check_env.py
"""

import importlib
import sys
import tempfile
from pathlib import Path

LIBS = [
    "torch", "torchvision", "mne", "antropy", "cv2", "mediapipe", "numpy",
    "pandas", "scipy", "sklearn", "statsmodels", "onnx", "onnxruntime", "yaml",
]


def check_imports() -> bool:
    ok = True
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]   {name:<12} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name:<12} {exc}")
            ok = False
    return ok


def check_torch_onnx() -> None:
    """Tiny CNN forward pass, ONNX export and ONNX Runtime inference."""
    import numpy as np
    import onnxruntime as ort
    import torch

    model = torch.nn.Sequential(
        torch.nn.Conv1d(4, 8, 7), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(8, 2),
    ).eval()
    x = torch.randn(1, 4, 512)  # 4 EEG channels x 4 s at 128 Hz
    with torch.no_grad():
        y = model(x).numpy()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.onnx"
        torch.onnx.export(model, x, str(path), input_names=["x"], output_names=["y"],
                          dynamo=False)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        y_onnx = sess.run(None, {"x": x.numpy()})[0]
    assert np.allclose(y, y_onnx, atol=1e-5), "ONNX output differs from PyTorch"
    print(f"  [ok]   PyTorch -> ONNX -> ONNX Runtime round trip "
          f"(CUDA available: {torch.cuda.is_available()})")


def check_mne() -> None:
    import mne
    import numpy as np

    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(4, 256 * 8) * 1e-5, info, verbose=False)
    raw.filter(0.5, 45, verbose=False).notch_filter(50, verbose=False)
    print("  [ok]   MNE band-pass + 50 Hz notch filter")


def main() -> int:
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    print("Imports:")
    if not check_imports():
        return 1
    print("\nFunctional checks:")
    check_torch_onnx()
    check_mne()
    print("\nEnvironment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
