"""Late (decision-level) fusion: ``p = α p_eeg + (1 - α) p_face``.

Reuses the EEG and face baselines trained on each fold. α is chosen on that
fold's validation subjects (the same ones the baselines used for early
stopping) by minimising log-loss, then applied to the test subject.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..evaluation.metrics import binary_metrics
from ..models.branches import build_model, count_parameters
from .loso import _predict, choose_val_subjects, write_results

ALPHAS = np.round(np.linspace(0, 1, 21), 2)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def best_alpha(y: np.ndarray, p_eeg: np.ndarray, p_face: np.ndarray) -> float:
    losses = [log_loss(y, a * p_eeg + (1 - a) * p_face) for a in ALPHAS]
    return float(ALPHAS[int(np.argmin(losses))])


def _load(name: str, kwargs: dict, run_dir: Path, subject: int):
    model = build_model(name, **kwargs)
    model.load_state_dict(torch.load(run_dir / f"fold_subject{subject}.pt", weights_only=True, map_location="cpu"))
    return model.eval()


def run_late_fusion(inputs: dict, eeg_run: str | Path, face_run: str | Path, out_dir: str | Path,
                    eeg_kwargs: dict, face_kwargs: dict, n_val_subjects: int = 2,
                    folds: list[int] | None = None, log=print) -> dict:
    out_dir, eeg_run, face_run = Path(out_dir), Path(eeg_run), Path(face_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects, labels = inputs["subject"], inputs["label"]
    folds = folds or sorted(set(subjects.tolist()))
    n_params = (count_parameters(build_model("eeg", **eeg_kwargs))
                + count_parameters(build_model("face", **face_kwargs)) + 1)

    rows, fold_metrics, alphas, t0 = [], [], {}, time.time()
    for test in folds:
        val_s = choose_val_subjects(test, subjects, labels, n_val_subjects)
        val_idx = np.flatnonzero(np.isin(subjects, val_s))
        test_idx = np.flatnonzero(subjects == test)
        eeg, face = _load("eeg", eeg_kwargs, eeg_run, test), _load("face", face_kwargs, face_run, test)
        a = best_alpha(labels[val_idx], _predict(eeg, inputs, val_idx), _predict(face, inputs, val_idx))
        prob = a * _predict(eeg, inputs, test_idx) + (1 - a) * _predict(face, inputs, test_idx)
        m = binary_metrics(labels[test_idx], prob >= 0.5)
        m.update({"subject": test, "val_subjects": val_s, "alpha": a})
        fold_metrics.append(m)
        alphas[test] = a
        rows += [(int(subjects[i]), int(inputs["test"][i]), int(inputs["window"][i]), int(labels[i]),
                  float(p)) for i, p in zip(test_idx, prob)]
        log(f"[subject {test}] alpha = {a:.2f} (EEG weight)  acc {m['accuracy']:.3f}  "
            f"recall {m['recall']:.3f}  specificity {m['specificity']:.3f}")

    config = {"model": "late", "eeg_run": str(eeg_run), "face_run": str(face_run),
              "alpha_grid": ALPHAS.tolist(), "n_val_subjects": n_val_subjects}
    return write_results(out_dir, "late", config, n_params, rows, fold_metrics,
                         (time.time() - t0) / 60, extra={"alpha_per_fold": alphas})
