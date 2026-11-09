"""Turn prepared DROZY arrays into normalised model inputs."""

from __future__ import annotations

import numpy as np
import torch

from ..data.eeg import zscore_per_subject


def zscore_features_per_subject(x: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Per-subject z-score of a ``[n, features]`` array; NaNs become 0 (the subject mean)."""
    out = np.zeros_like(x, dtype=np.float32)
    for s in np.unique(subjects):
        m = subjects == s
        mu, sd = np.nanmean(x[m], axis=0), np.nanstd(x[m], axis=0) + 1e-6
        out[m] = np.nan_to_num((x[m] - mu) / sd)
    return out


def build_inputs(ds: dict[str, np.ndarray], use_clean_only: bool = True) -> dict[str, np.ndarray]:
    """Normalise the dataset from :func:`load_prepared` for training.

    * EEG: per-subject, per-channel z-score (proposal 3.6.1, step 4)
    * crops: scaled to 0-1
    * face features: per-subject z-score

    Per-subject statistics use only the input signals, never labels, so using
    the test subject's own statistics is allowed (as a deployed system would
    calibrate on a driver's first minutes).
    Windows flagged for extreme EEG amplitude are dropped when
    ``use_clean_only`` is set.
    """
    m = ds["keep"] if use_clean_only else np.ones(len(ds["label"]), bool)
    subj = ds["subject"][m]
    out = {
        "label": ds["label"][m].astype(np.int64),
        "subject": subj,
        "test": ds["test"][m],
        "window": ds["window_in_session"][m],
        "eeg": zscore_per_subject(ds["eeg"][m], subj),
    }
    if "crops" in ds:
        out["crops"] = ds["crops"][m]  # uint8; scaled to 0-1 per batch to save memory
        out["face_feats"] = zscore_features_per_subject(ds["face_feats"][m], subj)
    return out


INPUT_KEYS = ("eeg", "crops", "face_feats")


def to_batch(inputs: dict[str, np.ndarray], idx: np.ndarray, device="cpu") -> dict[str, torch.Tensor]:
    batch = {}
    for k in INPUT_KEYS:
        if k in inputs:
            v = torch.from_numpy(inputs[k][idx])
            batch[k] = (v.float() / 255.0 if k == "crops" else v.float()).to(device)
    return batch
