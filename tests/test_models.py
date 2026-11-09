import json

import numpy as np
import pytest
import torch

from driver_monitoring.data.face import (FEATURE_NAMES, count_blinks, eye_aspect_ratio,
                                         eye_mouth_crop, head_pose, mouth_aspect_ratio,
                                         window_features)
from driver_monitoring.evaluation.metrics import binary_metrics, summarise_folds
from driver_monitoring.models.branches import EEGNet, FaceNet, count_parameters
from driver_monitoring.training.data import build_inputs, zscore_features_per_subject
from driver_monitoring.training.loso import TrainConfig, choose_val_subjects, run_loso


def _face(eye_open: float = 1.0, mouth_open: float = 0.2) -> np.ndarray:
    """A synthetic 68-point face (2D) with controllable eye and mouth opening."""
    p = np.zeros((68, 2), np.float32)
    for start, cx in ((36, 100.0), (42, 160.0)):
        xs = cx + np.array([-15, -7, 7, 15, 7, -7])
        ys = 100 + np.array([0, -1, -1, 0, 1, 1]) * 5 * eye_open
        p[start:start + 6] = np.stack([xs, ys], 1)
    p[48:60] = np.stack([130 + 25 * np.cos(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
                         170 + 10 * np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))], 1)
    xs = 130 + np.array([-20, -10, 0, 10, 20, 10, 0, -10])
    ys = 170 + np.array([0, -1, -1, -1, 0, 1, 1, 1]) * 20 * mouth_open
    p[60:68] = np.stack([xs, ys], 1)
    p[8] = [130, 220]
    p[30] = [130, 140]
    return p


def test_ear_and_mar_track_opening():
    open_, closed = _face(1.0), _face(0.1)
    ear = eye_aspect_ratio(np.stack([open_, closed]))
    assert ear[0] > 0.3 > 0.05 > ear[1]
    mar = mouth_aspect_ratio(np.stack([_face(mouth_open=0.1), _face(mouth_open=1.5)]))
    assert mar[1] > 5 * mar[0]


def test_head_pose_is_zero_for_frontal_face():
    face2d = _face()
    s3 = np.concatenate([face2d, np.full((68, 1), 600.0, np.float32)], axis=1)[None]
    assert np.allclose(head_pose(s3), 0, atol=1e-3)
    tilted = s3.copy()
    tilted[0, 42:48, 1] += 20  # subject's left eye lower -> roll
    assert abs(head_pose(tilted)[0, 2]) > 5


def test_blinks_and_window_features():
    assert count_blinks(np.array([0, 1, 1, 0, 0, 1, 0], bool)) == 2
    fps, window_s = 10, 4
    times = np.arange(80) / fps                       # 8 s of frames
    ear = np.full(80, 0.3, np.float32)
    ear[45:50] = 0.05                                  # one 0.5 s closure in window 1
    feats = window_features(ear, np.full(80, 0.1), np.zeros((80, 3)), times, 3, window_s, 0.0, 0.15)
    assert feats.shape == (3, len(FEATURE_NAMES))
    perclos = FEATURE_NAMES.index("perclos")
    assert feats[0, perclos] == 0 and feats[1, perclos] == pytest.approx(5 / 40)
    assert feats[1, FEATURE_NAMES.index("blinks_per_min")] == pytest.approx(15.0)
    assert np.isnan(feats[2]).all()                    # no frames in 8-12 s


def test_eye_mouth_crop_shape():
    img = np.random.default_rng(0).integers(0, 255, (424, 512), dtype=np.uint8)
    crop = eye_mouth_crop(img, _face(), size=64)
    assert crop.shape == (64, 64) and crop.dtype == np.uint8


def test_eegnet_and_facenet_shapes():
    eeg = EEGNet(n_channels=5, n_samples=512)
    out = eeg({"eeg": torch.randn(4, 5, 512)})
    assert out.shape == (4, 2) and eeg.embed_dim == 256
    face = FaceNet(n_feats=10)
    batch = {"crops": torch.rand(3, 10, 64, 64), "face_feats": torch.randn(3, 10)}
    assert face(batch).shape == (3, 2)
    assert face.embed(batch).shape == (3, face.embed_dim)
    assert count_parameters(eeg) < 5_000 and count_parameters(face) < 50_000


def test_binary_metrics():
    m = binary_metrics([1, 1, 1, 0, 0], [1, 0, 1, 0, 1])
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["recall"] == pytest.approx(2 / 3) and m["fnr"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(0.5)
    no_pos = binary_metrics([0, 0], [0, 1])
    assert np.isnan(no_pos["recall"]) and no_pos["accuracy"] == 0.5
    s = summarise_folds([m, no_pos])
    assert s["recall"]["folds"] == 1 and s["accuracy"]["folds"] == 2


def test_choose_val_subjects_prefers_both_classes():
    subjects = np.repeat([1, 2, 3, 4, 5], 2)
    labels = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])  # subject 2 is alert-only
    assert choose_val_subjects(1, subjects, labels, 2) == [3, 4]
    assert choose_val_subjects(5, subjects, labels, 2) == [1, 3]


def test_feature_zscore_handles_nan():
    x = np.array([[1.0], [3.0], [np.nan], [10.0], [20.0]], np.float32)
    z = zscore_features_per_subject(x, np.array([1, 1, 1, 2, 2]))
    assert z[2, 0] == 0 and z[0, 0] == pytest.approx(-1, abs=1e-3)


def _synthetic_dataset(n_subjects=4, per=24):
    """EEG where decline windows carry extra 10 Hz power, so the task is learnable."""
    rng = np.random.default_rng(0)
    t = np.arange(512) / 128
    subj = np.repeat(np.arange(1, n_subjects + 1), per)
    label = np.tile(np.r_[np.zeros(per // 2), np.ones(per // 2)], n_subjects).astype(int)
    eeg = rng.normal(0, 1, (len(subj), 5, 512)).astype(np.float32)
    eeg += (label[:, None, None] * 2 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
    return {"eeg": eeg, "keep": np.ones(len(subj), bool), "label": label, "subject": subj,
            "test": np.ones(len(subj), int), "window_in_session": np.tile(np.arange(per), n_subjects)}


def test_run_loso_end_to_end(tmp_path):
    inputs = build_inputs(_synthetic_dataset())
    cfg = TrainConfig(model="eeg", epochs=15, patience=15, batch_size=16, lr=3e-3,
                      model_kwargs={"n_channels": 5, "n_samples": 512})
    summary = run_loso(cfg, inputs, tmp_path, folds=[1, 2], log=lambda *_: None)
    assert (tmp_path / "predictions.csv").exists() and (tmp_path / "fold_subject1.pt").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["model"] == "eeg"
    assert summary["pooled"]["n"] == 48
    assert summary["pooled"]["accuracy"] > 0.8  # an easy signal must be learned
