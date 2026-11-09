import numpy as np
import pytest
import torch
import yaml

from driver_monitoring.models.branches import build_model, count_parameters
from driver_monitoring.training.distill import kd_loss, run_distillation, teacher_logits
from driver_monitoring.training.loso import TrainConfig, run_loso
from test_fusion import EEG_KW, FACE_KW, _synthetic


def test_kd_loss_reduces_to_cross_entropy_and_matches_teacher():
    s = torch.tensor([[2.0, -1.0], [0.5, 0.3]])
    y = torch.tensor([0, 1])
    ce = torch.nn.functional.cross_entropy(s, y)
    assert torch.isclose(kd_loss(s, torch.randn(2, 2), y, 4, 0.0), ce)
    # soft term is zero when the student copies the teacher
    assert torch.isclose(kd_loss(s, s, y, 4, 1.0), torch.tensor(0.0), atol=1e-6)
    assert kd_loss(s, -s, y, 4, 1.0) > 0


def test_student_is_at_least_70_percent_smaller():
    cfg = yaml.safe_load(open("configs/default.yaml"))["distillation"]

    def size(spec):
        return count_parameters(build_model(spec["strategy"], eeg_kwargs={**EEG_KW, **spec["eeg"]},
                                            face_kwargs={**FACE_KW, **spec["face"]}, hidden=spec["hidden"]))
    assert size(cfg["student"]) <= (1 - cfg["target_size_reduction"]) * size(cfg["teacher"])


def test_teacher_logits_only_fill_requested_windows():
    inputs = _synthetic()
    model = build_model("early", eeg_kwargs=EEG_KW, face_kwargs=FACE_KW)
    out = teacher_logits(model, inputs, np.arange(10))
    assert out.shape == (64, 2) and np.abs(out[10:]).sum() == 0 and np.abs(out[:10]).sum() > 0


def test_distillation_end_to_end(tmp_path):
    inputs = _synthetic()
    quiet = lambda *_: None  # noqa: E731
    small = {"eeg_kwargs": {**EEG_KW, "f1": 4, "f2": 8}, "face_kwargs": {**FACE_KW, "width": 8, "hidden": 8},
             "hidden": 8}
    big = {"eeg_kwargs": EEG_KW, "face_kwargs": FACE_KW}
    base = dict(epochs=2, patience=2, batch_size=16, lr=3e-3)
    run_loso(TrainConfig(model="early", model_kwargs=big, **base), inputs, tmp_path / "t", [1, 2], quiet)
    s = run_distillation(TrainConfig(model="early", model_kwargs=small, **base), inputs, tmp_path / "t",
                         "early", big, tmp_path / "s", temperatures=(2, 4), alphas=(0.5,), folds=[1, 2], log=quiet)
    assert s["pooled"]["n"] == 32
    assert set(s["grid_test_pooled"]) == {"T2_a0.5", "T4_a0.5"}
    assert all(v["temperature"] in (2, 4) for v in s["chosen_per_fold"].values())


def test_loso_resumes_finished_folds(tmp_path):
    inputs = _synthetic()
    cfg = TrainConfig(model="eeg", model_kwargs=EEG_KW, epochs=1, patience=1, batch_size=16)
    run_loso(cfg, inputs, tmp_path, [1], lambda *_: None)
    logs = []
    merged = run_loso(cfg, inputs, tmp_path, [1, 2], logs.append)
    assert any("already done" in line for line in logs)
    assert merged["pooled"]["n"] == 32
