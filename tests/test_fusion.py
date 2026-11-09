import numpy as np
import pytest
import torch

from driver_monitoring.evaluation.stats import mcnemar, wilcoxon_folds
from driver_monitoring.models.branches import EEGNet, FaceNet, build_model
from driver_monitoring.models.fusion import EarlyFusion, HybridFusion
from driver_monitoring.training.late_fusion import best_alpha, run_late_fusion
from driver_monitoring.training.loso import TrainConfig, init_branches, run_loso

EEG_KW, FACE_KW = {"n_channels": 5, "n_samples": 512}, {"n_feats": 10}


def _batch(b=4):
    return {"eeg": torch.randn(b, 5, 512), "crops": torch.rand(b, 10, 64, 64),
            "face_feats": torch.randn(b, 10)}


@pytest.mark.parametrize("cls", [EarlyFusion, HybridFusion])
def test_fusion_shapes(cls):
    model = cls(EEG_KW, FACE_KW).eval()
    assert model(_batch()).shape == (4, 2)
    assert model.embed(_batch()).shape == (4, model.embed_dim)


def test_hybrid_gate_weights_sum_to_one():
    model = HybridFusion(EEG_KW, FACE_KW).eval()
    model(_batch(6))
    w = model.last_weights
    assert w.shape == (6, 2) and torch.allclose(w.sum(1), torch.ones(6))


def test_modality_dropout_only_in_training():
    model = EarlyFusion(EEG_KW, FACE_KW, modality_dropout=0.5)
    torch.manual_seed(0)
    model.train()
    h_e, h_f = model._embeddings(_batch(64))
    dropped_e = (h_e.abs().sum(1) == 0)
    dropped_f = (h_f.abs().sum(1) == 0)
    assert dropped_e.any() and dropped_f.any()
    assert not (dropped_e & dropped_f).any()  # never both
    model.eval()
    h_e, h_f = model._embeddings(_batch(64))
    assert (h_e.abs().sum(1) > 0).all() and (h_f.abs().sum(1) > 0).all()


def test_init_branches_loads_baseline_weights(tmp_path):
    eeg, face = EEGNet(**EEG_KW), FaceNet(**FACE_KW)
    (tmp_path / "e").mkdir(), (tmp_path / "f").mkdir()
    torch.save(eeg.state_dict(), tmp_path / "e" / "fold_subject3.pt")
    torch.save(face.state_dict(), tmp_path / "f" / "fold_subject3.pt")
    fused = build_model("hybrid", eeg_kwargs=EEG_KW, face_kwargs=FACE_KW)
    init_branches(fused, {"eeg": tmp_path / "e", "face": tmp_path / "f"}, 3)
    for a, b in zip(fused.eeg.state_dict().values(), eeg.state_dict().values()):
        assert torch.equal(a, b)


def test_best_alpha_prefers_informative_model():
    y = np.array([0, 0, 1, 1] * 10)
    good, noise = np.where(y == 1, 0.9, 0.1), np.full(len(y), 0.5)
    assert best_alpha(y, good, noise) == 1.0
    assert best_alpha(y, noise, good) == 0.0


def test_mcnemar_and_wilcoxon():
    y = np.ones(30, int)
    a = np.ones(30, int)
    b = np.r_[np.ones(10), np.zeros(20)].astype(int)
    r = mcnemar(y, a, b)
    assert r["only_a_correct"] == 20 and r["only_b_correct"] == 0 and r["p_value"] < 1e-5
    assert mcnemar(y, a, a)["p_value"] == 1.0
    w = wilcoxon_folds(np.array([0.9, 0.8, 0.85, 0.95, 0.7, 0.88]),
                       np.array([0.6, 0.5, 0.55, 0.65, 0.4, 0.58]))
    assert w["median_diff"] == pytest.approx(0.3) and w["p_value"] < 0.05


def _synthetic(n_subjects=4, per=16):
    """Decline shows up in EEG (10 Hz) for half the windows and in the face features for the rest."""
    rng = np.random.default_rng(0)
    t = np.arange(512) / 128
    subj = np.repeat(np.arange(1, n_subjects + 1), per)
    label = np.tile(np.r_[np.zeros(per // 2), np.ones(per // 2)], n_subjects).astype(int)
    eeg = rng.normal(0, 1, (len(subj), 5, 512)).astype(np.float32)
    eeg += (label[:, None, None] * 2 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
    feats = rng.normal(0, 1, (len(subj), 10)).astype(np.float32)
    feats[:, 2] += 2 * label
    return {"label": label, "subject": subj, "test": np.ones(len(subj), int),
            "window": np.tile(np.arange(per), n_subjects), "eeg": eeg,
            "crops": rng.integers(0, 255, (len(subj), 10, 64, 64), dtype=np.uint8),
            "face_feats": feats}


def test_fusion_and_late_fusion_end_to_end(tmp_path):
    inputs = _synthetic()
    quiet = lambda *_: None  # noqa: E731
    base = dict(epochs=4, patience=4, batch_size=16, lr=3e-3)
    run_loso(TrainConfig(model="eeg", model_kwargs=EEG_KW, **base), inputs, tmp_path / "eeg", [1, 2], quiet)
    run_loso(TrainConfig(model="face", model_kwargs=FACE_KW, **base), inputs, tmp_path / "face", [1, 2], quiet)

    hybrid = run_loso(TrainConfig(model="hybrid", model_kwargs={"eeg_kwargs": EEG_KW, "face_kwargs": FACE_KW},
                                  init_from={"eeg": tmp_path / "eeg", "face": tmp_path / "face"}, **base),
                      inputs, tmp_path / "hybrid", [1, 2], quiet)
    assert hybrid["pooled"]["n"] == 32

    late = run_late_fusion(inputs, tmp_path / "eeg", tmp_path / "face", tmp_path / "late",
                           EEG_KW, FACE_KW, folds=[1, 2], log=quiet)
    assert late["pooled"]["n"] == 32 and set(late["alpha_per_fold"]) == {1, 2}
    assert (tmp_path / "late" / "predictions.csv").exists()
