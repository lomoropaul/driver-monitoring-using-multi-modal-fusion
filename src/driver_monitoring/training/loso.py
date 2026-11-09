"""Leave-one-subject-out training and evaluation (proposal section 3.8, step 1).

For each test subject, two other subjects are held out for validation (early
stopping) and the rest are used for training, so no subject is ever in more
than one of train, validation and test.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..evaluation.metrics import binary_metrics, summarise_folds
from ..models.branches import build_model, count_parameters
from .data import to_batch


@dataclass
class TrainConfig:
    model: str = "eeg"
    epochs: int = 40
    patience: int = 6
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    n_val_subjects: int = 2
    seed: int = 42
    model_kwargs: dict = field(default_factory=dict)


def choose_val_subjects(test: int, subjects: np.ndarray, labels: np.ndarray, n: int = 2) -> list[int]:
    """The ``n`` subjects after ``test`` (cyclically) that have both classes."""
    uniq = sorted(set(subjects.tolist()))
    both = {s for s in uniq if len(set(labels[subjects == s].tolist())) == 2}
    start = uniq.index(test)
    order = uniq[start + 1:] + uniq[:start]
    picked = [s for s in order if s in both][:n]
    return picked if len(picked) == n else [s for s in order][:n]


def _predict(model: nn.Module, inputs: dict, idx: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            logits = model(to_batch(inputs, idx[i:i + batch_size]))
            probs.append(torch.softmax(logits, 1)[:, 1].numpy())
    return np.concatenate(probs) if probs else np.empty(0)


def _loss(model: nn.Module, inputs: dict, idx: np.ndarray, batch_size: int = 256) -> float:
    model.eval()
    total, ce = 0.0, nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            total += ce(model(to_batch(inputs, b)), torch.from_numpy(inputs["label"][b])).item()
    return total / max(len(idx), 1)


def train_one(cfg: TrainConfig, inputs: dict, train_idx: np.ndarray, val_idx: np.ndarray,
              log=print) -> tuple[nn.Module, dict]:
    """Train with AdamW and early stopping on validation loss; returns the best model."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = build_model(cfg.model, **cfg.model_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    best, best_state, bad, history = float("inf"), None, 0, []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        running = 0.0
        for i in range(0, len(order), cfg.batch_size):
            b = order[i:i + cfg.batch_size]
            if len(b) < 2:
                continue  # BatchNorm needs more than one sample
            opt.zero_grad()
            loss = ce(model(to_batch(inputs, b)), torch.from_numpy(inputs["label"][b]))
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        val = _loss(model, inputs, val_idx)
        history.append({"epoch": epoch, "train_loss": running / len(order), "val_loss": val})
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    log(f"    stopped after {len(history)} epochs, best val loss {best:.3f}")
    return model, {"history": history, "best_val_loss": best, "epochs_run": len(history)}


def run_loso(cfg: TrainConfig, inputs: dict, out_dir: str | Path, folds: list[int] | None = None,
             log=print) -> dict:
    """Run leave-one-subject-out CV; writes predictions, per-fold metrics and a summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects, labels = inputs["subject"], inputs["label"]
    folds = folds or sorted(set(subjects.tolist()))
    n_params = count_parameters(build_model(cfg.model, **cfg.model_kwargs))
    log(f"{cfg.model} model: {n_params:,} parameters, {len(folds)} folds, "
        f"{len(labels)} windows ({labels.mean():.1%} decline)")

    rows, fold_metrics, t0 = [], [], time.time()
    for k, test in enumerate(folds, 1):
        val_s = choose_val_subjects(test, subjects, labels, cfg.n_val_subjects)
        test_idx = np.flatnonzero(subjects == test)
        val_idx = np.flatnonzero(np.isin(subjects, val_s))
        train_idx = np.flatnonzero(~np.isin(subjects, [test, *val_s]))
        log(f"[fold {k}/{len(folds)}] test subject {test}, val {val_s}, "
            f"train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} windows")
        model, info = train_one(cfg, inputs, train_idx, val_idx, log)
        prob = _predict(model, inputs, test_idx)
        m = binary_metrics(labels[test_idx], prob >= 0.5)
        m.update({"subject": test, "val_subjects": val_s, "epochs_run": info["epochs_run"]})
        fold_metrics.append(m)
        torch.save(model.state_dict(), out_dir / f"fold_subject{test}.pt")
        for i, p in zip(test_idx, prob):
            rows.append((int(subjects[i]), int(inputs["test"][i]), int(inputs["window"][i]),
                         int(labels[i]), float(p)))
        log(f"    acc {m['accuracy']:.3f}  recall {m['recall']:.3f}  "
            f"specificity {m['specificity']:.3f}  ({time.time() - t0:.0f}s elapsed)")

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "test", "window", "label", "prob_decline"])
        w.writerows(rows)
    with open(out_dir / "fold_metrics.csv", "w", newline="") as f:
        keys = [k for k in fold_metrics[0] if k != "val_subjects"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(fold_metrics)

    y = np.array([r[3] for r in rows])
    p = np.array([r[4] for r in rows])
    summary = {
        "model": cfg.model,
        "parameters": n_params,
        "config": asdict(cfg),
        "pooled": binary_metrics(y, p >= 0.5),
        "per_fold": summarise_folds(fold_metrics),
        "minutes": (time.time() - t0) / 60,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
