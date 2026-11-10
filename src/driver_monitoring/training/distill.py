"""Knowledge distillation from a Teacher to a smaller Student (proposal 3.7.4).

    L_KD = (1 - α) CE(y, σ(z_s)) + α T² KL(σ(z_t / T) || σ(z_s / T))

For each leave-one-subject-out fold the Teacher trained on that fold (same
train and validation subjects, so the test subject is never seen) labels the
training windows. A Student is trained for every (T, α) in the grid and the
one with the lowest hard-label validation loss is evaluated on the test
subject. Test results of every grid point are also kept, for reporting only.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..evaluation.metrics import binary_metrics
from ..models.branches import build_model, count_parameters
from .data import to_batch
from .loso import (TrainConfig, _predict, choose_val_subjects, cpu_state, fold_rows, load_fold,
                   resolve_device, save_fold, train_one, write_results)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor,
            temperature: float, alpha: float) -> torch.Tensor:
    hard = F.cross_entropy(student_logits, y)
    soft = F.kl_div(F.log_softmax(student_logits / temperature, 1),
                    F.softmax(teacher_logits / temperature, 1), reduction="batchmean")
    return (1 - alpha) * hard + alpha * temperature ** 2 * soft


def teacher_logits(model: torch.nn.Module, inputs: dict, idx: np.ndarray,
                   batch_size: int = 256) -> np.ndarray:
    """Teacher logits for ``idx``, as an [N, 2] array over all windows (zeros elsewhere)."""
    model.eval()
    out = np.zeros((len(inputs["label"]), 2), np.float32)
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            out[b] = model(to_batch(inputs, b, dev)).cpu().numpy()
    return out


def _grid_label(t: float, a: float) -> str:
    return f"T{t:g}_a{a:g}"


def run_distillation(cfg: TrainConfig, inputs: dict, teacher_run: str | Path, teacher_model: str,
                     teacher_kwargs: dict, out_dir: str | Path, temperatures=(2, 4, 8),
                     alphas=(0.3, 0.5, 0.7), folds: list[int] | None = None, log=print) -> dict:
    out_dir, teacher_run = Path(out_dir), Path(teacher_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects, labels = inputs["subject"], inputs["label"]
    folds = folds or sorted(set(subjects.tolist()))
    grid = list(itertools.product(temperatures, alphas))
    n_params = count_parameters(build_model(cfg.model, **cfg.model_kwargs))
    log(f"Student {cfg.model}: {n_params:,} parameters; grid of {len(grid)} (T, alpha); {len(folds)} folds")

    rows, fold_metrics, chosen, minutes = [], [], {}, 0.0
    grid_rows = {_grid_label(t, a): [] for t, a in grid}
    for k, test in enumerate(folds, 1):
        done = load_fold(out_dir, test)
        if done is None:
            done = _distil_fold(cfg, inputs, teacher_run, teacher_model, teacher_kwargs, out_dir,
                                test, grid, log, f"[fold {k}/{len(folds)}]")
        else:
            log(f"[fold {k}/{len(folds)}] test subject {test}: already done, loaded")
        m = done["metrics"]
        fold_metrics.append(m)
        chosen[test] = {"temperature": m["temperature"], "alpha": m["alpha"]}
        rows += [tuple(r) for r in done["rows"]]
        for name, r in done["grid"].items():
            grid_rows[name] += r
        minutes += done["minutes"]

    # Test results of each fixed grid point across all folds (for reporting, not selection).
    grid_summary = {}
    for name, r in grid_rows.items():
        y = np.array([row[3] for row in r])
        p = np.array([row[4] for row in r])
        grid_summary[name] = binary_metrics(y, p >= 0.5)

    config = {**asdict(replace(cfg)), "teacher_run": str(teacher_run), "teacher_model": teacher_model,
              "teacher_kwargs": teacher_kwargs, "temperatures": list(temperatures), "alphas": list(alphas)}
    return write_results(out_dir, cfg.model, config, n_params, rows, fold_metrics, minutes,
                         extra={"chosen_per_fold": chosen, "grid_test_pooled": grid_summary})


def _distil_fold(cfg, inputs, teacher_run, teacher_model, teacher_kwargs, out_dir, test, grid, log, tag) -> dict:
    """Train one Student per (T, alpha) on this fold; keep the one with the lowest validation loss."""
    t0 = time.time()
    subjects, labels = inputs["subject"], inputs["label"]
    val_s = choose_val_subjects(test, subjects, labels, cfg.n_val_subjects)
    test_idx = np.flatnonzero(subjects == test)
    val_idx = np.flatnonzero(np.isin(subjects, val_s))
    train_idx = np.flatnonzero(~np.isin(subjects, [test, *val_s]))
    teacher = build_model(teacher_model, **teacher_kwargs)
    teacher.load_state_dict(torch.load(teacher_run / f"fold_subject{test}.pt", weights_only=True,
                                       map_location="cpu"))
    teacher.to(resolve_device(cfg.device))
    logits = teacher_logits(teacher, inputs, train_idx)
    log(f"{tag} test subject {test}, val {val_s}")

    best, grid_rows = None, {}
    for t, a in grid:
        model, info = train_one(cfg, inputs, train_idx, val_idx, log=lambda *_: None, test_subject=test,
                                kd={"logits": logits, "temperature": t, "alpha": a})
        prob = _predict(model, inputs, test_idx)
        grid_rows[_grid_label(t, a)] = fold_rows(inputs, test_idx, prob)
        log(f"    T={t:g} alpha={a:g}: val loss {info['best_val_loss']:.3f} ({info['epochs_run']} epochs)")
        if best is None or info["best_val_loss"] < best[0]:
            best = (info["best_val_loss"], t, a, model, prob, info)

    _, t, a, model, prob, info = best
    torch.save(cpu_state(model), out_dir / f"fold_subject{test}.pt")
    m = binary_metrics(labels[test_idx], prob >= 0.5)
    m.update({"subject": test, "val_subjects": val_s, "temperature": t, "alpha": a,
              "epochs_run": info["epochs_run"]})
    done = {"metrics": m, "rows": fold_rows(inputs, test_idx, prob), "grid": grid_rows,
            "minutes": (time.time() - t0) / 60}
    save_fold(out_dir, test, done)
    log(f"    chose T={t:g} alpha={a:g}: acc {m['accuracy']:.3f}  recall {m['recall']:.3f}  "
        f"specificity {m['specificity']:.3f}  ({done['minutes']:.1f} min)")
    return done
