"""Model size and per-window inference latency (proposal Table 3.1)."""

from __future__ import annotations

import io
import time

import numpy as np
import torch


def size_mb(model: torch.nn.Module) -> float:
    """Size of the saved float32 weights (state dict) in megabytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def example_window(n_channels: int = 5, n_samples: int = 512, frames: int = 10, crop: int = 64,
                   n_feats: int = 10) -> dict:
    """One random input window (batch size 1), shaped like the real data."""
    return {"eeg": torch.randn(1, n_channels, n_samples), "crops": torch.rand(1, frames, crop, crop),
            "face_feats": torch.randn(1, n_feats)}


def latency_ms(model: torch.nn.Module, batch: dict, runs: int = 200, warmup: int = 20,
               threads: int | None = 1) -> dict:
    """Mean and 95th-percentile time for one window, batch size 1, as a deployed system would run.

    One thread by default, to be closer to a small edge CPU than a desktop's
    full core count.
    """
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model.eval()
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            t = time.perf_counter()
            model(batch)
            if i >= warmup:
                times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(old)
    return {"mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))}
