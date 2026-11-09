"""EEG preprocessing and windowing (proposal section 3.6.1)."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def preprocess_raw(raw, bandpass=(0.5, 45.0), notch=50.0, resample_hz: float | None = 128.0,
                   reference: str | None = "average"):
    """Band-pass, mains notch, re-reference and resample an MNE Raw object in place.

    ICA artefact removal is left out here: with DROZY's 5 EEG channels ICA cannot
    separate ocular and muscle sources reliably. Extreme-amplitude windows are
    rejected in :func:`make_windows` instead.
    """
    raw.filter(bandpass[0], bandpass[1], verbose="error")
    if notch and notch < raw.info["sfreq"] / 2:
        raw.notch_filter(notch, verbose="error")
    if reference:
        raw.set_eeg_reference(reference, projection=False, verbose="error")
    if resample_hz and resample_hz != raw.info["sfreq"]:
        raw.resample(resample_hz, verbose="error")
    return raw


def make_windows(data: np.ndarray, sfreq: float, window_s: float = 4.0,
                 max_ptp: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``[channels, samples]`` into non-overlapping windows.

    Returns ``(windows [n, channels, window_samples], keep_mask [n])``. Windows
    whose peak-to-peak amplitude on any channel exceeds ``max_ptp`` (same unit
    as ``data``) are marked ``False`` in the mask but still returned, so they can
    be used for robustness testing as the proposal requires.
    """
    step = int(round(window_s * sfreq))
    n = data.shape[1] // step
    if n == 0:
        return np.empty((0, data.shape[0], step), dtype=np.float32), np.zeros(0, bool)
    windows = data[:, : n * step].reshape(data.shape[0], n, step).transpose(1, 0, 2)
    keep = np.ones(n, bool)
    if max_ptp is not None:
        keep = np.ptp(windows, axis=2).max(axis=1) <= max_ptp
    return windows.astype(np.float32), keep


def zscore_per_subject(windows: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Normalise each channel using the mean/std of that subject's windows only."""
    out = np.empty_like(windows, dtype=np.float32)
    for s in np.unique(subjects):
        idx = subjects == s
        mu = windows[idx].mean(axis=(0, 2), keepdims=True)
        sd = windows[idx].std(axis=(0, 2), keepdims=True) + 1e-8
        out[idx] = (windows[idx] - mu) / sd
    return out


def band_powers(windows: np.ndarray, sfreq: float, bands: dict[str, tuple[float, float]]) -> np.ndarray:
    """Log band power per window, channel and band: ``[n, channels, n_bands]``."""
    freqs, psd = welch(windows, fs=sfreq, nperseg=min(windows.shape[-1], int(2 * sfreq)), axis=-1)
    out = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        out.append(np.log(np.trapezoid(psd[..., mask], freqs[mask], axis=-1) + 1e-12))
    return np.stack(out, axis=-1).astype(np.float32)
