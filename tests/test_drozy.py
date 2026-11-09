"""Tests for the DROZY pipeline using small synthetic files with DROZY's layout."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from driver_monitoring.data.drozy import (discover_sessions, kss_to_label, load_eeg, parse_kss,
                                         read_frame_times)
from driver_monitoring.data.eeg import band_powers, make_windows, zscore_per_subject
from driver_monitoring.data.prepare import load_prepared, prepare_drozy
from driver_monitoring.data.video import (read_frames, window_frame_indices,
                                          window_frame_indices_from_times)

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())
SFREQ, SECONDS, FPS = 512, 22, 30
EDF_START = datetime(2015, 3, 2, 14, 30, 0)


def _write_edf(path: Path, seconds: float = SECONDS, alpha_hz: float = 10.0) -> None:
    import mne

    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    names = ["EEG Fz", "EEG Cz", "EEG C3", "EEG C4", "EEG Pz", "EOG-V", "ECG"]
    data = rng.normal(0, 5e-6, (len(names), t.size)) + 20e-6 * np.sin(2 * np.pi * alpha_hz * t)
    info = mne.create_info(names, SFREQ, ["eeg"] * 5 + ["eog", "ecg"])
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(EDF_START.replace(tzinfo=timezone.utc))
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="error")


def _write_timestamps(path: Path, start: datetime, seconds: float, fps: float, drop_every=0) -> None:
    lines = []
    for i in range(int(seconds * fps)):
        if drop_every and i % drop_every == drop_every - 1:
            continue  # simulate a dropped frame
        elapsed = i * 1000.0 / fps
        t = start + timedelta(milliseconds=elapsed)
        lines.append(f"{t.year} {t.month} {t.day} {t.hour} {t.minute} {t.second} "
                     f"{t.microsecond // 1000} {elapsed:.0f}")
    path.write_text("\n".join(lines) + "\n")


def _write_video(path: Path, seconds: float) -> None:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(int(seconds * FPS)):
        frame = np.full((48, 64, 3), i % 256, np.uint8)
        w.write(frame)
    w.release()


@pytest.fixture()
def drozy_root(tmp_path: Path) -> Path:
    root = tmp_path / "drozy"
    (root / "psg").mkdir(parents=True)
    (root / "videos_i8").mkdir()
    # subject 1: tests 1-3; subject 2: test 2 missing (NaN), test 3 has no EDF
    (root / "KSS.txt").write_text("3 6 8\n2 NaN 9\n")
    for name in ["1-1", "1-2", "1-3", "2-1"]:
        _write_edf(root / "psg" / f"{name}.edf")
    for name in ["1-1", "1-2", "1-3"]:
        _write_video(root / "videos_i8" / f"{name}.mp4", seconds=18)  # shorter than EEG
    return root


def test_parse_kss_skips_missing(tmp_path):
    p = tmp_path / "KSS.txt"
    p.write_text("3 6 8\n2\tNaN\t9\n\n4,5,-\n")
    assert parse_kss(p) == {(1, 1): 3, (1, 2): 6, (1, 3): 8, (2, 1): 2, (2, 3): 9, (3, 1): 4, (3, 2): 5}


def test_parse_kss_skips_zero_placeholder(tmp_path):
    p = tmp_path / "KSS.txt"
    p.write_text("\n".join(["3 6 8"] * 6 + ["0 7 9"]) + "\n")  # DROZY: test 7-1 = 0
    kss = parse_kss(p)
    assert (7, 1) not in kss and kss[(7, 2)] == 7


def test_frame_times_handle_dropped_frames(tmp_path):
    p = tmp_path / "1-1.txt"
    _write_timestamps(p, EDF_START, seconds=10, fps=30, drop_every=4)
    times, start = read_frame_times(p)
    assert start == EDF_START and times[0] == 0
    idx = window_frame_indices_from_times(2, times, window_s=4, frames_per_window=10)
    # each chosen frame is within one frame period of its target time
    targets = (np.arange(2)[:, None] * 4 + (np.arange(10) + 0.5) * 0.4).ravel()
    assert np.abs(times[idx.ravel()] - targets).max() <= 1 / 30 + 1e-9
    past_end = window_frame_indices_from_times(3, times, window_s=4, frames_per_window=10)
    assert (past_end[2] == -1).any()  # 8-12 s runs past the 10 s video


def test_auto_offset_from_timestamps(drozy_root, tmp_path):
    ts = drozy_root / "timestamps"
    ts.mkdir()
    # video starts 2 s after EEG and runs 18 s -> covers EEG 2..20 s -> windows 1..4
    _write_timestamps(ts / "1-1.txt", EDF_START + timedelta(seconds=2), seconds=18, fps=FPS)
    prepare_drozy(CFG, drozy_root, tmp_path / "p")
    s = np.load(tmp_path / "p" / "1-1.npz")
    assert float(s["video_offset_s"]) == pytest.approx(2.0)
    assert s["eeg"].shape[0] == 4


def test_parse_kss_rejects_out_of_range(tmp_path):
    p = tmp_path / "KSS.txt"
    p.write_text("3 12 8\n")
    with pytest.raises(ValueError):
        parse_kss(p)


def test_kss_threshold():
    assert [kss_to_label(k) for k in range(1, 10)] == [0] * 6 + [1] * 3


def test_discover_sessions(drozy_root):
    sessions = discover_sessions(drozy_root)
    assert [s.name for s in sessions] == ["1-1", "1-2", "1-3", "2-1"]
    assert [s.kss for s in sessions] == [3, 6, 8, 2]
    assert [s.has_both_modalities for s in sessions] == [True, True, True, False]


def test_load_eeg_picks_and_renames_channels(drozy_root):
    raw = load_eeg(discover_sessions(drozy_root)[0])
    assert raw.ch_names == ["Fz", "Cz", "C3", "C4", "Pz"]
    assert raw.info["sfreq"] == SFREQ


def test_make_windows_shapes_and_rejection():
    data = np.zeros((5, 128 * 10))
    data[2, 128 * 4 + 10] = 1.0  # spike in window 1
    windows, keep = make_windows(data, 128, 4.0, max_ptp=0.5)
    assert windows.shape == (2, 5, 512)
    assert keep.tolist() == [True, False]


def test_zscore_per_subject():
    rng = np.random.default_rng(1)
    w = np.concatenate([rng.normal(5, 2, (10, 3, 50)), rng.normal(-3, 7, (10, 3, 50))])
    z = zscore_per_subject(w, np.repeat([1, 2], 10))
    for s in (slice(0, 10), slice(10, 20)):
        assert np.allclose(z[s].mean(axis=(0, 2)), 0, atol=1e-5)
        assert np.allclose(z[s].std(axis=(0, 2)), 1, atol=1e-4)


def test_band_power_peaks_in_alpha():
    t = np.arange(512) / 128
    w = np.sin(2 * np.pi * 10 * t)[None, None, :]
    bp = band_powers(w, 128, CFG["eeg"]["bands"])
    assert bp.shape == (1, 1, 4)
    assert np.argmax(bp[0, 0]) == list(CFG["eeg"]["bands"]).index("alpha")


def test_frame_alignment():
    idx = window_frame_indices(3, fps=30, window_s=4, frames_per_window=10)
    assert idx.shape == (3, 10)
    assert idx[0, 0] == 6 and idx[1, 0] == 126  # 0.2 s and 4.2 s at 30 fps
    assert np.all(np.diff(idx.ravel()) > 0)
    shifted = window_frame_indices(1, fps=30, window_s=4, frames_per_window=10, offset_s=-1.0)
    assert shifted[0, 0] == 36  # video started 1 s before EEG
    late = window_frame_indices(2, fps=30, window_s=4, frames_per_window=10, offset_s=2.0)
    assert late[0, 0] < 0 and late[1, 0] >= 0  # first window precedes the video


def test_prepare_drops_windows_outside_video(drozy_root, tmp_path):
    cfg = yaml.safe_load(yaml.safe_dump(CFG))
    cfg["data"]["video_offset_s"] = 2.0  # video covers EEG 2..20 s -> windows 1..4
    prepare_drozy(cfg, drozy_root, tmp_path / "p")
    s = np.load(tmp_path / "p" / "1-1.npz")
    assert s["eeg"].shape[0] == 4
    assert s["frame_idx"].min() >= 0


def test_read_frames_returns_requested_frames(drozy_root):
    path = drozy_root / "videos_i8" / "1-1.mp4"
    frames = read_frames(path, np.array([[0, 5], [10, 5]]))
    assert frames.shape == (2, 2, 48, 64)
    assert abs(int(frames[0, 1].mean()) - 5) <= 2  # lossy codec tolerance
    assert np.array_equal(frames[0, 1], frames[1, 1])


def test_prepare_end_to_end(drozy_root, tmp_path):
    out = tmp_path / "processed"
    written = prepare_drozy(CFG, drozy_root, out)
    assert sorted(p.stem for p in written) == ["1-1", "1-2", "1-3", "2-1"]

    s = np.load(out / "1-3.npz")
    assert s["eeg"].shape == (4, 5, 512)        # video (18 s) limits to 4 windows
    assert s["frame_idx"].shape == (4, 10)
    assert s["label"].tolist() == [1] * 4         # KSS 8 -> decline
    assert s["band_power"].shape == (4, 5, 4)
    assert s["keep"].all()

    no_video = np.load(out / "2-1.npz")
    assert no_video["eeg"].shape[0] == 5          # EEG only: 22 s -> 5 windows
    assert (no_video["frame_idx"] == -1).all()

    ds = load_prepared(out)
    assert ds["eeg"].shape[0] == 4 + 4 + 4 + 5
    assert ds["subject"].tolist() == [1] * 12 + [2] * 5
    assert ds["label"].sum() == 4
