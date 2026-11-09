"""Turn raw DROZY sessions into windowed arrays saved under ``data/processed/drozy``.

Each session becomes one ``<subject>-<test>.npz`` with:

* ``eeg``        float32 ``[n, channels, samples]`` filtered EEG windows (volts)
* ``keep``       bool ``[n]`` False for windows rejected for extreme amplitude
* ``band_power`` float32 ``[n, channels, bands]`` log band power
* ``frame_idx``  int64 ``[n, frames_per_window]`` video frames for each window
                 (all -1 if the session has no video; with video, windows not
                 fully covered by it are dropped)
* ``window_index`` int64 ``[n]`` position of each window on the EEG clock
                 (window i covers ``[i*window_s, (i+1)*window_s)`` seconds)
* ``label``      int ``[n]`` 1 = vigilance decline (KSS >= 7), 0 = alert
* ``subject``, ``test``, ``kss``, ``sfreq``, ``channels``, ``fps``

Per-subject z-scoring is applied later, at training time, so the saved
windows stay in physical units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .drozy import (DrozySession, camera_start_s, discover_sessions, kss_to_label, load_eeg,
                    read_frame_times)
from .eeg import band_powers, make_windows, preprocess_raw
from .video import video_info, window_frame_indices, window_frame_indices_from_times

MAX_AUTO_OFFSET_S = 120.0


def video_offset(session: DrozySession, raw, setting) -> float:
    """Seconds between EEG start and first video frame (positive = video starts later).

    With ``setting == "auto"`` it uses the first edge of the EDF's Cam-Sync
    channel (exact, on the EEG clock). Without that channel it falls back to
    the EDF start time vs. the first line of the timestamp file (wall clock;
    EDF stores whole seconds, so this is only accurate to about 1 s).
    Implausible values (over 2 minutes) fall back to 0 with a warning.
    """
    if setting != "auto":
        return float(setting)
    synced = camera_start_s(session.edf_path)
    if synced is not None:
        return synced
    meas = raw.info.get("meas_date")
    if session.timestamps_path is None or meas is None:
        print(f"[warn] {session.name}: cannot compute video offset automatically, using 0")
        return 0.0
    _, first_frame = read_frame_times(session.timestamps_path)
    offset = (first_frame - meas.replace(tzinfo=None)).total_seconds()
    if abs(offset) > MAX_AUTO_OFFSET_S:
        print(f"[warn] {session.name}: computed video offset {offset:.1f} s looks wrong, using 0")
        return 0.0
    return offset


def prepare_session(session: DrozySession, cfg: dict) -> dict:
    d, e, lab = cfg["data"], cfg["eeg"], cfg["labels"]
    window_s = d["window_seconds"]

    raw = preprocess_raw(load_eeg(session, e["channels"]), bandpass=e["bandpass_hz"],
                         notch=e["notch_hz"], resample_hz=e["resample_hz"])
    sfreq = raw.info["sfreq"]
    data = raw.get_data()

    windows, keep = make_windows(data, sfreq, window_s, max_ptp=e["reject_ptp_uv"] * 1e-6)
    window_index = np.arange(len(windows))

    fps, offset = float("nan"), float("nan")
    if session.video_path is not None:
        fps, n_frames = video_info(session.video_path)
        offset = video_offset(session, raw, d["video_offset_s"])
        if session.timestamps_path is not None:
            times, _ = read_frame_times(session.timestamps_path)
            times = times[:n_frames]
            frame_idx = window_frame_indices_from_times(len(windows), times, window_s,
                                                        d["frames_per_window"], offset)
        else:
            frame_idx = window_frame_indices(len(windows), fps, window_s,
                                             d["frames_per_window"], offset)
        # keep only windows whose frames all fall inside the video
        inside = ((frame_idx >= 0) & (frame_idx < n_frames)).all(axis=1)
        windows, keep, frame_idx = windows[inside], keep[inside], frame_idx[inside]
        window_index = window_index[inside]
    else:
        frame_idx = np.full((len(windows), d["frames_per_window"]), -1, dtype=np.int64)
    n = len(windows)

    return {
        "eeg": windows,
        "keep": keep,
        "band_power": band_powers(windows, sfreq, e["bands"]) if n
        else np.empty((0, windows.shape[1], len(e["bands"])), dtype=np.float32),
        "frame_idx": frame_idx,
        "window_index": window_index,
        "label": np.full(n, kss_to_label(session.kss, lab["kss_decline_min"]), dtype=np.int64),
        "subject": session.subject,
        "test": session.test,
        "kss": session.kss,
        "sfreq": sfreq,
        "channels": np.array(raw.ch_names),
        "fps": fps,
        "video_offset_s": offset,
    }


def prepare_drozy(cfg: dict, root: str | Path | None = None, out: str | Path | None = None,
                  overwrite: bool = False) -> list[Path]:
    root = Path(root or Path(cfg["data"]["root"]) / "drozy")
    out = Path(out or Path(cfg["data"]["processed"]) / "drozy")
    out.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(root)
    if not sessions:
        raise FileNotFoundError(f"No <subject>-<test>.edf files found under {root}")

    written = []
    for s in sessions:
        target = out / f"{s.name}.npz"
        if s.kss is None:
            print(f"[skip] {s.name}: no KSS rating")
            continue
        if target.exists() and not overwrite:
            print(f"[keep] {s.name}: already prepared")
            written.append(target)
            continue
        arrays = prepare_session(s, cfg)
        np.savez_compressed(target, **arrays)
        n, kept = len(arrays["label"]), int(arrays["keep"].sum())
        video = "video" if s.video_path else "NO VIDEO"
        print(f"[done] {s.name}: KSS {s.kss} -> label {arrays['label'][0] if n else '-'}, "
              f"{n} windows ({kept} clean), {video}")
        written.append(target)
    return written


def prepare_face(cfg: dict, root: str | Path | None = None, processed: str | Path | None = None,
                 overwrite: bool = False) -> list[Path]:
    """Add facial data for every prepared session, saved to ``<processed>/drozy_face/``.

    Each ``<subject>-<test>.npz`` holds, aligned row-for-row with the EEG file:

    * ``crops``  uint8 ``[n, frames_per_window, size, size]`` eyes (top) + mouth (bottom)
    * ``feats``  float32 ``[n, len(FEATURE_NAMES)]`` EAR, MAR, PERCLOS, blinks, head pose

    The closed-eye threshold for PERCLOS and blinks is set per subject from
    all of that subject's frames, so it does not use labels.
    """
    from .face import (closed_eye_threshold, eye_aspect_ratio, head_pose, load_landmarks,
                       mouth_aspect_ratio, read_crops, window_features)

    root = Path(root or Path(cfg["data"]["root"]) / "drozy")
    processed = Path(processed or cfg["data"]["processed"])
    eeg_dir, out = processed / "drozy", processed / "drozy_face"
    out.mkdir(parents=True, exist_ok=True)
    size, window_s = cfg["face"]["crop_size"], cfg["data"]["window_seconds"]

    sessions = [s for s in discover_sessions(root)
                if (eeg_dir / f"{s.name}.npz").exists() and s.landmarks_path is not None]
    written = []
    for subject in sorted({s.subject for s in sessions}):
        subj = [s for s in sessions if s.subject == subject]
        todo = [s for s in subj if overwrite or not (out / f"{s.name}.npz").exists()]
        written += [out / f"{s.name}.npz" for s in subj if s not in todo]
        if not todo:
            continue
        lms = {s.name: load_landmarks(s.landmarks_path, s.landmarks3d_path) for s in subj}
        thr = closed_eye_threshold(np.concatenate([eye_aspect_ratio(lms[s.name][0]) for s in subj]))
        for s in todo:
            s2, s3 = lms[s.name]
            eeg = np.load(eeg_dir / f"{s.name}.npz")
            times, _ = read_frame_times(s.timestamps_path)
            n_total = int(eeg["window_index"].max()) + 1 if len(eeg["window_index"]) else 0
            feats = window_features(eye_aspect_ratio(s2), mouth_aspect_ratio(s2), head_pose(s3),
                                    times[: len(s2)], n_total, window_s,
                                    float(eeg["video_offset_s"]), thr)[eeg["window_index"]]
            crops = read_crops(s.video_path, eeg["frame_idx"], s2, size)
            np.savez_compressed(out / f"{s.name}.npz", crops=crops, feats=feats)
            print(f"[face] {s.name}: {len(crops)} windows, closed-eye EAR < {thr:.3f}, "
                  f"mean PERCLOS {np.nanmean(feats[:, 2]):.2f}")
            written.append(out / f"{s.name}.npz")
    return written


def load_prepared(folder: str | Path, face_folder: str | Path | None = None) -> dict[str, np.ndarray]:
    """Concatenate all prepared sessions into one dataset with per-window subject ids.

    If ``face_folder`` is given, ``crops`` and ``face_feats`` are added for the
    sessions found there; sessions without face data are dropped so both
    modalities always line up.
    """
    paths = sorted(Path(folder).glob("*.npz"))
    if face_folder is not None:
        paths = [p for p in paths if (Path(face_folder) / p.name).exists()]
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    if not parts:
        raise FileNotFoundError(f"No prepared .npz files in {folder}")
    n = [len(p["label"]) for p in parts]
    data = {
        "eeg": np.concatenate([p["eeg"] for p in parts]),
        "keep": np.concatenate([p["keep"] for p in parts]),
        "band_power": np.concatenate([p["band_power"] for p in parts]),
        "frame_idx": np.concatenate([p["frame_idx"] for p in parts]),
        "label": np.concatenate([p["label"] for p in parts]),
        "subject": np.repeat([int(p["subject"]) for p in parts], n),
        "test": np.repeat([int(p["test"]) for p in parts], n),
        "window_in_session": np.concatenate([np.arange(k) for k in n]),
    }
    if face_folder is not None:
        faces = [np.load(Path(face_folder) / p.name) for p in paths]
        data["crops"] = np.concatenate([f["crops"] for f in faces])
        data["face_feats"] = np.concatenate([f["feats"] for f in faces])
    return data
