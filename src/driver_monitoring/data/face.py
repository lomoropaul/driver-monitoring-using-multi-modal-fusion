"""Facial features from DROZY's 68-point landmarks (proposal section 3.6.2).

DROZY ships automatic landmarks for every video frame
(``annotations-auto/<s>-<t>-s2.txt`` in pixels, ``-s3.txt`` in millimetres),
using the standard iBUG 68-point layout:

* 36-41 right eye, 42-47 left eye (subject's right/left)
* 48-59 outer lips, 60-67 inner lips
* 30 nose tip, 8 chin

From these we compute, per frame, the eye aspect ratio (EAR), mouth aspect
ratio (MAR) and head pose, and per window the hand-crafted features listed in
the proposal (EAR, MAR, PERCLOS, blink rate, head pose). We also cut eye and
mouth crops from the sampled frames for the CNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
INNER_MOUTH = list(range(60, 68))

FEATURE_NAMES = ["ear_mean", "ear_min", "perclos", "blinks_per_min", "mar_mean", "mar_max",
                 "pitch_mean", "yaw_mean", "roll_mean", "pitch_std"]


def load_landmarks(s2_path: str | Path, s3_path: str | Path | None = None):
    """Return ``(s2 [frames, 68, 2], s3 [frames, 68, 3] or None)``."""
    s2 = np.loadtxt(s2_path, dtype=np.float32).reshape(-1, 68, 2)
    s3 = np.loadtxt(s3_path, dtype=np.float32).reshape(-1, 68, 3) if s3_path else None
    return s2, s3


def _dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def eye_aspect_ratio(s2: np.ndarray) -> np.ndarray:
    """Mean EAR of both eyes per frame (Soukupova & Cech, 2016)."""
    def one(eye):
        p = s2[:, eye]
        return (_dist(p[:, 1], p[:, 5]) + _dist(p[:, 2], p[:, 4])) / (2 * _dist(p[:, 0], p[:, 3]) + 1e-6)
    return (one(RIGHT_EYE) + one(LEFT_EYE)) / 2


def mouth_aspect_ratio(s2: np.ndarray) -> np.ndarray:
    """Inner-lip opening relative to mouth width per frame (yawning cue)."""
    p = s2[:, INNER_MOUTH]
    vertical = _dist(p[:, 1], p[:, 7]) + _dist(p[:, 2], p[:, 6]) + _dist(p[:, 3], p[:, 5])
    return vertical / (2 * _dist(p[:, 0], p[:, 4]) + 1e-6)


def head_pose(s3: np.ndarray) -> np.ndarray:
    """Approximate (pitch, yaw, roll) in degrees per frame from 3D landmarks.

    Builds a face frame: x from the subject's right to left eye centre (image
    left to right), "down" from the eyes to the chin, and z = x × down, which
    points from the face towards the back of the head (+Z, away from the
    camera) for a frontal face. A frontal, level face gives (0, 0, 0).
    Camera axes: X right, Y down, Z forward.
    """
    re, le = s3[:, RIGHT_EYE].mean(1), s3[:, LEFT_EYE].mean(1)
    x = le - re
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    down = s3[:, 8] - (re + le) / 2
    z = np.cross(x, down)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
    pitch = np.degrees(np.arctan2(z[:, 1], z[:, 2]))   # head tilted down/up (nodding)
    yaw = np.degrees(np.arctan2(z[:, 0], z[:, 2]))     # head turned left/right
    roll = np.degrees(np.arctan2(x[:, 1], x[:, 0]))    # head tilted sideways
    return np.stack([pitch, yaw, roll], axis=1).astype(np.float32)


def closed_eye_threshold(ear: np.ndarray, fraction: float = 0.6) -> float:
    """EAR below which the eye counts as closed (PERCLOS P80 approximation).

    ``fraction`` × the subject's open-eye EAR (90th percentile of all their
    frames). Computed from frames only, never from labels.
    """
    return float(fraction * np.percentile(ear, 90))


def count_blinks(closed: np.ndarray) -> int:
    """Number of open→closed transitions."""
    c = closed.astype(np.int8)
    return int(np.sum((c[1:] == 1) & (c[:-1] == 0)) + (c[0] == 1 if len(c) else 0))


def window_features(ear, mar, pose, frame_times, n_windows, window_s, offset_s, closed_thr):
    """Hand-crafted features for each window: ``[n_windows, len(FEATURE_NAMES)]``.

    Uses every frame whose EEG-clock time ``frame_times + offset_s`` falls in
    the window, not just the sampled frames. Windows with no frames get NaN.
    """
    t = frame_times + offset_s
    win = np.floor(t / window_s).astype(np.int64)
    out = np.full((n_windows, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    for i in range(n_windows):
        m = win == i
        if not m.any():
            continue
        e, closed = ear[m], ear[m] < closed_thr
        out[i] = [e.mean(), e.min(), closed.mean(), count_blinks(closed) * 60.0 / window_s,
                  mar[m].mean(), mar[m].max(), pose[m, 0].mean(), pose[m, 1].mean(),
                  pose[m, 2].mean(), pose[m, 0].std()]
    return out


def _crop(gray: np.ndarray, pts: np.ndarray, out_hw: tuple[int, int], pad: float) -> np.ndarray:
    import cv2

    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = out_hw
    half_w = (x1 - x0) / 2 * (1 + pad)
    half_h = max((y1 - y0) / 2 * (1 + pad), half_w * h / w)  # keep output aspect ratio
    half_w = half_h * w / h
    src = np.float32([[cx - half_w, cy - half_h], [cx + half_w, cy - half_h], [cx - half_w, cy + half_h]])
    dst = np.float32([[0, 0], [w, 0], [0, h]])
    return cv2.warpAffine(gray, cv2.getAffineTransform(src, dst), (w, h), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def eye_mouth_crop(gray: np.ndarray, landmarks: np.ndarray, size: int = 64) -> np.ndarray:
    """``size x size`` image: both eyes on the top half, mouth on the bottom half."""
    half = (size // 2, size)
    eyes = _crop(gray, landmarks[RIGHT_EYE + LEFT_EYE], half, pad=0.35)
    mouth = _crop(gray, landmarks[MOUTH], half, pad=0.35)
    return np.vstack([eyes, mouth])


def read_crops(video_path: str | Path, frame_idx: np.ndarray, s2: np.ndarray, size: int = 64) -> np.ndarray:
    """Read frames sequentially and return crops ``[*frame_idx.shape, size, size]`` uint8."""
    import cv2

    wanted = np.unique(frame_idx)
    crops: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    i, j = 0, 0
    while j < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i == wanted[j]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crops[i] = eye_mouth_crop(gray, s2[min(i, len(s2) - 1)], size)
            j += 1
        i += 1
    cap.release()
    if j < len(wanted):
        raise OSError(f"{video_path}: video ended at frame {i}, needed up to {wanted[-1]}")
    out = np.stack([crops[int(k)] for k in frame_idx.ravel()])
    return out.reshape(*frame_idx.shape, size, size)
