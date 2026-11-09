"""Video-to-window alignment (proposal section 3.5).

Face detection and landmark features come in a later step; this module only
decides which frames belong to each EEG window and reads them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def video_info(path: str | Path) -> tuple[float, int]:
    """Return ``(fps, frame_count)`` of a video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, count


def window_frame_indices(n_windows: int, fps: float, window_s: float = 4.0,
                         frames_per_window: int = 10, offset_s: float = 0.0) -> np.ndarray:
    """Evenly spaced frame indices for each window: ``[n_windows, frames_per_window]``.

    Window ``i`` covers EEG time ``[i*window_s, (i+1)*window_s)``; frames are taken
    at the centres of ``frames_per_window`` equal sub-intervals. ``offset_s`` is
    the video start time relative to the EEG start (positive = video starts later).
    """
    starts = np.arange(n_windows)[:, None] * window_s
    centres = (np.arange(frames_per_window) + 0.5) * window_s / frames_per_window
    return np.floor((starts + centres - offset_s) * fps).astype(np.int64)


def read_frames(path: str | Path, indices: np.ndarray, gray: bool = True) -> np.ndarray:
    """Read the given frame indices (any shape) into an array ``[*indices.shape, H, W]``."""
    import cv2

    flat = indices.ravel()
    order = np.argsort(flat)
    cap = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    for i in flat[order]:
        if i in frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise OSError(f"Could not read frame {i} from {path}")
        frames[int(i)] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray else frame
    cap.release()
    stacked = np.stack([frames[int(i)] for i in flat])
    return stacked.reshape(*indices.shape, *stacked.shape[1:])
