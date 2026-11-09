"""DROZY (ULg Multimodality Drowsiness Database) loader.

DROZY has 14 subjects, each doing up to three 10-minute psychomotor vigilance
tests (PVT) with increasing sleep deprivation. A session is named
``<subject>-<test>``, e.g. ``3-2`` is subject 3, test 2. Per session there is:

* a polysomnography recording (EDF) with EEG channels Fz, Cz, C3, C4, Pz plus
  EOG, ECG and EMG, sampled at 512 Hz
* a Kinect v2 near-infrared intensity video
* one KSS rating (1-9), stored for all sessions in ``KSS.txt``

Some sessions are missing in the release, so every lookup here tolerates gaps.
File names are matched by the ``<subject>-<test>`` stem anywhere under the
dataset root, so the exact folder names in the download do not matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SESSION_RE = re.compile(r"^(\d{1,2})-(\d)$")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")
DEFAULT_EEG_CHANNELS = ("Fz", "Cz", "C3", "C4", "Pz")


@dataclass(frozen=True)
class DrozySession:
    subject: int
    test: int
    edf_path: Path
    video_path: Path | None
    kss: int | None

    @property
    def name(self) -> str:
        return f"{self.subject}-{self.test}"

    @property
    def has_both_modalities(self) -> bool:
        return self.video_path is not None


def parse_kss(path: str | Path) -> dict[tuple[int, int], int]:
    """Read ``KSS.txt``: one line per subject, one rating per test.

    Line ``i`` (1-based) holds subject ``i``'s ratings for tests 1, 2, 3,
    separated by spaces, tabs, commas or semicolons. Missing ratings may be
    written as ``NaN``, ``-``, ``x`` or left empty; they are skipped.
    """
    ratings: dict[tuple[int, int], int] = {}
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    for subject, line in enumerate(lines, start=1):
        for test, token in enumerate(re.split(r"[\s,;]+", line.strip()), start=1):
            try:
                value = float(token)
            except ValueError:
                continue
            if np.isnan(value):
                continue
            if not 1 <= value <= 9:
                raise ValueError(f"KSS {value} out of range for session {subject}-{test}")
            ratings[(subject, test)] = int(round(value))
    return ratings


def kss_to_label(kss: int, decline_min: int = 7) -> int:
    """Binary label from the proposal: 1 = vigilance decline (KSS >= 7), 0 = alert."""
    return int(kss >= decline_min)


def _index_by_stem(root: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and SESSION_RE.match(p.stem):
            found.setdefault(p.stem, p)
    return found


def discover_sessions(root: str | Path, kss_file: str = "KSS.txt") -> list[DrozySession]:
    """Find every session that has an EDF file, pairing it with its video and KSS."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DROZY folder not found: {root}")

    kss_paths = list(root.rglob(kss_file))
    kss = parse_kss(kss_paths[0]) if kss_paths else {}
    edfs = _index_by_stem(root, (".edf",))
    videos = _index_by_stem(root, VIDEO_EXTS)

    sessions = []
    for stem, edf in edfs.items():
        subject, test = map(int, SESSION_RE.match(stem).groups())
        sessions.append(DrozySession(subject, test, edf, videos.get(stem), kss.get((subject, test))))
    return sorted(sessions, key=lambda s: (s.subject, s.test))


def load_eeg(session: DrozySession, channels=DEFAULT_EEG_CHANNELS):
    """Load the session's EEG channels as an MNE Raw object (other channels dropped).

    Channel names are matched case-insensitively and ignoring prefixes such as
    ``EEG `` that some EDF writers add.
    """
    import mne

    raw = mne.io.read_raw_edf(session.edf_path, preload=True, verbose="error")
    lookup = {re.sub(r"^(eeg[\s-]*)", "", ch, flags=re.I).strip().lower(): ch for ch in raw.ch_names}
    picked = [lookup[c.lower()] for c in channels if c.lower() in lookup]
    missing = [c for c in channels if c.lower() not in lookup]
    if not picked:
        raise ValueError(f"{session.name}: none of {channels} found in {raw.ch_names}")
    if missing:
        print(f"[drozy] {session.name}: missing EEG channels {missing}")
    raw.pick(picked)
    raw.rename_channels({old: new for old, new in zip(picked, [c for c in channels if c not in missing])})
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    return raw
