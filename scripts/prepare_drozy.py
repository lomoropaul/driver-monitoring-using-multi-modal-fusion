"""Prepare DROZY for training.

Put the extracted DROZY download in data/raw/drozy/ (any sub-folder layout), then:

    .venv\\Scripts\\python.exe scripts\\prepare_drozy.py --list    # check what was found
    .venv\\Scripts\\python.exe scripts\\prepare_drozy.py           # EEG -> data/processed/drozy/
    .venv\\Scripts\\python.exe scripts\\prepare_drozy.py --face    # face -> data/processed/drozy_face/
"""

import argparse
from pathlib import Path

import yaml

from driver_monitoring.data.drozy import discover_sessions, kss_to_label
from driver_monitoring.data.prepare import prepare_drozy, prepare_face

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    ap.add_argument("--root", type=Path, help="DROZY folder (default: data/raw/drozy)")
    ap.add_argument("--out", type=Path, help="output folder (default: data/processed/drozy)")
    ap.add_argument("--list", action="store_true", help="only list discovered sessions")
    ap.add_argument("--face", action="store_true",
                    help="build face crops and features (run after the EEG step)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for key in ("root", "processed"):
        cfg["data"][key] = str(ROOT / cfg["data"][key])

    if args.list:
        root = args.root or Path(cfg["data"]["root"]) / "drozy"
        sessions = discover_sessions(root)
        thr = cfg["labels"]["kss_decline_min"]
        print(f"{'session':<8} {'KSS':>3}  label    video")
        for s in sessions:
            label = "-" if s.kss is None else ("decline" if kss_to_label(s.kss, thr) else "alert")
            print(f"{s.name:<8} {s.kss if s.kss is not None else '-':>3}  {label:<8} "
                  f"{s.video_path.name if s.video_path else 'MISSING'}")
        both = sum(s.has_both_modalities and s.kss is not None for s in sessions)
        print(f"\n{len(sessions)} sessions, {both} with EEG + video + KSS, "
              f"{len({s.subject for s in sessions})} subjects")
        return

    if args.face:
        written = prepare_face(cfg, args.root, args.out, args.overwrite)
        print(f"\n{len(written)} sessions with face data")
        return

    written = prepare_drozy(cfg, args.root, args.out, args.overwrite)
    print(f"\n{len(written)} sessions ready in {args.out or cfg['data']['processed'] + '/drozy'}")


if __name__ == "__main__":
    main()
