"""Write a portable Hoi! recording manifest and per-recording interaction windows."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    args = parser.parse_args()
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    windows = args.output / "windows"
    windows.mkdir()
    recordings = []
    for uid, clips in boundaries.items():
        if Path(uid).name != uid or uid.count("__") != 1 or not clips:
            raise ValueError(f"Invalid recording ID or missing boundaries for {uid}")
        (windows / f"{uid}.json").write_text(json.dumps(clips, indent=2) + "\n", encoding="utf-8")
        recordings.append({
            "recording_uid": uid,
            "label_root": os.path.relpath(args.labels_dir / uid, args.output),
            "geometry_root": os.path.relpath(args.geometry_dir / uid, args.output),
        })
    (args.output / "recordings.json").write_text(
        json.dumps({"recordings": recordings}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
