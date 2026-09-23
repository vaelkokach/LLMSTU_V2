#!/usr/bin/env python
"""Detect + crop students from frames.

  python scripts/01_crop.py --frames ./work/frames --out ./work/crops
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu import dataset_io as io
from llmstu.config import load
from llmstu import crop as crop_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", help="local frames dir; if omitted, download from HF")
    ap.add_argument("--out", default="./work/crops")
    ap.add_argument("--manifest", default="./work/crops_manifest.jsonl")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    if args.frames:
        frames_dir = Path(args.frames)
    else:
        frames_dir = io.download_frames(cfg.data.source_repo, cfg.data.source_subdir,
                                        Path(cfg.data.workdir) / "frames_dl", args.token)
    crop_mod.run(io.iter_frames(frames_dir, cfg.data.frames_glob),
                 Path(args.out), cfg.crop, manifest_path=Path(args.manifest))


if __name__ == "__main__":
    main()
