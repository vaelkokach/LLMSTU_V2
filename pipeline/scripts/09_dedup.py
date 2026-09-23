#!/usr/bin/env python
"""Remove near-duplicate crops (1-fps footage) BETWEEN cropping and captioning.

  python scripts/09_dedup.py --manifest work/crops_manifest.jsonl \
      --crops work/crops --out work/crops_manifest_dedup.jsonl

Then caption the deduped manifest:
  python scripts/02_caption.py --manifest work/crops_manifest_dedup.jsonl \
      --crops work/crops --out work/pseudo_labels.jsonl

--threshold is the dHash Hamming distance below which two crops of the same seat are
treated as duplicates. Higher = more aggressive (drops more). 6 is a good start;
try 4 (gentler) or 8-10 (more aggressive) and eyeball the kept crops.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--manifest", default="./work/crops_manifest.jsonl")
    ap.add_argument("--crops", default="./work/crops")
    ap.add_argument("--out", default="./work/crops_manifest_dedup.jsonl")
    ap.add_argument("--threshold", type=int, default=None, help="dHash Hamming cutoff")
    ap.add_argument("--cell", type=int, default=None, help="seat-region cell size (px)")
    args = ap.parse_args()

    cfg = load(args.config)
    dedup.dedup_manifest(
        Path(args.manifest), Path(args.crops), Path(args.out),
        hamming_threshold=args.threshold if args.threshold is not None else cfg.dedup.hamming_threshold,
        cell_px=args.cell if args.cell is not None else cfg.dedup.cell_px,
        hash_size=cfg.dedup.hash_size)


if __name__ == "__main__":
    main()
