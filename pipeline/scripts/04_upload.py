#!/usr/bin/env python
"""Package crops + captions into an HF image dataset and push it (resumable).

Produces in the repo:
  crops/shard_000/*.jpg ...     (sharded <10k files/dir)
  crops/metadata.jsonl          (each crop image paired with its caption + label fields)
  pseudo_labels.parquet         (the full label table)
  golden_eval.json              (if you pass --golden)

  python scripts/04_upload.py --crops ./work/crops \
      --labels ./work/pseudo_labels.jsonl --repo CHANGE_ME/your-crops-dataset
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--crops", default="./work/crops")
    ap.add_argument("--labels", default="./work/pseudo_labels.jsonl")
    ap.add_argument("--golden", default="./labeling/golden.json",
                    help="exported golden labels from the labeling tool (optional)")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--copy", action="store_true",
                    help="copy crops into the staging dir instead of moving them")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    shard_size = args.shard_size or cfg.data.upload_shard_size
    stage = Path(cfg.data.workdir) / "dataset"
    stage.mkdir(parents=True, exist_ok=True)

    # 1. shard crops + write metadata.jsonl (caption travels with each image)
    io.build_crops_dataset(Path(args.crops), Path(args.labels), stage,
                           shard_size=shard_size, move=not args.copy)
    # 2. full label table as parquet
    io.build_parquet(Path(args.labels), stage / "pseudo_labels.parquet")
    # 3. golden eval set, if present
    if Path(args.golden).exists():
        shutil.copy(args.golden, stage / "golden_eval.json")

    # 4. push (resumable large-folder upload)
    io.upload_dataset(stage, repo, token=args.token, private=not args.public)


if __name__ == "__main__":
    main()
