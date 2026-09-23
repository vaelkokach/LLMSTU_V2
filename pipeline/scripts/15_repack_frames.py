#!/usr/bin/env python
"""One-time: repack the source frames into ONE tar per shard on the Hub.

Turns CHANGE_ME/your-frames-dataset frames/shard_XXX/ (5000 tiny files) into frames_tar/shard_XXX.tar
(one file). After this, the pipeline downloads 1 file per shard instead of 5000 —
which avoids HF request rate-limits and the per-file Xet reconstruction that chokes
Colab. Resumable: shards whose tar already exists are skipped.

  python scripts/15_repack_frames.py --num-shards 21

This still has to download the 104k frames ONCE (unavoidable), but you pay it a
single time; every run afterwards (and every Colab reconnect) is fast. Best run on
a box with decent CPU/bandwidth if Colab keeps throttling.

Then set in config.yaml:
  data:
    use_tar_frames: true
and re-run scripts/12_run_shards.py as usual.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source-repo", default=None)
    ap.add_argument("--tar-repo", default=None, help="where to put the tars (default: source repo)")
    ap.add_argument("--tar-subdir", default=None)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--num-shards", type=int, default=21)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    src = args.source_repo or cfg.data.source_repo
    tar_repo = args.tar_repo or cfg.data.frames_tar_repo or src
    tar_subdir = args.tar_subdir or cfg.data.frames_tar_subdir
    token = args.token or os.environ.get("HF_TOKEN")
    shards = (args.shards.split(",") if args.shards
              else [f"shard_{i:03d}" for i in range(args.num_shards)])

    work = Path(cfg.data.workdir)
    for shard in shards:
        tar_in_repo = f"{tar_subdir}/{shard}.tar"
        if io.hub_file_exists(tar_repo, tar_in_repo, token):
            print(f"[repack {shard}] tar already on {tar_repo} — skipping")
            continue

        dl = work / "repack_dl"
        tars = work / "repack_tars"
        shutil.rmtree(dl, ignore_errors=True)
        shutil.rmtree(tars, ignore_errors=True)

        frames_root = io.download_frames(src, cfg.data.source_subdir, dl,
                                         token=token, shards=[shard])
        shard_dir = frames_root / shard
        tar_path = tars / f"{shard}.tar"
        n = io.make_shard_tar(shard_dir, tar_path)
        io.upload_one_file(tar_path, tar_in_repo, tar_repo, token=token)
        print(f"[repack {shard}] packed {n} frames -> {tar_repo}/{tar_in_repo}")

        shutil.rmtree(dl, ignore_errors=True)
        shutil.rmtree(tars, ignore_errors=True)

    print("[repack] done. Set data.use_tar_frames: true in config.yaml, then run 12_run_shards.py")


if __name__ == "__main__":
    main()
