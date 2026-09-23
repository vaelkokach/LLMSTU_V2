#!/usr/bin/env python
"""Upload the per-shard zips (from 16_run_local.py --output zip) to the crops repo.

Run this LATER, off GPU time — even from your own PC. Each zip contains that shard's
crops/<shard>/... + labels/<shard>.jsonl; this extracts and uploads them, and writes
the _done marker so 14/13 work afterwards. Resumable (skips shards already on HF).

  python scripts/17_upload_zips.py --zips-dir ./downloaded_zips --repo CHANGE_ME/your-crops-dataset
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
    ap.add_argument("--zips-dir", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    token = args.token or os.environ.get("HF_TOKEN")
    zips = sorted(Path(args.zips_dir).glob("*.zip"))
    assert zips, f"no .zip files in {args.zips_dir}"
    io.ensure_repo(repo, token=token)

    for zp in zips:
        shard = zp.stem
        if io.shard_done(repo, shard, token):
            print(f"[upload {shard}] already on {repo} — skipping")
            continue
        stage = Path(cfg.data.workdir) / "unzip" / shard
        shutil.rmtree(stage, ignore_errors=True)
        io.unzip_to(zp, stage)                       # -> stage/crops/... + stage/labels/...
        io.upload_dir(stage, repo, token=token)
        io.mark_shard_done(repo, shard, token=token)
        print(f"[upload {shard}] -> {repo}")
        shutil.rmtree(stage, ignore_errors=True)

    print(f"[upload] done -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
