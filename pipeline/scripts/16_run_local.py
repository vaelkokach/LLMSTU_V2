#!/usr/bin/env python
"""Full run from a SINGLE zip in an HF Storage Bucket (best for rate-limit issues).

Downloads the one zip (one request, no per-file throttling), extracts it, then
processes shard-by-shard FROM LOCAL DISK: crop -> dedup -> caption -> upload
crops+labels to the Hub -> mark done. No per-shard frame downloads at all.
Resumable via the same _done markers, so a disconnect just resumes remaining shards.

  # find the zip's path inside the bucket:
  python scripts/16_run_local.py --bucket CHANGE_ME/your-frames-dataset --list

  # then run it:
  python scripts/16_run_local.py --bucket CHANGE_ME/your-frames-dataset --zip all_frames.zip \
      --repo CHANGE_ME/your-crops-dataset

After all shards finish, use 14_golden_from_hf.py / 07_eval_golden.py / 13_finalize.py
exactly as with the sharded run.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import caption as caption_mod, crop as crop_mod, dedup, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bucket", default="CHANGE_ME/your-frames-dataset")
    ap.add_argument("--zip", default=None, help="zip path inside the bucket")
    ap.add_argument("--repo", default=None, help="crops repo (default: config data.crops_repo)")
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--output", choices=["hf", "zip"], default="hf",
                    help="hf = upload each shard to the Hub; zip = save each shard as a "
                         "local .zip to --out-dir (upload later with 17_upload_zips.py)")
    ap.add_argument("--out-dir", default="./work/out",
                    help="where shard zips go in --output zip mode (point at a mounted "
                         "Google Drive folder to persist across disconnects)")
    ap.add_argument("--token", default=None)
    ap.add_argument("--list", action="store_true", help="list bucket contents and exit")
    args = ap.parse_args()

    cfg = load(args.config)
    token = args.token or os.environ.get("HF_TOKEN")

    if args.list:
        print(f"[bucket {args.bucket}] contents:")
        io.list_bucket(args.bucket, token=token)
        return

    assert args.zip, "pass --zip <path in bucket> (use --list to find it)"
    repo = args.repo or cfg.data.crops_repo
    shard_size = args.shard_size or cfg.data.upload_shard_size
    work = Path(cfg.data.workdir)
    frames_dir = work / "frames_all"

    # 1. download + extract the zip ONCE (skip if frames already extracted)
    if not any(frames_dir.glob("**/*.jpg")):
        extracted = io.download_bucket_zip(args.bucket, args.zip, work / "bucket_dl", token=token)
        frames_dir = extracted
    # recursive glob -> works whether the zip has one flat folder of images or nested dirs
    all_frames = sorted(frames_dir.glob("**/*.jpg"))
    print(f"[local] {len(all_frames)} frames on disk")

    # 2. deterministic shard partition (sorted flat list -> chunks of shard_size).
    #    The source zip can be a single folder of images; we shard it by position here.
    shards = {}
    for i, f in enumerate(all_frames):
        shards.setdefault("shard_%03d" % (i // shard_size), []).append(f)
    names = sorted(shards)
    out_dir = Path(args.out_dir)

    def _done(shard):
        if args.output == "zip":
            return (out_dir / f"{shard}.zip").exists()      # resume by local zip presence
        return io.shard_done(repo, shard, token)

    todo = [s for s in names if not _done(s)]
    print(f"[local] {len(names)} shards, {len(todo)} remaining ({args.output} output): {todo or 'none'}")
    if not todo:
        print("[local] all shards done. Next: scripts/14_golden_from_hf.py "
              "(or 17_upload_zips.py if you saved zips)")
        return

    cap = caption_mod.load_captioner(cfg.caption)      # load model once
    for shard in names:
        if _done(shard):
            continue
        crops = work / "crops"
        stage = work / "stage"
        manifest = work / "manifest.jsonl"
        manifest_dedup = work / "manifest_dedup.jsonl"
        labels = work / "labels.jsonl"
        shutil.rmtree(crops, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        for fp in (manifest, manifest_dedup, labels):
            fp.unlink(missing_ok=True)

        crop_mod.run(iter(shards[shard]), crops, cfg.crop, manifest_path=manifest)
        dedup.dedup_manifest(manifest, crops, manifest_dedup,
                             hamming_threshold=cfg.dedup.hamming_threshold,
                             cell_px=cfg.dedup.cell_px, hash_size=cfg.dedup.hash_size)
        caption_mod.run(manifest_dedup, crops, labels, cfg.caption, cap=cap)
        n = io.stage_shard(crops, labels, stage, shard, shard_size=shard_size)
        if args.output == "zip":
            zp = io.zip_dir(stage, out_dir / f"{shard}.zip")
            print(f"[shard {shard}] saved {n} crops + labels -> {zp}")
        else:
            io.ensure_repo(repo, token=token)
            io.upload_dir(stage, repo, token=token)
            io.mark_shard_done(repo, shard, token=token)
            print(f"[shard {shard}] uploaded {n} crops + labels -> {repo}")
        shutil.rmtree(crops, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)

    if args.output == "zip":
        print(f"[local] done. {len(names)} shard zips in {out_dir}. Download them, then "
              f"later run: python scripts/17_upload_zips.py --zips-dir <dir> --repo {repo}")
    else:
        print("[local] all shards processed. Next: scripts/14_golden_from_hf.py")


if __name__ == "__main__":
    main()
