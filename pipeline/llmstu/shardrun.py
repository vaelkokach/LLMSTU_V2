"""Process ONE source shard end-to-end and stream it to the Hub.

download frames -> crop (pose+quality) -> dedup -> caption -> upload crops+labels
-> mark done -> wipe local disk. Resumable: a shard with a _done marker is skipped,
so a VM wipe only costs the shard in progress.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .config import Config
from . import crop as crop_mod
from . import caption as caption_mod
from . import dataset_io as io


def _rmtree(*paths):
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)


def process_one(cfg: Config, shard: str, crops_repo: str, token: Optional[str],
                cap=None, cleanup: bool = True) -> dict:
    """Run one shard. `cap` = preloaded captioner to reuse across shards."""
    if io.shard_done(crops_repo, shard, token):
        print(f"[shard {shard}] already done on {crops_repo} — skipping")
        return {"shard": shard, "skipped": True}

    work = Path(cfg.data.workdir)
    frames_dl = work / "frames_dl"
    crops = work / "crops"
    manifest = work / "manifest.jsonl"
    manifest_dedup = work / "manifest_dedup.jsonl"
    labels = work / "labels.jsonl"
    stage = work / "stage"
    _rmtree(frames_dl, crops, stage)                 # clean slate for this shard
    for f in (manifest, manifest_dedup, labels):
        f.unlink(missing_ok=True)

    # 1. get this shard's frames: one tar (fast, rate-limit-safe) or 5000 files
    if cfg.data.use_tar_frames:
        tar_repo = cfg.data.frames_tar_repo or cfg.data.source_repo
        shard_dir = io.download_frames_tar(tar_repo, cfg.data.frames_tar_subdir,
                                           shard, frames_dl, token=token)
    else:
        frames_root = io.download_frames(cfg.data.source_repo, cfg.data.source_subdir,
                                         frames_dl, token=token, shards=[shard])
        shard_dir = frames_root / shard

    # 2. crop (pose + quality gates)
    crop_mod.run(io.iter_frames(shard_dir, "*.jpg"), crops, cfg.crop,
                 manifest_path=manifest)

    # 3. dedup near-duplicate crops
    io_dedup(cfg, manifest, crops, manifest_dedup)

    # 4. caption the deduped crops (reuse the preloaded model)
    caption_mod.run(manifest_dedup, crops, labels, cfg.caption, cap=cap)

    # 5. stage + upload crops + labels, then mark done
    n = io.stage_shard(crops, labels, stage, shard, shard_size=cfg.data.upload_shard_size)
    io.ensure_repo(crops_repo, token=token)
    io.upload_dir(stage, crops_repo, token=token)
    io.mark_shard_done(crops_repo, shard, token=token)
    print(f"[shard {shard}] uploaded {n} crops + labels -> {crops_repo}")

    if cleanup:
        _rmtree(frames_dl, crops, stage)
        for f in (manifest, manifest_dedup, labels):
            f.unlink(missing_ok=True)
    return {"shard": shard, "crops": n, "skipped": False}


def io_dedup(cfg: Config, manifest, crops, out):
    from . import dedup
    dedup.dedup_manifest(manifest, crops, out,
                         hamming_threshold=cfg.dedup.hamming_threshold,
                         cell_px=cfg.dedup.cell_px, hash_size=cfg.dedup.hash_size)
