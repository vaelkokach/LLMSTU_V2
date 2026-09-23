"""End-to-end orchestration: download -> crop -> caption -> package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import crop as crop_mod
from . import caption as caption_mod
from . import dataset_io as io
from .config import Config, load


def run_all(cfg: Optional[Config] = None, token: Optional[str] = None,
            caption_limit: Optional[int] = None) -> dict:
    cfg = cfg or load()
    token = token or os.environ.get("HF_TOKEN")
    work = Path(cfg.data.workdir)
    frames_dir = work / "frames"
    crops_dir = work / "crops"
    manifest = work / "crops_manifest.jsonl"
    labels = work / "pseudo_labels.jsonl"
    parquet = work / "dataset" / "pseudo_labels.parquet"

    # 1. frames (skip download if already present locally)
    if not any(frames_dir.glob(cfg.data.frames_glob)):
        frames_dir = io.download_frames(
            cfg.data.source_repo, cfg.data.source_subdir, work / "frames_dl", token)

    # 2. crop
    crop_mod.run(io.iter_frames(frames_dir, cfg.data.frames_glob),
                 crops_dir, cfg.crop, manifest_path=manifest)

    # 3. caption
    caption_mod.run(manifest, crops_dir, labels, cfg.caption, limit=caption_limit)

    # 4. package
    io.build_parquet(labels, parquet)
    return {"crops_dir": str(crops_dir), "manifest": str(manifest),
            "labels": str(labels), "parquet": str(parquet)}
