"""Configuration loading. Reads config.yaml and allows env overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class CropConfig:
    detector_model: str = "yolo11m.pt"   # ultralytics person detector (bbox path)
    person_class_id: int = 0
    conf: float = 0.35
    iou: float = 0.6
    imgsz: int = 1280                    # frames are wide-angle; keep resolution high
    crop_mode: str = "head_wide"         # head_wide | upper_body | person (bbox path only)
    head_frac: float = 0.55              # top fraction of person box treated as head+shoulders
    margin_frac: float = 0.35            # extra context padding around the crop
    min_side_px: int = 48                # drop crops smaller than this (final crop box)
    max_per_frame: int = 60
    out_size: int = 448                  # square resize written to disk (0 = keep native)
    # --- pose-based, quality-gated cropping (recommended) ---
    use_pose: bool = True                # crop from head keypoints -> guarantees a head in-frame
    pose_model: str = "yolo11m-pose.pt"  # ultralytics pose model
    down_extend: float = 2.5             # how far below shoulders to extend (x head span) ->
                                         #   include hands/desk/phone (raise for more, capped at person box)
    kpt_conf: float = 0.35               # min confidence for a keypoint to count
    min_head_kpts: int = 1               # require >=this many head keypoints (nose/eyes/ears)
    min_face_kpts: int = 2               # require >=this many FACE keypoints (nose/eyes) ->
                                         #   drops faces occluded behind monitors (0 disables)
    min_native_px: int = 120             # min head span in ORIGINAL px (raise -> stricter on pixelation)
    blur_min_var: float = 40.0           # min Laplacian variance (below -> blurry; 0 disables)


@dataclass
class CaptionConfig:
    # Qwen3.5 is natively multimodal (vision built into the base model). The weights
    # are downloaded and run ON the GPU via transformers/vLLM — NO inference API is
    # called. AutoModelForImageTextToText picks the right class automatically.
    model_id: str = "Qwen/Qwen3.5-27B"   # see docs/COLAB_A100.md for size vs GPU
    backend: str = "transformers"        # transformers | vllm
    dtype: str = "bfloat16"
    load_in_4bit: bool = False           # bitsandbytes 4-bit (fits 27B on a 40GB A100)
    attn_implementation: str = "sdpa"    # sdpa | flash_attention_2 | eager
    prompt_name: str = "v1_structured"   # key in llmstu/prompts.PROMPTS
    schema_name: str = "full"            # key in llmstu/schema.SCHEMAS (field set to use)
    enable_thinking: bool = False        # False = tell Qwen to skip <think> and emit JSON directly
    max_new_tokens: int = 512
    batch_size: int = 8                  # transformers micro-batch
    max_pixels: int = 768 * 28 * 28      # cap visual tokens per crop for speed
    temperature: float = 0.0
    # vLLM backend tuning (used only when backend == "vllm")
    vllm_gpu_mem_util: float = 0.90      # fraction of VRAM vLLM may claim
    vllm_max_num_seqs: int = 64          # concurrent sequences (raise on big cards)


@dataclass
class DataConfig:
    source_repo: str = "CHANGE_ME/your-frames-dataset"           # frames live here (frames/ subdir)
    source_subdir: str = "frames"
    crops_repo: str = "CHANGE_ME/your-crops-dataset"      # where enhanced crops+labels are pushed
    workdir: str = "./work"
    frames_glob: str = "**/*.jpg"
    upload_shard_size: int = 5000               # crops per shard folder on upload (<10k HF limit)
    # tar-packed frames (1 file/shard instead of 5000 -> avoids HF rate limits)
    use_tar_frames: bool = False                # True once you've run scripts/15_repack_frames.py
    frames_tar_repo: str = ""                   # default = source_repo
    frames_tar_subdir: str = "frames_tar"       # where shard tars live in the repo


@dataclass
class DedupConfig:
    hamming_threshold: int = 6   # dHash distance below which same-seat crops = duplicate
    cell_px: int = 64            # seat-region grid size for grouping crops in space
    hash_size: int = 8           # dHash resolution (size*size bits)


@dataclass
class EvalConfig:
    n_samples: int = 1500                       # final golden set size (1.5-2k is plenty)
    stratify_by: str = "engagement_level"       # legacy single-field (03 now multi-field)
    seed: int = 13


@dataclass
class Config:
    crop: CropConfig = field(default_factory=CropConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(dc, d: Dict[str, Any]):
    for k, v in (d or {}).items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load(path: Optional[str] = None) -> Config:
    cfg = Config()
    path = path or os.environ.get("LLMSTU_CONFIG", "config.yaml")
    p = Path(path)
    if p.exists() and yaml is not None:
        raw = yaml.safe_load(p.read_text()) or {}
        _merge(cfg.crop, raw.get("crop", {}))
        _merge(cfg.caption, raw.get("caption", {}))
        _merge(cfg.data, raw.get("data", {}))
        _merge(cfg.dedup, raw.get("dedup", {}))
        _merge(cfg.eval, raw.get("eval", {}))
    # env overrides for the things you tweak most on a cloud box
    cfg.caption.model_id = os.environ.get("LLMSTU_MODEL_ID", cfg.caption.model_id)
    cfg.caption.backend = os.environ.get("LLMSTU_BACKEND", cfg.caption.backend)
    cfg.data.source_repo = os.environ.get("LLMSTU_SOURCE_REPO", cfg.data.source_repo)
    cfg.data.crops_repo = os.environ.get("LLMSTU_CROPS_REPO", cfg.data.crops_repo)
    cfg.data.workdir = os.environ.get("LLMSTU_WORKDIR", cfg.data.workdir)
    return cfg
