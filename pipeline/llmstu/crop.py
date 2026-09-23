"""Student detection + wide face/upper-body cropping.

Best-practice choices baked in:
  * Ultralytics YOLO (yolo11) person detector, run at high imgsz because the
    source frames are wide-angle rooms with many small students.
  * Each detected person becomes ONE crop. For classroom footage we crop the
    top `head_frac` of the person box (head + shoulders) and pad by
    `margin_frac` so the face keeps surrounding context ("wide angle face").
  * Tiny detections (min_side_px) are dropped so we don't caption a 20px blob.
  * Everything is streamed to JSONL so a 100k-frame run is resumable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from .config import CropConfig
from . import quality

# COCO pose keypoint indices
_HEAD_KPTS = [0, 1, 2, 3, 4]      # nose, eyes, ears (used to anchor the crop box)
_FACE_KPTS = [0, 1, 2]            # nose, left_eye, right_eye (visibility of the actual face)
_SHOULDER_KPTS = [5, 6]


def _load_detector(cfg: CropConfig):
    from ultralytics import YOLO
    return YOLO(cfg.pose_model if cfg.use_pose else cfg.detector_model)


def _wide_face_box(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int, cfg: CropConfig
) -> Tuple[int, int, int, int]:
    """Turn a full-person box into a padded head/upper-body crop box."""
    bw, bh = x2 - x1, y2 - y1
    if cfg.crop_mode == "person":
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    elif cfg.crop_mode == "upper_body":
        cx1, cy1, cx2, cy2 = x1, y1, x2, y1 + 0.65 * bh
    else:  # head_wide (default)
        cx1, cy1, cx2, cy2 = x1, y1, x2, y1 + cfg.head_frac * bh
    # pad with context; heads need a little extra above for hair/hands-on-face
    px, py = cfg.margin_frac * (cx2 - cx1), cfg.margin_frac * (cy2 - cy1)
    ex1 = max(0, int(cx1 - px))
    ey1 = max(0, int(cy1 - py * 1.2))
    ex2 = min(W, int(cx2 + px))
    ey2 = min(H, int(cy2 + py))
    return ex1, ey1, ex2, ey2


def _finalize(img, box, meta, cfg, stats) -> Optional[Tuple[Image.Image, Dict]]:
    """Apply size + blur gates, resize, and return (crop, meta) or None (dropped)."""
    bx1, by1, bx2, by2 = box
    if min(bx2 - bx1, by2 - by1) < cfg.min_side_px:
        stats["too_small"] = stats.get("too_small", 0) + 1
        return None
    crop = img.crop((bx1, by1, bx2, by2))
    if cfg.blur_min_var and cfg.blur_min_var > 0:
        if quality.laplacian_var(crop) < cfg.blur_min_var:   # measure BEFORE upscale
            stats["blurry"] = stats.get("blurry", 0) + 1
            return None
    if cfg.out_size and cfg.out_size > 0:
        crop = _square_pad_resize(crop, cfg.out_size)
    meta["bbox_crop"] = [bx1, by1, bx2, by2]
    stats["kept"] = stats.get("kept", 0) + 1
    return crop, meta


def _crop_pose(img, res, cfg, stats) -> List[Tuple[Image.Image, Dict]]:
    """Head-keypoint-anchored crops. Skips people with no visible head, so a crop
    always contains a face/head (no hand-only or skull-top crops)."""
    W, H = img.size
    out = []
    if res.keypoints is None or res.boxes is None:
        return out
    kps = res.keypoints.data.cpu().numpy()          # (N, 17, 3): x, y, conf
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    order = confs.argsort()[::-1][: cfg.max_per_frame]
    for i in order:
        kp = kps[i]
        face = [kp[j] for j in _FACE_KPTS if kp[j, 2] >= cfg.kpt_conf]
        if cfg.min_face_kpts and len(face) < cfg.min_face_kpts:
            stats["face_occluded"] = stats.get("face_occluded", 0) + 1  # face hidden (monitor) -> drop
            continue
        head = [kp[j] for j in _HEAD_KPTS if kp[j, 2] >= cfg.kpt_conf]
        if len(head) < cfg.min_head_kpts:
            stats["no_head"] = stats.get("no_head", 0) + 1     # no visible head -> drop
            continue
        xs = [p[0] for p in head]
        ys = [p[1] for p in head]
        for j in _SHOULDER_KPTS:                                # add shoulders for context
            if kp[j, 2] >= cfg.kpt_conf:
                xs.append(kp[j, 0]); ys.append(kp[j, 1])
        hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
        head_span = max(hx2 - hx1, hy2 - hy1)
        if head_span < cfg.min_native_px:                       # too far/small -> pixelated
            stats["too_far"] = stats.get("too_far", 0) + 1
            continue
        vspan = max(hy2 - hy1, head_span)                        # vertical unit (head->shoulders)
        px1, py1, px2, py2 = boxes[i]                            # this person's full box
        pw = cfg.margin_frac * (hx2 - hx1 + 1)                   # side padding (widen for hands)
        top = hy1 - cfg.margin_frac * vspan * 1.6               # a little above the head
        # extend DOWN toward desk/lap so hands + phone are visible, but never past
        # this student's own person box (avoids grabbing the row behind).
        bottom = min(hy2 + cfg.down_extend * vspan, py2)
        box = (max(0, int(hx1 - pw)), max(0, int(top)),
               min(W, int(hx2 + pw)), min(H, int(bottom)))
        meta = {"person_idx": int(i),
                "bbox_person": [float(v) for v in boxes[i]],
                "det_conf": float(confs[i]),
                "head_kpts": int(len(head)),
                "face_kpts": int(len(face)),
                "head_span_px": float(head_span)}
        r = _finalize(img, box, meta, cfg, stats)
        if r:
            out.append(r)
    return out


def _crop_bbox(img, res, cfg, stats) -> List[Tuple[Image.Image, Dict]]:
    """Legacy person-box crop (crop_mode head_wide/upper_body/person) + quality gates."""
    W, H = img.size
    out = []
    boxes = res.boxes
    if boxes is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    order = confs.argsort()[::-1][: cfg.max_per_frame]
    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        box = _wide_face_box(x1, y1, x2, y2, W, H, cfg)
        meta = {"person_idx": int(i),
                "bbox_person": [float(x1), float(y1), float(x2), float(y2)],
                "det_conf": float(confs[i])}
        r = _finalize(img, box, meta, cfg, stats)
        if r:
            out.append(r)
    return out


def crop_frame(
    img: Image.Image, detector, cfg: CropConfig, stats: Optional[Dict] = None
) -> List[Tuple[Image.Image, Dict]]:
    """Return list of (crop_image, meta) for one frame. `stats` accumulates drop reasons."""
    if stats is None:
        stats = {}
    res = detector.predict(
        img, classes=[cfg.person_class_id], conf=cfg.conf, iou=cfg.iou,
        imgsz=cfg.imgsz, verbose=False,
    )[0]
    if cfg.use_pose:
        return _crop_pose(img, res, cfg, stats)
    return _crop_bbox(img, res, cfg, stats)


def _square_pad_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def run(
    frames: Iterable[Path],
    out_dir: Path,
    cfg: CropConfig,
    manifest_path: Optional[Path] = None,
    resume: bool = True,
) -> Path:
    """Crop every frame; write crops to out_dir and a JSONL manifest.

    Manifest rows: {crop_path, src_frame, person_idx, bbox_person, bbox_crop, det_conf}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or (out_dir.parent / "crops_manifest.jsonl")

    done_frames = set()
    if resume and manifest_path.exists():
        for line in manifest_path.open():
            try:
                done_frames.add(json.loads(line)["src_frame"])
            except Exception:
                pass

    detector = _load_detector(cfg)
    stats: Dict[str, int] = {}
    n_crops = 0
    with manifest_path.open("a") as mf:
        for frame in frames:
            key = str(frame.name)
            if key in done_frames:
                continue
            try:
                img = Image.open(frame).convert("RGB")
            except Exception:
                continue
            stem = frame.stem
            for crop, meta in crop_frame(img, detector, cfg, stats):
                crop_name = f"{stem}__p{meta['person_idx']:02d}.jpg"
                crop_path = out_dir / crop_name
                crop.save(crop_path, quality=92)
                row = {"crop_path": str(crop_path.relative_to(out_dir.parent)),
                       "src_frame": key, **meta}
                mf.write(json.dumps(row) + "\n")
                n_crops += 1
            mf.flush()
    dropped = {k: v for k, v in stats.items() if k != "kept"}
    mode = "pose" if cfg.use_pose else f"bbox/{cfg.crop_mode}"
    print(f"[crop] {mode}: wrote {n_crops} crops -> {out_dir}")
    if dropped:
        print(f"[crop] dropped -> " + ", ".join(f"{k}:{v}" for k, v in dropped.items())
              + "  (face_occluded=eyes/nose not visible, no_head=no head kpts, "
                "too_far=below min_native_px, blurry=below blur_min_var, "
                "too_small=crop box < min_side_px)")
    return manifest_path
