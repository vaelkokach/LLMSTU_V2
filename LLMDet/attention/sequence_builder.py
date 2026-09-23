"""Build temporal training sequences for the attention transformer.

Primary path (``--format llmstu``): reads the LLMSTU per-student label jsonl
shards (structured fields), maps each record to a visible-cue class via
:mod:`attention.taxonomy`, groups crops into per-(video, seat) timelines and
emits NPZ sequences with **per-frame** labels, split video-wise into
``train/`` and ``val/`` subdirectories.

Legacy path (``--format legacy``): the old ODVG whole-frame jsonl. Deprecated —
its weak keyword labels are unreliable (they used to be scored against the
whole caption + tags, collapsing every student in a frame to one label; that
bug is fixed here, but the label source itself remains noisy). Kept only for
comparison experiments.

Video identity: LLMSTU filenames encode the source video for only ~2% of
frames, so the builder requires a ``frame_to_video.json`` mapping (produced by
``grounding_data/llmstu_tools``). Seat identity: students are stationary, so
seats are recovered by greedy centroid clustering of ``bbox_person`` within a
video.
"""

import argparse
import json
import random
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from attention.taxonomy import (CUE_CLASSES, candidate_set, map_record,
                                parse_stem_time)

# --- legacy 4-class taxonomy (deprecated) ---------------------------------
LEGACY_CLASS_NAMES = ["attentive", "distracted", "sleeping", "engaged"]
LEGACY_CLASS_TO_ID = {c: i for i, c in enumerate(LEGACY_CLASS_NAMES)}

WEAK_LABEL_MAP = {
    "focused": "attentive",
    "thinking": "attentive",
    "concentrated": "attentive",
    "studying": "attentive",
    "typing": "engaged",
    "writing": "engaged",
    "reading": "engaged",
    "engaged": "engaged",
    "looking at screen": "engaged",
    "looking at monitor": "engaged",
    "curious": "engaged",
    "calm": "attentive",
    "resting hand": "distracted",
    "looking away": "distracted",
    "talking": "distracted",
    "chatting": "distracted",
    "phone": "distracted",
    "idle": "distracted",
    "distracted": "distracted",
    "sleeping": "sleeping",
    "eyes closed": "sleeping",
    "tired": "sleeping",
    "yawning": "sleeping",
}

CLASS_NAMES = CUE_CLASSES  # backward-compatible export


@dataclass
class CropObs:
    src_frame: str
    time_s: float
    bbox_xyxy: List[float]
    label_id: int
    #: Every cue the record supports (taxonomy.candidate_set). label_id is the
    #: precedence winner and is always a member. Kept so a partial-label
    #: objective can credit any candidate instead of only the winner.
    cand_ids: List[int] = field(default_factory=list)
    head_span_px: float = 150.0
    meta: Dict = field(default_factory=dict)


def _region_to_label(region_phrase: str, tags: List[str], caption: str) -> Optional[int]:
    """Legacy weak labeling. Scores ONLY the region's own phrase — the old
    version matched against the whole caption + tags, which made every region
    in a frame share the same label."""
    text = region_phrase.lower()
    scores = defaultdict(int)
    for key, cls in WEAK_LABEL_MAP.items():
        if key in text:
            scores[cls] += 1
    if not scores:
        return None
    cls = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]
    return LEGACY_CLASS_TO_ID[cls]


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


# ---------------------------------------------------------------------------
# LLMSTU path
# ---------------------------------------------------------------------------

def load_frame_to_video(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"frame_to_video mapping at {path} is empty or malformed")
    return mapping


def _video_id_from_filename(fname: str) -> Optional[str]:
    m = re.search(r"(video_\d+[^.]*)", fname)
    return m.group(1) if m else None


#: The record fields every cue rule reads (attention.taxonomy.cue_conditions).
#: Carried through into CropObs.meta so a label set can be recomputed under a
#: different ruleset WITHOUT re-extracting features -- see
#: attention.thesis_eval.build_cue_labels. Kept as an explicit list rather than
#: stashing the whole record: the records also carry captions and absolute
#: image paths, and 284k of those is memory spent on nothing.
CUE_FIELDS = ("activity", "gaze_direction", "attention_target", "posture",
              "hand_state", "engagement_level", "occluded", "face_kpts",
              "phone_visible", "talking")


def parse_llmstu_labels(
    label_paths: List[Path],
    frame_to_video: Optional[Dict[str, str]],
    allow_filename_fallback: bool = False,
    ruleset: str = "v1",
) -> Dict[str, List[CropObs]]:
    """Parse LLMSTU label jsonl shards into per-video observation lists.

    ``ruleset`` picks the cue rule version used for ``label_id``/``cand_ids``.
    It defaults to ``v1``, so the builder and every existing caller are
    unchanged; the sequences on disk were all built under v1.
    """
    by_video: Dict[str, List[CropObs]] = defaultdict(list)
    n_total = 0
    n_unmapped = 0
    for lp in label_paths:
        with lp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_total += 1
                src = rec["src_frame"]
                stem = Path(src).stem
                if frame_to_video is not None and src in frame_to_video:
                    vid = frame_to_video[src]
                elif frame_to_video is not None and stem in frame_to_video:
                    vid = frame_to_video[stem]
                elif allow_filename_fallback:
                    vid = _video_id_from_filename(src) or "video_unknown"
                    n_unmapped += 1
                else:
                    n_unmapped += 1
                    continue
                t = parse_stem_time(stem)
                if t is None:
                    continue
                by_video[vid].append(
                    CropObs(
                        src_frame=src,
                        time_s=t,
                        bbox_xyxy=[float(v) for v in rec["bbox_person"]],
                        label_id=map_record(rec, ruleset),
                        cand_ids=candidate_set(rec, ruleset),
                        head_span_px=float(rec.get("head_span_px", 150.0)),
                        meta={"file_name": rec.get("file_name", ""),
                              **{k: rec.get(k) for k in CUE_FIELDS}},
                    )
                )
    if n_unmapped:
        pct = 100.0 * n_unmapped / max(1, n_total)
        warnings.warn(
            f"{n_unmapped}/{n_total} records ({pct:.1f}%) had no video mapping"
            + (" (used filename fallback)" if allow_filename_fallback else " and were dropped")
        )
    for vid in by_video:
        by_video[vid].sort(key=lambda o: (o.time_s, o.src_frame))
    return by_video


def assign_seats(video_obs: List[CropObs], eps_factor: float = 0.75) -> Dict[int, List[CropObs]]:
    """Cluster stationary students into seats by bbox centroid.

    Greedy nearest-seat assignment: a crop joins the nearest existing seat if
    its centroid is within ``eps_factor * median_head_span`` of the seat's
    running centroid, else it opens a new seat. Adequate because the cameras
    and students are static within a video.
    """
    if not video_obs:
        return {}
    med_span = float(np.median([o.head_span_px for o in video_obs]))
    eps = eps_factor * max(60.0, med_span)
    seats: Dict[int, List[CropObs]] = {}
    centroids: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    next_sid = 0
    for obs in video_obs:
        x1, y1, x2, y2 = obs.bbox_xyxy
        c = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        best_sid, best_d = -1, float("inf")
        for sid, cent in centroids.items():
            d = float(np.linalg.norm(c - cent))
            if d < best_d:
                best_d, best_sid = d, sid
        if best_sid >= 0 and best_d <= eps:
            seats[best_sid].append(obs)
            n = counts[best_sid]
            centroids[best_sid] = (centroids[best_sid] * n + c) / (n + 1)
            counts[best_sid] = n + 1
        else:
            seats[next_sid] = [obs]
            centroids[next_sid] = c
            counts[next_sid] = 1
            next_sid += 1
    return seats


def split_videos(video_ids: List[str], val_fraction: float, seed: int) -> Tuple[set, set]:
    vids = sorted(video_ids)
    rng = random.Random(seed)
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * val_fraction))) if len(vids) > 1 else 0
    val = set(vids[:n_val])
    train = set(vids[n_val:])
    return train, val


def build_sequences_llmstu(
    label_dir: Path,
    image_root: Path,
    output_dir: Path,
    frame_to_video_path: Path,
    min_track_len: int = 8,
    max_track_len: int = 128,
    val_fraction: float = 0.2,
    seed: int = 42,
    allow_filename_fallback: bool = False,
    allow_clip_fallback: bool = False,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    max_gap_s: float = 15.0,
    head_pose_backend: str = None,
    head_pose_cache: str = None,
    affect_cache: str = None,
    head_stream: bool = False,
    object_cache: "Optional[str]" = None,
    dynamic_features: bool = False,
) -> None:
    """Build per-(video, seat) sequences with per-frame cue labels.

    Sequences are cut whenever the seat disappears for more than ``max_gap_s``
    or when ``max_track_len`` frames are accumulated. NPZ fields:
    ``x`` [T, D] features, ``y_frames`` [T] per-frame labels, ``y_cand`` [T, K]
    multi-hot candidate sets, ``t`` [T] timestamps (s), ``y`` scalar majority
    label (backward compatibility).

    ``y_cand`` records every cue the annotation supports, not just the one the
    precedence rule kept. Loaders that do not know about it are unaffected.
    """
    from attention.features import StudentFeatureExtractor  # deferred: loads CLIP

    import cv2

    label_paths = sorted(label_dir.glob("*.jsonl")) if label_dir.is_dir() else [label_dir]
    if not label_paths:
        raise RuntimeError(f"No label jsonl found at {label_dir}")
    if not frame_to_video_path.exists():
        if not allow_filename_fallback:
            raise RuntimeError(
                f"frame_to_video mapping not found at {frame_to_video_path}. "
                "Generate it with grounding_data/llmstu_tools (video-ID recovery), or pass "
                "--allow-filename-video-fallback (NOT recommended: ~98% of frames lack a "
                "video ID in the filename and will be dropped into 'video_unknown', which "
                "breaks the subject-wise split)."
            )
        frame_to_video = None
    else:
        frame_to_video = load_frame_to_video(frame_to_video_path)

    by_video = parse_llmstu_labels(label_paths, frame_to_video, allow_filename_fallback)
    if not by_video:
        raise RuntimeError("No observations parsed — check label paths and video mapping.")

    train_vids, val_vids = split_videos(list(by_video.keys()), val_fraction, seed)
    # Head pose adds 4 dims (yaw, pitch, roll, face_found) -> 556 total.
    # face_found is the strongest single cue signal measured so far:
    # head_down detects at 8% vs screen_oriented 92% [internal notes, not included].
    hp = None
    if head_pose_backend:
        from attention.head_pose import HeadPoseEstimator
        hp = HeadPoseEstimator(backend=head_pose_backend,
                               cache_path=head_pose_cache)
        if not hp.available():
            raise RuntimeError(f"head-pose backend {head_pose_backend!r} unavailable")
    # backend="cached" is looked up per crop by file_name AFTER the CLIP block,
    # so the extractor itself stays head-pose-free and CLIP batching is
    # unaffected. Inline backends would pay ~140 ms/crop here.
    cached_hp = hp if (hp is not None and head_pose_backend == "cached") else None

    # --- Thesis_Topic.md channels: facial expression + body language + gaze ---
    # affect_cache supplies 4 head-pose + 7 facial-expression dims per crop;
    # dynamic_features adds 7 temporal dims (fidget/lean motion statistics and
    # personalised gaze deviation) computed from the track itself. Both are
    # derived from data already present, so neither needs new annotation.
    affect = None
    if affect_cache:
        d = np.load(affect_cache, allow_pickle=False)
        affect = ({str(n): i for i, n in enumerate(d["names"])},
                  d["vecs"].astype(np.float32))
        print(f"affect cache: {len(affect[0])} crops, {affect[1].shape[1]} dims")
    from attention.dynamic_features import compute_dynamic, DYNAMIC_DIM
    extra = (affect[1].shape[1] if affect is not None else 0) + \
            (DYNAMIC_DIM if dynamic_features else 0)
    # The head stream replaces express/dynamic rather than joining them: those
    # two are not deployable, and mixing all four would produce a column layout
    # no feature config names, which is the one failure that stays silent.
    if head_stream and extra:
        raise SystemExit(
            "--head-stream cannot be combined with --affect-cache or "
            "--dynamic-features: the head block occupies columns 556+, where "
            "express/dynamic live in the v570 layout. Build them separately.")

    # Object presence occupies [1074, 1080), i.e. it sits AFTER the head block,
    # so a v1080_obj build is a v1074_head build plus six columns. Asking for
    # objects without the head stream would leave [556, 1074) undefined and the
    # width assert below would catch it -- but the message would be about a
    # width rather than about the missing flag, so say it here.
    objects = None
    if object_cache:
        if not head_stream:
            raise SystemExit(
                "--object-cache requires --head-stream: the object block is "
                "defined at columns [1074, 1080) of the v1080_obj layout, which "
                "is v1074_head plus six. Without the head stream those columns "
                "do not exist.")
        if extra:
            raise SystemExit(
                "--object-cache cannot be combined with --affect-cache or "
                "--dynamic-features, for the same reason --head-stream cannot.")
        d = np.load(object_cache, allow_pickle=False)
        objects = ({str(n): i for i, n in enumerate(d["names"])},
                   d["vecs"].astype(np.float32))
        print(f"object cache: {len(objects[0])} crops, "
              f"{objects[1].shape[1]} dims, "
              f"objects={[str(x) for x in d['prompts']]}")
    extractor = StudentFeatureExtractor(
        clip_model_name=clip_model_name,
        allow_clip_fallback=allow_clip_fallback,
        head_pose=None if cached_hp is not None else hp,
        head_stream=head_stream)
    if extractor.clip_dim != 512:
        # The embedding width decides every column after it, so a non-CLIP
        # encoder produces a DIFFERENT layout under the same block names. Say so
        # loudly: a v570 config sliced into one of these vectors would read the
        # middle of the embedding as head pose and train without complaint.
        print(f"encoder: {clip_model_name} -> {extractor.clip_dim}-dim embedding "
              f"(NOT the 512-dim CLIP the v570/v1074/v1080 layouts assume)")
    # The layout follows the ENCODER first: a 1152-dim embedding cannot be any
    # of the CLIP layouts whatever else is switched on.
    if extractor.clip_dim != 512:
        if object_cache or head_stream:
            raise SystemExit(
                f"--clip-model {clip_model_name} embeds at {extractor.clip_dim} "
                f"dims; no head-stream or object layout is declared for it yet. "
                f"Build the base layout first.")
        layout_name = "v1196_sig"
    else:
        layout_name = ("v1080_obj" if object_cache
                       else "v1074_head" if head_stream else "v570")
    total_dim = (extractor.output_dim() + (4 if cached_hp is not None else 0)
                 + extra + (objects[1].shape[1] if objects is not None else 0))
    print(f"feature dim: {total_dim} (layout={layout_name}, "
          f"head_pose={head_pose_backend or 'off'}, "
          f"head_stream={'on' if head_stream else 'off'})")
    from attention.thesis_eval.data import LAYOUT_WIDTH
    if total_dim != LAYOUT_WIDTH[layout_name]:
        raise SystemExit(
            f"built width {total_dim} != {LAYOUT_WIDTH[layout_name]} declared "
            f"for layout {layout_name}. Every feature config slices by absolute "
            f"column, so writing this would misalign them silently.")

    for split in ("train", "val"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    n_obj_missing = 0
    meta_rows = []
    frame_cache: Tuple[Optional[str], Optional[np.ndarray]] = (None, None)

    for video_id, obs_list in sorted(by_video.items()):
        split = "train" if video_id in train_vids else "val"
        seats = assign_seats(obs_list)
        for sid, seat_obs in seats.items():
            if len(seat_obs) < min_track_len:
                continue
            # cut into chunks on time gaps / max length
            chunks: List[List[CropObs]] = [[]]
            for obs in seat_obs:
                cur = chunks[-1]
                if cur and (obs.time_s - cur[-1].time_s > max_gap_s or len(cur) >= max_track_len):
                    chunks.append([])
                    cur = chunks[-1]
                cur.append(obs)
            for chunk in chunks:
                if len(chunk) < min_track_len:
                    continue
                feats, labels, times, track_boxes = [], [], [], []
                cand_masks = []
                # Encode the whole chunk in sub-batches before the per-obs loop.
                # This used to call extract() once per crop, i.e. a batch-of-one
                # forward pass each time -- tolerable for CLIP ViT-B/32 at
                # 2.97 ms, but a so400m tower at 384px runs ~3.5x its batched
                # per-crop cost that way, which is the difference between a
                # 3-hour corpus build and an 11-hour one. Sub-batched rather
                # than all at once because a chunk is up to 128 frames and a
                # 1918x1080 frame is ~6 MB decoded.
                ENC_BATCH = 32
                chunk_fv: list = [None] * len(chunk)
                for s0 in range(0, len(chunk), ENC_BATCH):
                    grp = chunk[s0:s0 + ENC_BATCH]
                    pairs, at = [], []
                    for k, o in enumerate(grp):
                        ip = image_root / o.src_frame
                        if frame_cache[0] == str(ip):
                            fr = frame_cache[1]
                        else:
                            fr = cv2.imread(str(ip))
                            frame_cache = (str(ip), fr)
                        if fr is None:
                            continue
                        pairs.append((fr, o.bbox_xyxy)); at.append(s0 + k)
                    if pairs:
                        vs = extractor.extract_many(pairs, batch=ENC_BATCH)
                        for r, idx in enumerate(at):
                            chunk_fv[idx] = vs[r]
                for ci, obs in enumerate(chunk):
                    fv = chunk_fv[ci]
                    if fv is None:                 # unreadable frame
                        continue
                    if cached_hp is not None:
                        fv = np.concatenate(
                            [fv, cached_hp.estimate_by_name(
                                obs.meta.get("file_name", ""))]).astype(np.float32)
                    if affect is not None:
                        amap, avecs = affect
                        j = amap.get(obs.meta.get("file_name", ""))
                        av = avecs[j] if j is not None else np.zeros(
                            avecs.shape[1], dtype=np.float32)
                        fv = np.concatenate([fv, av]).astype(np.float32)
                    if objects is not None:
                        omap, ovecs = objects
                        j = omap.get(obs.meta.get("file_name", ""))
                        # A crop with no cache entry gets zeros, which is what
                        # "no object detected" already looks like. Counted below
                        # so a systematically missing cache is visible rather
                        # than silently training every student as object-free.
                        if j is None:
                            n_obj_missing += 1
                            ov = np.zeros(ovecs.shape[1], dtype=np.float32)
                        else:
                            ov = ovecs[j]
                        fv = np.concatenate([fv, ov]).astype(np.float32)
                    feats.append(fv)
                    track_boxes.append(list(obs.bbox_xyxy))
                    labels.append(obs.label_id)
                    cand_masks.append(obs.cand_ids or [obs.label_id])
                    times.append(obs.time_s)
                if len(feats) < min_track_len:
                    continue
                x = np.stack(feats, axis=0).astype(np.float32)
                if dynamic_features:
                    # Body language + gaze deviation need the WHOLE track:
                    # fidgeting is motion variance over time, and the gaze
                    # baseline is this student's own median pose. Neither can
                    # be computed per frame, which is why the per-frame block
                    # could never represent the two indicators the topic names.
                    pose_cols = None
                    if affect is not None:
                        # pose occupies the first 4 of the affect block
                        a0 = x.shape[1] - affect[1].shape[1]
                        pose_cols = x[:, a0:a0 + 4]
                    elif cached_hp is not None:
                        pose_cols = x[:, -4:]
                    dyn = compute_dynamic(track_boxes, pose_cols)
                    x = np.concatenate([x, dyn], axis=1).astype(np.float32)
                y_frames = np.array(labels, dtype=np.int64)
                # [T, K] multi-hot: 1 where the record supports that cue. A
                # frame always has at least its own label set, so a row is
                # never empty and a partial-label loss never divides by zero.
                y_cand = np.zeros((len(labels), len(CUE_CLASSES)), dtype=np.uint8)
                for _i, _ids in enumerate(cand_masks):
                    y_cand[_i, list(_ids)] = 1
                y_major = int(np.bincount(y_frames).argmax())
                out_name = f"sample_{sample_idx:06d}.npz"
                np.savez_compressed(
                    output_dir / split / out_name,
                    x=x,
                    layout=layout_name,
                    y_frames=y_frames,
                    y_cand=y_cand,
                    y=np.array(y_major, dtype=np.int64),
                    t=np.array(times, dtype=np.float64),
                )
                meta_rows.append(
                    {
                        "file": f"{split}/{out_name}",
                        "video_id": video_id,
                        "seat_id": sid,
                        "length": int(x.shape[0]),
                        "label_majority": y_major,
                        "split": split,
                    }
                )
                sample_idx += 1

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": sample_idx,
                "class_names": CUE_CLASSES,
                "label_source": "llmstu_structured_fields",
                "split": {"train_videos": sorted(train_vids), "val_videos": sorted(val_vids)},
                "seed": seed,
                "samples": meta_rows,
            },
            f,
            indent=2,
        )
    print(f"Built {sample_idx} sequences ({len(train_vids)} train / {len(val_vids)} val videos) in {output_dir}")


# ---------------------------------------------------------------------------
# Legacy ODVG path (deprecated)
# ---------------------------------------------------------------------------

def _extract_video_id(filename: str) -> str:
    m = re.search(r"(video_\d+)", filename)
    return m.group(1) if m else "video_unknown"


def _extract_frame_idx(filename: str) -> int:
    m = re.search(r"_f(\d+)_", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"f(\d+)", filename)
    return int(m.group(1)) if m else 0


@dataclass
class RegionObs:
    filename: str
    frame_idx: int
    video_id: str
    bbox_xyxy: List[float]
    label_id: int


def parse_jsonl(jsonl_path: Path) -> Dict[str, List[RegionObs]]:
    warnings.warn(
        "Legacy ODVG weak-label path is deprecated; use --format llmstu.",
        DeprecationWarning,
    )
    by_video: Dict[str, List[RegionObs]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fname = item["filename"]
            grounding = item.get("grounding", {})
            caption = grounding.get("caption", "")
            tags = item.get("tags", [])
            for reg in grounding.get("regions", []):
                bbox = reg.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                label_id = _region_to_label(reg.get("phrase", ""), tags, caption)
                if label_id is None:
                    continue
                by_video[_extract_video_id(fname)].append(
                    RegionObs(
                        filename=fname,
                        frame_idx=_extract_frame_idx(fname),
                        video_id=_extract_video_id(fname),
                        bbox_xyxy=[float(x) for x in bbox],
                        label_id=label_id,
                    )
                )
    for vid in by_video:
        by_video[vid].sort(key=lambda x: (x.frame_idx, x.filename))
    return by_video


def parse_args():
    p = argparse.ArgumentParser(description="Build temporal training sequences.")
    p.add_argument("--format", type=str, default="llmstu", choices=["llmstu", "legacy"])
    p.add_argument("--labels", type=str, required=True,
                   help="LLMSTU labels dir (shard_*.jsonl) or a single jsonl file.")
    p.add_argument("--image-root", type=str, required=True,
                   help="Directory with source frames (grounding_data/stu_img/frames).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--frame-to-video", type=str,
                   default="../grounding_data/llmstu_tools/frame_to_video.json")
    p.add_argument("--min-track-len", type=int, default=8)
    p.add_argument("--max-track-len", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-filename-video-fallback", action="store_true")
    p.add_argument("--object-cache", default=None,
                   help="npz from attention.precompute_objects: per-crop "
                        "object-presence features (cell phone, laptop). "
                        "Requires --head-stream; produces a v1080_obj build.")
    p.add_argument("--head-stream", action="store_true",
                   help="add the 518-dim head stream: a second CLIP pass over "
                        "the head region cropped from the full frame. A person "
                        "crop resized to 224x224 puts the head on ~1 of "
                        "CLIP-B/32's 49 patches; this gives it all 49. Produces "
                        "the v1074_head layout, which excludes express/dynamic.")
    p.add_argument("--head-pose-backend", type=str, default=None,
                   choices=["mediapipe", "opencv", "cached"],
                   help="enable the 4-dim head-pose block (556-dim features); "
                        "requires retraining the temporal model")
    p.add_argument("--head-pose-cache", type=str, default=None,
                   help="npz from precompute_head_pose.py (for backend=cached)")
    p.add_argument("--affect-cache", type=str, default=None,
                   help="npz from precompute_affect.py: 4 head-pose + 7 facial-"
                        "expression dims per crop (Thesis_Topic facial expressions)")
    p.add_argument("--dynamic-features", action="store_true",
                   help="add 7 temporal dims: fidget/lean motion statistics and "
                        "personalised gaze deviation (Thesis_Topic body language "
                        "+ gaze direction)")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                   help="visual encoder. `google/siglip2-so400m-patch14-384` "
                        "embeds at 1152 dims and therefore builds the v1196_sig "
                        "layout, which is NOT interchangeable with v570.")
    p.add_argument("--allow-clip-fallback", action="store_true",
                   help="Continue with zeroed CLIP features if CLIP fails to load.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.format != "llmstu":
        raise SystemExit("Legacy path is deprecated for building; use --format llmstu.")
    build_sequences_llmstu(
        label_dir=Path(args.labels),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        frame_to_video_path=Path(args.frame_to_video),
        min_track_len=args.min_track_len,
        max_track_len=args.max_track_len,
        val_fraction=args.val_fraction,
        seed=args.seed,
        allow_filename_fallback=args.allow_filename_video_fallback,
        allow_clip_fallback=args.allow_clip_fallback,
        clip_model_name=args.clip_model,
        head_pose_backend=args.head_pose_backend,
        head_pose_cache=args.head_pose_cache,
        affect_cache=args.affect_cache,
        dynamic_features=args.dynamic_features,
        head_stream=args.head_stream,
        object_cache=args.object_cache,
    )
