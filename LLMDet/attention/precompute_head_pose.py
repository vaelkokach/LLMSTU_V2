"""Precompute MediaPipe head pose for every LLMSTU crop, in parallel.

WHY: running FaceLandmarker inline inside sequence_builder costs ~140 ms/crop
single-threaded. Over the 283,913 dense crops that is ~11 h and dominates the
build. Pose extraction is embarrassingly parallel and depends only on the crop,
so it is precomputed once here across all cores (~5-10 min on 128) and cached.
sequence_builder then does a dict lookup and the rebuild reverts to being
CLIP-bound (~2 h), same as the 552-dim baseline.

The cache is keyed by the crop's `file_name`, so it is reusable by any later
build, the runtime path, and the P0.3 evaluator.

Usage:
    python -m attention.precompute_head_pose \
        --labels ../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl \
        --crops-root ../grounding_data/LLMSTU/crops \
        --out ../grounding_data/llmstu_tools/outputs/head_pose_cache.npz \
        --workers 96
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_LM = None  # one FaceLandmarker per worker process


def _init(model_path):
    global _LM
    import cv2  # noqa: F401  (ensure per-worker import)
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    _LM = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True))


def _one(args):
    """-> (file_name, yaw, pitch, roll, face_found) with angles in [-1, 1]."""
    import cv2
    import mediapipe as mp
    fn, path = args
    im = cv2.imread(path)
    if im is None or im.size == 0 or min(im.shape[:2]) < 16:
        return fn, 0.0, 0.0, 0.0, 0.0
    res = _LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if not res.facial_transformation_matrixes:
        # face_found = 0. Distinct from "facing forward" — this flag is the
        # strongest single cue signal (head_down 8% vs screen_oriented 92%).
        return fn, 0.0, 0.0, 0.0, 0.0
    M = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
    sy = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    pitch = np.degrees(np.arctan2(-M[2, 0], sy)) / 90.0
    yaw = np.degrees(np.arctan2(M[1, 0], M[0, 0])) / 90.0
    roll = np.degrees(np.arctan2(M[2, 1], M[2, 2])) / 90.0
    c = lambda v: float(np.clip(v, -1.0, 1.0))
    return fn, c(yaw), c(pitch), c(roll), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--model", default="../huggingface/mediapipe/face_landmarker.task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"model bundle missing: {args.model}")

    seen, jobs = set(), []
    for line in open(args.labels):
        fn = json.loads(line)["file_name"]
        if fn in seen:
            continue
        seen.add(fn)
        jobs.append((fn, str(Path(args.crops_root) / fn)))
    print(f"{len(jobs)} unique crops, {args.workers} workers")

    names, vecs, done, found = [], [], 0, 0
    with Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        for fn, y, p, r, f in pool.imap_unordered(_one, jobs, chunksize=args.chunk):
            names.append(fn)
            vecs.append((y, p, r, f))
            found += int(f > 0)
            done += 1
            if done % 20000 == 0:
                print(f"  {done}/{len(jobs)}  face_found={found/done:.1%}", flush=True)

    arr = np.asarray(vecs, dtype=np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.asarray(names), vecs=arr)
    print(f"\nwrote {args.out}: {len(names)} crops, "
          f"face_found {found/max(done,1):.1%}")


if __name__ == "__main__":
    main()
