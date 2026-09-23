"""Precompute MediaPipe head pose from the FULL FRAME + ``bbox_person``.

Why a second precompute exists
------------------------------
``precompute_head_pose.py`` reads the stored 512x512 crops in
``LLMSTU/crops``. Those crops are ``letterbox(frame[bbox_crop], 512)``, and
``bbox_crop`` is a **tighter sub-region** of ``bbox_person`` — median IoU [value removed],
contained in the person box but never containing it [internal notes, not included].

The deployed pipeline has no access to ``bbox_crop``: at inference the only box
available is the detector's, which approximates ``bbox_person``. So the cached
training feature and the live feature were computed from different regions, and
**11% of frames disagreed on ``face_found``** — the single dimension carrying
~80% of the head-pose contribution [internal notes, not included]. Letterboxing the runtime
crop does not close the gap (89.0% -> 89.5% agreement); the region does.

``bbox_crop`` cannot be reconstructed at runtime either: its vertical relation
to ``bbox_person`` is nearly fixed (bottom offset -[value removed] +/- [value removed]) but the
horizontal trim is not (width ratio [value removed] +/- [value removed]), so no rule recoverable
from the recorded fields reproduces it.

This script therefore computes pose the way **deployment** does, so training
matches inference rather than the other way round.

Grouping
--------
Records are grouped by ``src_frame`` so each 2812x1050 frame is decoded once and
all of its students are processed together. Reading per record instead would
decode the same frame ~3 times on average.

Output is the same ``(names, vecs)`` npz layout as ``head_pose_cache.npz``, so
it is a drop-in replacement everywhere that cache is consumed.

Usage (from LLMDet/):
    python -m attention.precompute_head_pose_frames \
        --labels ../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl \
        --frames-root ../grounding_data/stu_img/frames \
        --out ../grounding_data/llmstu_tools/outputs/head_pose_cache_bbox_person.npz \
        --workers 64
"""

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_LM = None       # one detector/landmarker per worker process
_BACKEND = "landmarker"


def _init(model_path, backend="landmarker"):
    """``landmarker`` gives yaw/pitch/roll + the flag; ``detector`` gives only
    the flag, at roughly 60% of the cost (74.9 -> 47.2 ms/frame measured over
    1,200 frames, attention/bench_face_backends.py). Since ~80% of the block's
    value is the flag [internal notes, not included], the detector variant exists to be tested
    downstream rather than argued about."""
    global _LM, _BACKEND
    import cv2  # noqa: F401  (ensure the per-worker import happens once)
    from mediapipe.tasks.python import vision, BaseOptions
    _BACKEND = backend
    if backend == "detector":
        _LM = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.5))
    else:
        _LM = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True))


def _pose(im) -> tuple:
    """(yaw, pitch, roll, face_found), angles normalised to [-1, 1].

    Identical maths to ``precompute_head_pose._one`` — only the input differs.
    A miss returns zeros **with face_found = 0**, which is what keeps "no face
    detected" distinguishable from "facing straight ahead".
    """
    import cv2
    import mediapipe as mp
    if im is None or im.size == 0 or min(im.shape[:2]) < 16:
        return 0.0, 0.0, 0.0, 0.0
    res = _LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if _BACKEND == "detector":
        # No pose available. Angles are ZERO and the flag says whether a face
        # was seen, so a consumer selecting only column 3 (`553_facefound`) gets
        # exactly the same semantics as with the landmarker.
        return 0.0, 0.0, 0.0, (1.0 if res.detections else 0.0)
    if not res.facial_transformation_matrixes:
        return 0.0, 0.0, 0.0, 0.0
    M = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
    sy = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    pitch = np.degrees(np.arctan2(-M[2, 0], sy)) / 90.0
    yaw = np.degrees(np.arctan2(M[1, 0], M[0, 0])) / 90.0
    roll = np.degrees(np.arctan2(M[2, 1], M[2, 2])) / 90.0
    c = lambda v: float(np.clip(v, -1.0, 1.0))
    return c(yaw), c(pitch), c(roll), 1.0


def _one_frame(args):
    """One source frame and every student in it -> [(file_name, y, p, r, found)]."""
    import cv2
    frame_path, people = args
    im = cv2.imread(frame_path)
    out = []
    if im is None:
        return [(fn, 0.0, 0.0, 0.0, 0.0) for fn, _ in people]
    h, w = im.shape[:2]
    for fn, bbox in people:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            out.append((fn, 0.0, 0.0, 0.0, 0.0))
            continue
        out.append((fn, *_pose(im[y1:y2, x1:x2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64,
                    help="shared machine — do not take every core")
    ap.add_argument("--model", default="../huggingface/mediapipe/face_landmarker.task")
    ap.add_argument("--backend", default="landmarker", choices=["landmarker", "detector"])
    args = ap.parse_args()

    by_frame = defaultdict(list)
    n = 0
    for line in open(args.labels):
        r = json.loads(line)
        by_frame[r["src_frame"]].append((r["file_name"], r["bbox_person"]))
        n += 1
    jobs = [(str(Path(args.frames_root) / f), people) for f, people in sorted(by_frame.items())]
    print(f"{n} records over {len(jobs)} distinct frames "
          f"({n / max(len(jobs), 1):.2f} students/frame), {args.workers} workers")

    t0 = time.time()
    names, vecs = [], []
    with Pool(args.workers, initializer=_init,
              initargs=(args.model, args.backend)) as pool:
        for i, rows in enumerate(pool.imap_unordered(_one_frame, jobs, chunksize=16)):
            for fn, y, p, r, f in rows:
                names.append(fn)
                vecs.append([y, p, r, f])
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)} frames  {time.time() - t0:.0f}s", flush=True)

    vecs = np.asarray(vecs, dtype=np.float32)
    order = np.argsort(names)          # deterministic order regardless of scheduling
    names = np.asarray(names)[order]
    vecs = vecs[order]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=names, vecs=vecs)
    found = float(vecs[:, 3].mean())
    print(f"wrote {args.out}: {len(names)} crops, face_found {found:.1%}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
