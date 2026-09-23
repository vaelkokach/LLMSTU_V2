"""Precompute head pose + face box + facial-expression probabilities per crop.

Addresses the `Thesis_Topic.md` requirement for **facial expressions (boredom,
perplexity, curiosity)** alongside head pose and gaze, which the pipeline
previously had no channel for at all.

Two stages, because the two models want different hardware:

  Stage 1 (CPU, N workers)  MediaPipe FaceLandmarker -> head pose (yaw, pitch,
                            roll), face_found, and the FACE BOUNDING BOX.
  Stage 2 (GPU, batched)    ViT FER over the face crop -> 7 expression
                            probabilities.

Why the face box matters: running FER on the whole student crop produces
nonsense (measured: "fear 0.83" on a seated lab student) because the model was
trained on aligned faces. Cropping to the MediaPipe face box first yields a
plausible spread instead of a collapsed one.

Output cache (per crop file_name), 11 dims:
    [yaw, pitch, roll, face_found, p_sad, p_disgust, p_angry, p_neutral,
     p_fear, p_surprise, p_happy]

HONEST SCOPE NOTE: the FER model predicts the 7 *basic* (Ekman) expressions.
The thesis names *academic* emotions (boredom, perplexity, curiosity), which are
a different construct — Pekrun's framework, not Ekman's. These 7 dims are used
as a learned FEATURE, not as a claim that we detect boredom. The boredom claim
is validated separately against DIPSER's expert academic-emotion labels
(code 1 = Boredom) in `validate_boredom.py`.

Usage:
    python -m attention.precompute_affect \
        --labels ../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl \
        --crops-root ../grounding_data/LLMSTU/crops \
        --out ../grounding_data/llmstu_tools/outputs/affect_cache.npz \
        --workers 96
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

AFFECT_DIM = 11
_LM = None


def _init(model_path):
    global _LM
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    _LM = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_facial_transformation_matrixes=True))


def _stage1(args):
    """-> (file_name, pose4, face_box4) ; face_box is (x1,y1,x2,y2) or zeros."""
    import cv2
    import mediapipe as mp
    fn, path = args
    zero = (np.zeros(4, np.float32), np.zeros(4, np.float32))
    im = cv2.imread(path)
    if im is None or im.size == 0 or min(im.shape[:2]) < 16:
        return (fn,) + zero
    h, w = im.shape[:2]
    res = _LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if not res.facial_transformation_matrixes or not res.face_landmarks:
        return (fn,) + zero
    M = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
    sy = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    pose = np.array([
        np.degrees(np.arctan2(M[1, 0], M[0, 0])) / 90.0,
        np.degrees(np.arctan2(-M[2, 0], sy)) / 90.0,
        np.degrees(np.arctan2(M[2, 1], M[2, 2])) / 90.0,
    ], dtype=np.float32)
    pose = np.concatenate([np.clip(pose, -1, 1), [1.0]]).astype(np.float32)
    xs = [p.x for p in res.face_landmarks[0]]
    ys = [p.y for p in res.face_landmarks[0]]
    box = np.array([max(0, min(xs) * w), max(0, min(ys) * h),
                    min(w, max(xs) * w), min(h, max(ys) * h)], dtype=np.float32)
    return fn, pose, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--crops-root", required=True)
    ap.add_argument("--mp-model", default="../huggingface/mediapipe/face_landmarker.task")
    ap.add_argument("--fer-model", default="../huggingface/fer_vit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    seen, jobs = set(), []
    for line in open(args.labels):
        fn = json.loads(line)["file_name"]
        if fn not in seen:
            seen.add(fn)
            jobs.append((fn, str(Path(args.crops_root) / fn)))
    print(f"{len(jobs)} unique crops")

    # ---- Stage 1: MediaPipe (CPU, parallel) --------------------------------
    names, poses, boxes = [], [], []
    with Pool(args.workers, initializer=_init, initargs=(args.mp_model,)) as pool:
        for i, (fn, pose, box) in enumerate(
                pool.imap_unordered(_stage1, jobs, chunksize=64), 1):
            names.append(fn); poses.append(pose); boxes.append(box)
            if i % 40000 == 0:
                print(f"  stage1 {i}/{len(jobs)}", flush=True)
    poses = np.stack(poses); boxes = np.stack(boxes)
    found = poses[:, 3] > 0
    print(f"stage1 done: face_found {found.mean():.1%}")

    # ---- Stage 2: FER over face crops (GPU, batched) -----------------------
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    proc = AutoImageProcessor.from_pretrained(args.fer_model)
    mdl = AutoModelForImageClassification.from_pretrained(
        args.fer_model).to(args.device).eval()
    n_emo = mdl.config.num_labels
    print(f"FER labels: {mdl.config.id2label}")
    emo = np.zeros((len(names), n_emo), dtype=np.float32)

    idx = np.nonzero(found)[0]
    root = Path(args.crops_root)
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        with torch.inference_mode():
            p = mdl(**proc(images=buf, return_tensors="pt").to(args.device)
                    ).logits.softmax(-1).cpu().numpy()
        emo[buf_idx] = p
        buf.clear(); buf_idx.clear()

    for c, i in enumerate(idx, 1):
        im = cv2.imread(str(root / names[i]))
        if im is None:
            continue
        h, w = im.shape[:2]
        x1, y1, x2, y2 = boxes[i]
        pad = 0.2 * max(1.0, x2 - x1)
        x1 = int(max(0, x1 - pad)); y1 = int(max(0, y1 - pad))
        x2 = int(min(w, x2 + pad)); y2 = int(min(h, y2 + pad))
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        buf.append(Image.fromarray(cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
        buf_idx.append(i)
        if len(buf) >= args.batch:
            flush()
        if c % 40000 == 0:
            print(f"  stage2 {c}/{len(idx)}", flush=True)
    flush()

    vecs = np.concatenate([poses, emo], axis=1).astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.asarray(names), vecs=vecs,
                        emotion_labels=np.asarray(
                            [mdl.config.id2label[i] for i in range(n_emo)]))
    scored = (emo.sum(1) > 0)
    print(f"\nwrote {args.out}")
    print(f"  {len(names)} crops, {AFFECT_DIM} dims "
          f"(4 pose + {n_emo} expression)")
    print(f"  face_found {found.mean():.1%}, expression scored {scored.mean():.1%}")
    for j in range(n_emo):
        m = emo[scored, j].mean() if scored.any() else 0.0
        print(f"    mean p({mdl.config.id2label[j]:8s}) = {m:.3f}")


if __name__ == "__main__":
    main()
