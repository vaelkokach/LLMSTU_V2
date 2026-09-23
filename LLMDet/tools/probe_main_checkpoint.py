#!/usr/bin/env python3
"""Sanity-probe a main-run checkpoint: does it actually detect and ground?

Two questions, answered separately because the deployed system only uses the
first one:

  1. LOCALIZATION  -- prompt "student": does it find the students?
                      Scored against the ODVG val boxes (IoU >= 0.5).
  2. GROUNDING     -- prompt = a cue phrase: does the top box for that phrase
                      land on a student the ground truth labels with it?
                      This is what R@1 measures.

Runs on CPU by default so it does not contend with a training job for GPUs
(the project cap is 4 GPUs and a training run holds all of them). CPU is
~5-15 s/image for Swin-T; --device cuda:N is ~20x faster if a GPU is free.

Usage:
  python tools/probe_main_checkpoint.py --ckpt work_dirs/student_llmstu_exact/iter_5000.pth \
      --n 8 --out work_dirs/probe/
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

CFG = "configs/student_llmstu_exact.py"
VAL = "../grounding_data/llmstu_tools/outputs/odvg_val.jsonl"
IMG_ROOT = "../grounding_data/stu_img/frames"
PHRASES = ["using laptop", "listening attentively", "sleeping head down",
           "looking away", "using phone", "talking to peer", "reading",
           "writing notes"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--out", default="work_dirs/probe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from mmdet.apis import init_detector, inference_detector

    rows = [json.loads(l) for l in open(VAL)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]

    print(f"loading {args.ckpt} on {args.device} ...")
    model = init_detector(CFG, args.ckpt, device=args.device)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    loc_tp = loc_fn = loc_fp = 0
    ground_hit = ground_tot = 0

    for r in rows:
        img_path = Path(IMG_ROOT) / r["filename"]
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"  MISSING IMAGE {img_path}")
            continue
        gt_boxes = [rg["bbox"] for rg in r["grounding"]["regions"]]
        gt_by_phrase = {}
        for rg in r["grounding"]["regions"]:
            gt_by_phrase.setdefault(rg["phrase"], []).append(rg["bbox"])

        # ---- 1. localization with the runtime prompt --------------------
        out = inference_detector(model, im, text_prompt="student",
                                 custom_entities=True)
        p = out.pred_instances
        keep = p.scores.cpu().numpy() >= args.score_thr
        pred = p.bboxes.cpu().numpy()[keep]
        matched = set()
        for pb in pred:
            best, bi = 0.0, -1
            for i, gb in enumerate(gt_boxes):
                v = iou(pb, gb)
                if v > best:
                    best, bi = v, i
            if best >= 0.5 and bi not in matched:
                matched.add(bi)
            else:
                loc_fp += 1
        loc_tp += len(matched)
        loc_fn += len(gt_boxes) - len(matched)

        # ---- 2. grounding: one prompt per phrase present in this frame ---
        for ph, boxes in gt_by_phrase.items():
            o = inference_detector(model, im, text_prompt=ph, custom_entities=True)
            pi = o.pred_instances
            if len(pi) == 0:
                ground_tot += 1
                continue
            top = pi.bboxes[pi.scores.argmax()].cpu().numpy()
            ground_tot += 1
            if any(iou(top, b) >= 0.5 for b in boxes):
                ground_hit += 1

        # ---- visual ------------------------------------------------------
        vis = im.copy()
        for gb in gt_boxes:
            cv2.rectangle(vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])),
                          (0, 200, 0), 2)
        for pb in pred:
            cv2.rectangle(vis, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])),
                          (0, 128, 255), 2)
        cv2.putText(vis, "green=GT  orange=pred(student)", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.imwrite(str(outdir / f"probe_{Path(r['filename']).stem[:40]}.jpg"), vis)
        print(f"  {r['filename'][:52]:52s} gt={len(gt_boxes):2d} pred={len(pred):2d} "
              f"matched={len(matched):2d}")

    prec = loc_tp / max(loc_tp + loc_fp, 1)
    rec = loc_tp / max(loc_tp + loc_fn, 1)
    print("\n--- LOCALIZATION (prompt 'student', IoU>=0.5) ---")
    print(f"  TP {loc_tp}  FP {loc_fp}  FN {loc_fn}")
    print(f"  precision {prec:.3f}   recall {rec:.3f}   "
          f"F1 {2*prec*rec/max(prec+rec,1e-6):.3f}")
    print("--- GROUNDING (top box for each cue phrase) ---")
    print(f"  {ground_hit}/{ground_tot} correct = {ground_hit/max(ground_tot,1):.3f}")
    print(f"\nannotated images -> {outdir}")


if __name__ == "__main__":
    main()
