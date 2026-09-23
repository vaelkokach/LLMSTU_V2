#!/usr/bin/env python
"""Per-crop object-presence features, cached once for the whole corpus.

Same shape as ``precompute_affect`` and ``precompute_head_pose``: run the
expensive model once over every frame, key the result by the crop's
``file_name``, and let ``sequence_builder`` look it up. The builder iterates
track by track and re-reads each frame many times, so detecting inside it would
run the detector once per student-observation instead of once per frame.

Which detector, and why not the deployed one
--------------------------------------------
The **pretrained** ``mm_grounding_dino`` swin-t, not the fine-tuned student
detector. The fine-tuned checkpoint ignores its text prompt entirely -- over 12
frames the nonsense string ``qwertyuiop`` retrieves students as well as ``a
student sitting`` does, 92% of its boxes matching a student box at IoU >= 0.9
[internal notes, not included]. It cannot be asked for a phone. The pretrained one can: the same
test returns **0** detections for the nonsense prompt and ``cell phone`` boxes an
order of magnitude smaller than person boxes.

Which features
--------------
The three that survived matched-pair validation [internal notes, not included] --
``score_contained`` (AUROC [value removed]), ``y_frac`` ([value removed]) and their product
([value removed]) -- per object, and only those. See ``attention/object_features.py``.

    python -m attention.precompute_objects \\
        --labels ../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl \\
        --image-root ../grounding_data/stu_img/frames \\
        --config configs/grounding_dino_swin_t_original.py \\
        --checkpoint ../huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \\
        --out ../grounding_data/object_features.npz --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from attention.object_features import (OBJECT_DIM, OBJECT_PROMPTS,
                                       ObjectDetector,
                                       student_object_features)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    # Sharding by FRAME, not by record: every crop of a frame must land in the
    # same shard or the frame gets detected more than once, which is the cost
    # this cache exists to avoid.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    import cv2

    # Group crops by their source frame: one detector pass serves every student
    # in it, which is the whole point of caching this.
    by_frame = defaultdict(list)
    n_rec = 0
    with args.labels.open() as fh:
        for line in fh:
            r = json.loads(line)
            fn, sf, bb = (r.get("file_name"), r.get("src_frame"),
                          r.get("bbox_person"))
            if not (fn and sf and bb):
                continue
            by_frame[sf].append((fn, bb))
            n_rec += 1
            if args.limit and n_rec >= args.limit:
                break
    if args.num_shards > 1:
        keys = sorted(by_frame)
        mine = {k for i, k in enumerate(keys) if i % args.num_shards == args.shard}
        by_frame = {k: v for k, v in by_frame.items() if k in mine}
        n_rec = sum(len(v) for v in by_frame.values())
        print(f"shard {args.shard + 1}/{args.num_shards}")
    print(f"{n_rec} crops over {len(by_frame)} frames; "
          f"{n_rec / max(len(by_frame), 1):.1f} students per frame")
    print(f"objects: {', '.join(OBJECT_PROMPTS)} -> {OBJECT_DIM} dims per crop")

    det = ObjectDetector(str(args.config), str(args.checkpoint), device=args.device)

    names, vecs = [], []
    missing = 0
    t0 = time.time()
    for i, (sf, crops) in enumerate(sorted(by_frame.items()), 1):
        img = cv2.imread(str(args.image_root / sf))
        if img is None:
            # A frame that is not on disk gets zeros for every crop in it, and
            # is counted. Skipping them instead would silently shorten the cache
            # and sequence_builder would then fall back to zeros anyway, without
            # anyone knowing how often.
            missing += len(crops)
            for fn, _ in crops:
                names.append(fn)
                vecs.append(np.zeros(OBJECT_DIM, dtype=np.float32))
            continue
        per_obj = det.detect(img)
        for fn, bb in crops:
            names.append(fn)
            vecs.append(student_object_features(per_obj, bb))
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(by_frame)} frames, {el:.0f}s, "
                  f"{i / max(el, 1e-9):.1f} fps, eta "
                  f"{(len(by_frame) - i) / max(i / max(el, 1e-9), 1e-9) / 60:.0f} min",
                  flush=True)

    V = np.stack(vecs).astype(np.float32)
    nz = (V != 0).any(axis=1).mean()
    print(f"\n{len(names)} crops cached, {OBJECT_DIM} dims")
    print(f"  frames not on disk: {missing} crops zeroed")
    print(f"  crops with any object detected: {100 * nz:.1f}%")
    for oi, name in enumerate(OBJECT_PROMPTS):
        col = V[:, oi * 3]
        print(f"  {name:12} present on {100 * (col > 0).mean():5.1f}% of crops, "
              f"mean score {col[col > 0].mean() if (col > 0).any() else 0:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), vecs=V,
                        prompts=np.array(OBJECT_PROMPTS),
                        provenance=json.dumps({
                            "checkpoint": str(args.checkpoint),
                            "config": str(args.config),
                            "labels": str(args.labels),
                            "n_crops": len(names), "dim": OBJECT_DIM}))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
