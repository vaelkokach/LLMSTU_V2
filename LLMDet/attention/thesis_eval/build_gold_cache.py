"""Extract the 570-dim feature sequences for the dense human-gold segments once.

``eval_events_vs_human.py`` re-runs CLIP over all 984 crops for every model
evaluated (~1 h GPU each). With five feature rungs times three seeds plus two
architectures that is prohibitive, and — worse — it invites comparing models
whose features were extracted on different days by different code paths.

This builds the cache **once**, in exactly the layout
``sequence_builder.build_sequences_llmstu`` produces, so downstream evaluation
is a pure column slice of the same array for every model:

    [  0:552]  base      full frame + bbox_person via StudentFeatureExtractor
    [552:556]  headpose  from outputs/affect_cache.npz (cols 0:4)
    [556:563]  express   from outputs/affect_cache.npz (cols 4:11)
    [563:570]  dynamic   compute_dynamic(track boxes, pose) over the whole track

The critical detail, and the bug that once turned 6/24 events into 0/24
[internal notes, not included]: features must come from the **full frame with bbox_person**, not
from the pre-cropped image with the whole image as the box. The latter makes 16
of 552 dims constant and silently off-distribution.

Usage (from LLMDet/):
    python -m attention.thesis_eval.build_gold_cache \
        --manifest ../grounding_data/llmstu_tools/outputs/dense_event_manifest.jsonl \
        --annotations ../event_gold_bundle/gold_annotations_Admin.jsonl \
        --frames-root ../grounding_data/stu_img/frames \
        --affect-cache ../grounding_data/llmstu_tools/outputs/affect_cache.npz \
        --out ../grounding_data/llmstu_tools/outputs/gold_event_features.npz
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attention.taxonomy import map_record

FIELDS = ["activity", "gaze_direction", "attention_target", "engagement_level",
          "posture", "hand_state", "phone_visible", "laptop_visible",
          "talking", "occluded"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--affect-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pose-cache", default=None,
                    help="npz from precompute_head_pose_frames.py. When given, it "
                         "OVERRIDES the affect cache's pose columns (0:4) so the gold "
                         "features match a model trained on bbox_person pose. The "
                         "dynamic block is derived from pose and is recomputed either "
                         "way, further down.")
    args = ap.parse_args()

    import cv2
    from attention.dynamic_features import compute_dynamic
    from attention.features import StudentFeatureExtractor

    ann = {}
    for line in open(args.annotations):
        r = json.loads(line)
        ann[r["file_name"]] = r          # append-only log, last state wins

    d = np.load(args.affect_cache, allow_pickle=False)
    amap = {str(n): i for i, n in enumerate(d["names"])}
    avecs = d["vecs"].astype(np.float32)
    if avecs.shape[1] != 11:
        raise SystemExit(f"affect cache has {avecs.shape[1]} dims, expected 11")
    pose_override = None
    if args.pose_cache:
        pc = np.load(args.pose_cache, allow_pickle=False)
        pose_override = ({str(n): i for i, n in enumerate(pc["names"])},
                         pc["vecs"].astype(np.float32))
        print(f"pose override: {len(pose_override[0])} crops from {args.pose_cache}")

    tracks = defaultdict(list)
    n_rej = n_hum = 0
    for line in open(args.manifest):
        m = json.loads(line)
        a = ann.get(m["file_name"])
        if a is not None and a.get("status") == "rejected":
            n_rej += 1
            continue                      # unverifiable -> a gap in the timeline
        human = -1
        if a is not None and a.get("status") in ("ok", "uncertain"):
            rec = dict(m)
            rec.update({f: a[f] for f in FIELDS if f in a})
            human = map_record(rec)
            n_hum += 1
        tracks[(m["video_id"], str(m["seat_id"]))].append(
            (float(m["t"]), m["src_frame"], [float(v) for v in m["bbox_person"]],
             map_record(m), human, m["file_name"]))
    for k in tracks:
        tracks[k].sort(key=lambda r: r[0])
    print(f"tracks {len(tracks)}  human-verified frames {n_hum}  rejected {n_rej}")

    feat = StudentFeatureExtractor(device=args.device)
    base_dim = feat.output_dim()
    if base_dim != 552:
        raise SystemExit(f"base extractor yields {base_dim} dims, expected 552")

    store = {}
    n_missing_affect = 0
    for key, rows in sorted(tracks.items()):
        boxes = [r[2] for r in rows]
        feats = []
        for _, src, bbox, _, _, fname in rows:
            im = cv2.imread(str(Path(args.frames_root) / src))
            if im is None:
                raise SystemExit(f"cannot read frame {src}")
            fv = feat.extract_batch(im, [bbox])[0]         # FULL frame + bbox
            j = amap.get(fname)
            if j is None:
                n_missing_affect += 1
                av = np.zeros(11, dtype=np.float32)
            else:
                av = avecs[j].copy()
            if pose_override is not None:
                pmap, pvecs = pose_override
                k = pmap.get(fname)
                # A miss must not silently leave crop-derived pose in place: that
                # is the very mismatch this override exists to remove.
                av[:4] = pvecs[k] if k is not None else 0.0
            feats.append(np.concatenate([fv, av]).astype(np.float32))
        arr = np.stack(feats)
        dyn = compute_dynamic(boxes, arr[:, base_dim:base_dim + 4])
        arr = np.concatenate([arr, dyn], axis=1).astype(np.float32)
        assert arr.shape[1] == 570, arr.shape
        name = f"{key[0]}|{key[1]}"
        store[f"x::{name}"] = arr
        store[f"t::{name}"] = np.array([r[0] for r in rows], dtype=np.float64)
        store[f"teacher::{name}"] = np.array([r[3] for r in rows], dtype=np.int64)
        store[f"human::{name}"] = np.array([r[4] for r in rows], dtype=np.int64)
        print(f"  {name}: {arr.shape[0]} frames", flush=True)

    store["track_keys"] = np.array(sorted(f"{k[0]}|{k[1]}" for k in tracks))
    store["n_human_frames"] = np.array(n_hum)
    store["n_rejected_frames"] = np.array(n_rej)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}  ({len(tracks)} tracks, "
          f"{n_missing_affect} crops missing from the affect cache)")


if __name__ == "__main__":
    main()
