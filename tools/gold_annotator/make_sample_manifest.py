#!/usr/bin/env python3
"""Build a small test manifest for the annotator directly from LLMSTU labels.

Used until the real stratified `gold_candidates.jsonl` from llmstu_tools exists.
Keeps one crop per source frame to avoid near-duplicates in the sample.

Usage: python make_sample_manifest.py [--n 50] [--out gold_candidates_sample.jsonl]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
LABELS = os.path.join(REPO, "grounding_data", "LLMSTU", "labels", "shard_000.jsonl")
CROPS_ROOT = os.path.join(REPO, "grounding_data", "LLMSTU", "crops")

KEEP_KEYS = [
    "file_name", "src_frame", "bbox_person", "det_conf", "head_kpts", "face_kpts",
    "activity", "gaze_direction", "attention_target", "engagement_level",
    "posture", "hand_state", "phone_visible", "laptop_visible", "talking",
    "occluded", "caption", "model_confidence",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(ROOT, "gold_candidates_sample.jsonl"))
    ap.add_argument("--labels", default=LABELS)
    args = ap.parse_args()

    seen_frames = set()
    picked = []
    with open(args.labels) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["src_frame"] in seen_frames:
                continue
            seen_frames.add(rec["src_frame"])
            out = {k: rec[k] for k in KEEP_KEYS if k in rec}
            out["abs_path"] = os.path.join(CROPS_ROOT, rec["file_name"])
            picked.append(out)
            if len(picked) >= args.n:
                break

    with open(args.out, "w") as fh:
        for rec in picked:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(picked)} records -> {args.out}")


if __name__ == "__main__":
    main()
