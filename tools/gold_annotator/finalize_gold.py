#!/usr/bin/env python3
"""Merge verified annotations into the final gold set and split calibration/held-out.

Usage:
    python finalize_gold.py --primary gold_annotations_A.jsonl \
        [--secondary gold_annotations_B.jsonl] \
        --manifest gold_candidates.jsonl \
        [--calib-frac 0.3] [--out-dir .] [--seed 42]

Rules:
- rejected items are dropped;
- if a secondary annotator is given, items where the two disagree on any field are
  marked `disputed: true` (they stay in the set — resolve or exclude them downstream,
  but the decision must be fixed before evaluation, per to-do item 10);
- split is video-wise when the manifest carries `video_id` (no student appears in
  both halves), otherwise by md5(file_name);
- calibration gets ~calib-frac of items, held-out the rest. Never tune on held-out.
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import ALL_LABEL_FIELDS  # noqa: E402


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--secondary")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = {r["file_name"]: r for r in load_jsonl(args.manifest)}
    primary = {r["file_name"]: r for r in load_jsonl(args.primary)}
    secondary = {r["file_name"]: r for r in load_jsonl(args.secondary)} if args.secondary else {}

    gold = []
    for name, ann in primary.items():
        if ann["status"] == "rejected":
            continue
        rec = {"file_name": name, "status": ann["status"]}
        for k in ALL_LABEL_FIELDS:
            rec[k] = ann[k]
        m = manifest.get(name, {})
        for k in ("src_frame", "bbox_person", "video_id", "seat_id", "caption", "model_confidence"):
            if k in m:
                rec[k] = m[k]
        sec = secondary.get(name)
        if sec is not None:
            if sec["status"] == "rejected":
                continue
            rec["disputed"] = any(sec[k] != ann[k] for k in ALL_LABEL_FIELDS)
        gold.append(rec)

    if not gold:
        raise SystemExit("no usable annotations")

    # group split units: video-wise when available, else per-item hash
    have_video = all("video_id" in r for r in gold)
    units = {}
    for r in gold:
        key = r["video_id"] if have_video else hashlib.md5(r["file_name"].encode()).hexdigest()
        units.setdefault(key, []).append(r)

    keys = sorted(units)
    random.Random(args.seed).shuffle(keys)
    target = args.calib_frac * len(gold)
    calib, heldout, acc = [], [], 0
    for k in keys:
        if acc < target:
            calib.extend(units[k])
            acc += len(units[k])
        else:
            heldout.extend(units[k])

    for fname, rows in (("gold_calibration.jsonl", calib), ("gold_heldout.jsonl", heldout)):
        p = os.path.join(args.out_dir, fname)
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{fname}: {len(rows)} items -> {p}")
    n_disp = sum(1 for r in gold if r.get("disputed"))
    n_unc = sum(1 for r in gold if r["status"] == "uncertain")
    print(f"split unit: {'video_id' if have_video else 'md5(file_name)'}   "
          f"uncertain: {n_unc}   disputed: {n_disp}")


if __name__ == "__main__":
    main()
