#!/usr/bin/env python3
"""Stratified 1,000-crop sample for the human gold set.

Strata: video_id x activity x occluded, sampled from the DEDUPLICATED pool so
near-identical frames don't waste annotation effort. Rare classes
(raising_hand, slumped-posture, eating_drinking, using_phone, talking_to_peer)
are oversampled: every available example is taken up to a per-class ceiling
before the remaining budget is spread proportionally.

Output: outputs/gold_candidates.jsonl — one record per crop with `file_name`,
absolute `image_path`, and all pseudo-label fields (feeds the annotation tool).
"""
import argparse
import json
import os
import random
from collections import defaultdict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CROPS_ROOT = "/home/jovyan/Computer_vision/grounding_data/LLMSTU/crops"
RARE_ACTIVITIES = {"raising_hand", "eating_drinking", "using_phone",
                   "talking_to_peer", "writing_notes"}
RARE_CEIL = 80          # max crops taken per rare activity
TARGET = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pool = []
    with open(os.path.join(OUT_DIR, "labels_dedup.jsonl")) as fh:
        for line in fh:
            pool.append(json.loads(line))

    chosen = []
    # 1) rare classes first (incl. slumped posture regardless of activity)
    rare_pool = [r for r in pool if r["activity"] in RARE_ACTIVITIES
                 or r["posture"] == "slumped"]
    by_act = defaultdict(list)
    for r in rare_pool:
        by_act[r["activity"]].append(r)
    for act, recs in sorted(by_act.items()):
        rng.shuffle(recs)
        chosen.extend(recs[:RARE_CEIL])

    # 2) fill remaining budget stratified by (video, activity, occluded)
    remaining = args.target - len(chosen)
    chosen_ids = {id(r) for r in chosen}
    strata = defaultdict(list)
    for r in pool:
        if id(r) not in chosen_ids:
            strata[(r["video_id"], r["activity"], r["occluded"])].append(r)
    keys = sorted(strata)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(strata[k])
    i = 0
    while remaining > 0 and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            chosen.append(strata[k].pop())
            remaining -= 1
        i += 1

    out_path = os.path.join(OUT_DIR, "gold_candidates.jsonl")
    with open(out_path, "w") as out:
        for r in sorted(chosen, key=lambda r: r["file_name"]):
            rec = dict(r)
            rec["image_path"] = os.path.join(CROPS_ROOT, r["file_name"])
            out.write(json.dumps(rec) + "\n")

    from collections import Counter
    print(f"wrote {len(chosen)} candidates -> {out_path}")
    print("activity:", dict(Counter(r['activity'] for r in chosen).most_common()))
    print("occluded:", dict(Counter(r['occluded'] for r in chosen)))
    print("videos covered:", len({r['video_id'] for r in chosen}))


if __name__ == "__main__":
    main()
