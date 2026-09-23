#!/usr/bin/env python3
"""Video-wise train/val/test split (70/15/15 by default).

Never splits by frame, shard, or crop: all crops of one source video land in
exactly one split, so near-duplicate 1-fps frames cannot leak across splits
(the failure that invalidated the old stu_img split).

Greedy balancing: videos are sorted by crop count (descending) and each is
assigned to the split whose current activity-class distribution deviates most
from its target share — i.e. largest remaining deficit weighted by the video's
class profile.

Input : outputs/labels_dedup.jsonl
Output: outputs/splits.json  {train:[video_id...], val:[...], test:[...]}
        outputs/labels_{train,val,test}.jsonl
"""
import argparse
import json
import os
from collections import Counter, defaultdict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    args = ap.parse_args()
    ratios = dict(zip(("train", "val", "test"),
                      (float(x) for x in args.ratios.split(","))))

    by_video = defaultdict(list)
    with open(os.path.join(OUT_DIR, "labels_dedup.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            by_video[rec["video_id"]].append(rec)

    total = sum(len(v) for v in by_video.values())
    profiles = {v: Counter(r["activity"] for r in recs)
                for v, recs in by_video.items()}

    assign = {}
    split_counts = {s: Counter() for s in ratios}
    split_sizes = {s: 0 for s in ratios}
    for vid in sorted(by_video, key=lambda v: -len(by_video[v])):
        best, best_score = None, None
        for s, r in ratios.items():
            # deficit of this split vs its target size, plus rare-class need
            size_deficit = r - split_sizes[s] / max(total, 1)
            rare_bonus = sum(
                max(0, r * cnt_total(profiles, c) - split_counts[s][c])
                for c in profiles[vid]) / max(len(by_video[vid]), 1)
            score = size_deficit * 1000 + rare_bonus * 0.001
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[vid] = best
        split_sizes[best] += len(by_video[vid])
        split_counts[best].update(profiles[vid])

    splits = {s: sorted(v for v, sp in assign.items() if sp == s)
              for s in ratios}
    with open(os.path.join(OUT_DIR, "splits.json"), "w") as fh:
        json.dump(splits, fh, indent=2)
    for s in ratios:
        path = os.path.join(OUT_DIR, f"labels_{s}.jsonl")
        with open(path, "w") as out:
            for vid in splits[s]:
                for r in by_video[vid]:
                    out.write(json.dumps(r) + "\n")
        print(f"{s}: {len(splits[s])} videos, {split_sizes[s]} crops "
              f"({split_sizes[s]/total:.1%}) -> {path}")
        print(f"   classes: {dict(split_counts[s].most_common())}")


def cnt_total(profiles, c):
    return sum(p[c] for p in profiles.values())


if __name__ == "__main__":
    main()
