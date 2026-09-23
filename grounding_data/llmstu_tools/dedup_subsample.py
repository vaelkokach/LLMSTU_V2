#!/usr/bin/env python3
"""Deduplicate the 1-fps LLMSTU crops and apply the occlusion policy.

Frames were sampled once per second, so each (video, seat) contributes
200-500 near-identical crops. Strategy (per seat track, sorted by timestamp):
  1. run-length-encode on the label state tuple
     (activity, gaze_direction, engagement_level, posture, hand_state, occluded)
  2. keep the first crop of every run (state transitions always survive)
  3. within a run, keep one crop every --stride seconds (default 10)

Occlusion policy (visual audit finding): crops with occluded=True AND
face_kpts==2 are unverifiable pixels -> dropped entirely. Other occluded crops
are kept (flag survives in the record).

Class cap: `listening` + `using_laptop` dominate (72.5% raw). After dedup they
are randomly downsampled (seeded) so that together they make up at most
--majority-cap of the final set (default 0.50).

Input : outputs/labels_tracked.jsonl
Output: outputs/labels_dedup.jsonl + outputs/dedup_report.json
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
STATE_KEYS = ("activity", "gaze_direction", "engagement_level",
              "posture", "hand_state", "occluded")
T_RE = re.compile(r"^t(\d{6})_(\d{3})_")


def ts(rec):
    m = T_RE.match(rec["src_frame"])
    return int(m.group(1)) + int(m.group(2)) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=10.0, help="seconds")
    ap.add_argument("--majority-cap", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tracks = defaultdict(list)
    n_in = n_occ_drop = n_noise = 0
    with open(os.path.join(OUT_DIR, "labels_tracked.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            n_in += 1
            if rec["seat_id"] < 0:
                n_noise += 1
                continue
            if rec["occluded"] and rec["face_kpts"] == 2:
                n_occ_drop += 1
                continue
            tracks[(rec["video_id"], rec["seat_id"])].append(rec)

    kept = []
    for key in sorted(tracks):
        recs = sorted(tracks[key], key=ts)
        run_state, last_kept_t = None, None
        for r in recs:
            state = tuple(r[k] for k in STATE_KEYS)
            t = ts(r)
            if state != run_state or t - last_kept_t >= args.stride:
                kept.append(r)
                run_state, last_kept_t = state, t

    hist_before = Counter(r["activity"] for rs in tracks.values() for r in rs)
    hist_after_dedup = Counter(r["activity"] for r in kept)

    # ---- majority-class cap ----
    majors = ("listening", "using_laptop")
    minor_n = sum(1 for r in kept if r["activity"] not in majors)
    target_major = int(minor_n * args.majority_cap / (1 - args.majority_cap))
    major_recs = [r for r in kept if r["activity"] in majors]
    if len(major_recs) > target_major:
        rng.shuffle(major_recs)
        keep_major = set(id(r) for r in major_recs[:target_major])
        kept = [r for r in kept
                if r["activity"] not in majors or id(r) in keep_major]

    hist_final = Counter(r["activity"] for r in kept)
    out_path = os.path.join(OUT_DIR, "labels_dedup.jsonl")
    with open(out_path, "w") as out:
        for r in kept:
            out.write(json.dumps(r) + "\n")

    report = {
        "input_records": n_in,
        "dropped_seat_noise": n_noise,
        "dropped_occluded_face2": n_occ_drop,
        "after_dedup": sum(hist_after_dedup.values()),
        "final": len(kept),
        "retention_after_dedup": sum(hist_after_dedup.values()) / max(n_in, 1),
        "activity_hist_before": dict(hist_before.most_common()),
        "activity_hist_after_dedup": dict(hist_after_dedup.most_common()),
        "activity_hist_final": dict(hist_final.most_common()),
        "stride_s": args.stride, "majority_cap": args.majority_cap,
    }
    with open(os.path.join(OUT_DIR, "dedup_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items()
                      if not k.startswith("activity_hist")}, indent=2))
    print("final activity histogram:", dict(hist_final.most_common()))


if __name__ == "__main__":
    main()
