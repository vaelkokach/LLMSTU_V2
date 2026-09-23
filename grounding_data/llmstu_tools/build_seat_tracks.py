#!/usr/bin/env python3
"""Attach (video_id, seat_id) to every LLMSTU label record.

Students in these lab videos are stationary, so within one video the crops of
one student cluster tightly around their seat position. `person_idx` is a
det-confidence rank and useless as identity; instead we DBSCAN the
`bbox_person` head-center points per video. eps = 0.5 x the video's median
head span, min_samples=5. Noise points (label -1) get their nearest core seat
if within 2*eps, else seat_id = -1 (dropped later by dedup).

Inputs : LLMSTU/labels_slim.jsonl, outputs/frame_to_video.json
Output : outputs/labels_tracked.jsonl (adds video_id, seat_id)
"""
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = "/home/jovyan/Computer_vision/grounding_data/LLMSTU"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def head_center(rec):
    x0, y0, x1, y1 = rec["bbox_person"]
    # head sits at the top-center of the person box
    return (x0 + x1) / 2.0, y0 + rec["head_span_px"] / 2.0


def main():
    f2v = json.load(open(os.path.join(OUT_DIR, "frame_to_video.json")))
    by_video = defaultdict(list)
    n_missing = 0
    with open(os.path.join(ROOT, "labels_slim.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            vid = f2v.get(rec["src_frame"])
            if vid is None:
                n_missing += 1
                continue
            rec["video_id"] = vid
            by_video[vid].append(rec)
    print(f"videos: {len(by_video)}, records without video mapping: {n_missing}")

    out_path = os.path.join(OUT_DIR, "labels_tracked.jsonl")
    seat_counts = []
    with open(out_path, "w") as out:
        for vid in sorted(by_video):
            recs = by_video[vid]
            pts = np.array([head_center(r) for r in recs])
            eps = 0.5 * float(np.median([r["head_span_px"] for r in recs]))
            db = DBSCAN(eps=eps, min_samples=5).fit(pts)
            labels = db.labels_
            # attach noise points to nearest core cluster if close enough
            core = labels >= 0
            if core.any():
                centroids = {}
                for s in set(labels[core]):
                    centroids[s] = pts[labels == s].mean(0)
                cents = np.array(list(centroids.values()))
                keys = list(centroids)
                for i in np.where(~core)[0]:
                    d = np.linalg.norm(cents - pts[i], axis=1)
                    j = int(d.argmin())
                    if d[j] < 2 * eps:
                        labels[i] = keys[j]
            seat_counts.append(len({s for s in labels if s >= 0}))
            for r, s in zip(recs, labels):
                r["seat_id"] = int(s)
                out.write(json.dumps(r) + "\n")
    sc = np.array(seat_counts)
    print(f"wrote {out_path}")
    print(f"seats/video: min {sc.min()} med {int(np.median(sc))} max {sc.max()}; "
          f"total tracks {sc.sum()}")


if __name__ == "__main__":
    main()
