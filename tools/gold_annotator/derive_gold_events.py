#!/usr/bin/env python3
"""Derive event-level gold from dense frame-level annotations (option 2).

Usage:
    python derive_gold_events.py \
        --annotations gold_annotations_<name>.jsonl \
        --manifest dense_event_manifest.jsonl \
        --out gold_events.jsonl

The manifest must be the dense per-segment manifest (has video_id, seat_id, t).
Human field values are mapped through the 6-class cue taxonomy, grouped into
per-(video, seat) timelines, and segmented into episodes with the SAME
attention.events code used for predictions, so gold and predicted events are
directly comparable. Frames the annotator rejected ('x') become gaps; frames
never annotated fall back to their pseudo-label cue (reported separately).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLMDet"))
from attention.taxonomy import map_record  # noqa: E402
from attention.events import segment_events  # noqa: E402

FIELDS = ["activity", "gaze_direction", "attention_target", "engagement_level",
          "posture", "hand_state", "phone_visible", "laptop_visible",
          "talking", "occluded"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="gold_events.jsonl")
    args = ap.parse_args()

    ann = {}
    for line in open(args.annotations):
        r = json.loads(line)
        ann[r["file_name"]] = r  # last state wins (append-only log)

    timelines = defaultdict(list)  # (video_id, seat_id) -> [(t, cue, source)]
    n_human = n_pseudo = n_rejected = 0
    for line in open(args.manifest):
        m = json.loads(line)
        a = ann.get(m["file_name"])
        if a is not None and a.get("status") == "rejected":
            n_rejected += 1
            continue  # gap in the timeline
        base = dict(m)
        if a is not None and a.get("status") in ("ok", "uncertain"):
            base.update({f: a[f] for f in FIELDS if f in a})
            src = "human"
            n_human += 1
        else:
            src = "pseudo"
            n_pseudo += 1
        cue = map_record(base)
        timelines[(m["video_id"], m["seat_id"])].append((m["t"], cue, src))

    out = open(args.out, "w")
    n_events = 0
    for (vid, seat), rows in sorted(timelines.items()):
        rows.sort()
        ts = [r[0] for r in rows]
        cues = [r[1] for r in rows]
        episodes = segment_events(ts, cues)
        for ep in episodes:
            out.write(json.dumps({"video_id": vid, "seat_id": seat,
                                  **ep.__dict__}) + "\n")
            n_events += 1
    out.close()
    print(f"frames: human={n_human} pseudo-fallback={n_pseudo} rejected={n_rejected}")
    print(f"gold events written: {n_events} -> {args.out}")
    if n_pseudo:
        print("WARNING: pseudo-fallback frames present — annotate the full "
              "segments for a purely human-derived event gold.")


if __name__ == "__main__":
    main()
