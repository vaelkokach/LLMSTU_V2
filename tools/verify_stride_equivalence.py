"""Replay one video at several detector/temporal strides and compare the cues.

Striding is a speed optimisation that trades freshness for compute, and the
argument for it — students are stationary, episodes are sustained — is an
argument, not a measurement. This measures it.

For each configuration the same video is replayed and every (frame, track) cue
is recorded. Stride 1 is the reference. Reported per configuration:

* **cue agreement** — fraction of (frame, track) pairs whose displayed cue
  matches the reference. This is the number that matters: it is what the
  instructor sees.
* **track coverage** — fraction of reference (frame, track) pairs that exist at
  all under the stride. A stride that quietly loses tracks would otherwise score
  high agreement on the few it kept.
* **episode agreement** — cue runs of >= 3 s (the event layer's
  ``min_duration_s``) that survive, since a difference that never forms an
  episode never reaches an alert.

Track ids are NOT comparable across runs — the tracker assigns them in
detection order — so tracks are matched between runs by box overlap at each
frame, which is what a human comparing the two overlays would do.

    python tools/verify_stride_equivalence.py \
        --config LLMDet/configs/attention_runtime.yaml --video LLMDet/0325.mp4 \
        --frames 300 --strides 1:1,5:3,10:5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def replay(config, video, frames, det_stride, tmp_stride):
    """-> {frame_idx: [(bbox, displayed_cue, raw_cue, confidence), ...]}"""
    import yaml
    from pipeline_bridge import run_live

    cfg_path = Path(config)
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("inference", {})["detector_stride"] = det_stride
    cfg["inference"]["temporal_stride"] = tmp_stride
    tmp = cfg_path.parent / f".stride_{det_stride}_{tmp_stride}.yaml"
    tmp.write_text(yaml.safe_dump(cfg))

    out = {}
    state = {"n": 0}

    def push(t, jpg, students, cues):
        # keyed by BOX, not track id: ids are assigned in detection order and
        # differ between runs.
        out[state["n"]] = [(v["bbox"], v["cue"], v.get("raw_cue"), v["conf"])
                           for v in students.values() if not v.get("warming")]
        state["n"] += 1

    rec = REPO / "tools" / "dashboard" / f".stride_rec_{det_stride}_{tmp_stride}.jsonl"
    try:
        run_live(str(tmp), str(video), push, max_frames=frames, record=str(rec))
    finally:
        tmp.unlink(missing_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--strides", default="1:1,5:3,10:5",
                    help="comma-separated detector:temporal pairs; the first is the reference")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="source fps, used to convert the 3 s episode floor to frames")
    ap.add_argument("--out", default="LLMDet/work_dirs/profiling/stride_equivalence.json")
    args = ap.parse_args()

    pairs = [tuple(int(x) for x in p.split(":")) for p in args.strides.split(",")]
    runs = {}
    for d, t in pairs:
        print(f"\n=== replay detector_stride={d} temporal_stride={t} ===", flush=True)
        runs[(d, t)] = replay(args.config, args.video, args.frames, d, t)

    ref_key = pairs[0]
    ref = runs[ref_key]
    min_run = max(1, int(round(3.0 * args.fps)))   # events layer's min_duration_s

    def episodes(run):
        """(seat, cue) runs of at least min_run frames.

        Students are stationary, so a box's rounded centre is a stable seat key
        within a run — stable enough to chain a cue across frames without
        depending on track ids.
        """
        seq = defaultdict(list)
        for fi in sorted(run):
            for bbox, cue, _raw, _c in run[fi]:
                seat = (round((bbox[0] + bbox[2]) / 40), round((bbox[1] + bbox[3]) / 40))
                seq[seat].append((fi, cue))
        eps = []
        for tid, rows in seq.items():
            start, cur = None, None
            for fi, cue in rows + [(None, None)]:
                if cue != cur:
                    if cur is not None and start is not None and fi is not None \
                            and fi - start >= min_run:
                        eps.append((tid, cur))
                    start, cur = fi, cue
        return sorted(eps)

    results = {}
    for key, run in runs.items():
        common = sorted(set(ref) & set(run))
        agree = total = covered = ref_total = 0
        for fi in common:
            here = list(run[fi])
            for rbox, cue, _r, _c in ref[fi]:
                ref_total += 1
                # match the reference student to the best-overlapping student
                # in this run; 0.5 IoU is unambiguous for seated, well-separated
                # students and never matched two seats in practice
                best, best_iou = None, 0.5
                for cand in here:
                    o = iou(rbox, cand[0])
                    if o > best_iou:
                        best, best_iou = cand, o
                if best is not None:
                    covered += 1
                    total += 1
                    agree += int(best[1] == cue)
        e_ref, e_run = set(episodes(ref)), set(episodes(run))
        results[f"{key[0]}:{key[1]}"] = {
            "detector_stride": key[0], "temporal_stride": key[1],
            "frames_compared": len(common),
            "track_coverage": covered / max(ref_total, 1),
            "cue_agreement": agree / max(total, 1),
            "n_episodes_reference": len(e_ref), "n_episodes_here": len(e_run),
            "episode_agreement": (len(e_ref & e_run) / max(len(e_ref), 1)),
            "is_reference": key == ref_key,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"video": args.video, "frames": args.frames,
         "reference": f"{ref_key[0]}:{ref_key[1]}", "runs": results}, indent=2))

    print(f"\n{'det:tmp':>8} {'frames':>7} {'coverage':>9} {'cue agree':>10} "
          f"{'episodes':>9} {'ep agree':>9}")
    for k, r in results.items():
        print(f"{k:>8} {r['frames_compared']:7d} {r['track_coverage']:9.1%} "
              f"{r['cue_agreement']:10.1%} {r['n_episodes_here']:4d}/"
              f"{r['n_episodes_reference']:<4d} {r['episode_agreement']:9.1%}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
