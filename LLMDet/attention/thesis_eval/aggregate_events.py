"""Aggregate per-seed event evaluations into thesis Table B.

Event metrics on this gold set are computed over **16 distinct episodes across
10 tracks**. That is a diagnostic sample, not a classroom-wide estimate, and a
single episode changing hands moves recall by 6 percentage points. Reporting a
single seed's event recall as a headline would be exactly the kind of
single-number framing this project's post-mortem exists to prevent, so this
tool reports mean ± sd over seeds and prints the sample size next to every
column.

    python -m attention.thesis_eval.aggregate_events \
        --inputs work_dirs/thesis/events/summary_s42.json ... --out work_dirs/thesis/tables
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from attention.thesis_eval import EVALUATOR_VERSION

ROWS = [
    ("frame_accuracy", lambda r, p: r["frame"]["accuracy"]),
    ("frame_macro_f1", lambda r, p: r["frame"]["macro_f1"]),
    ("f1@10", lambda r, p: r["segmentation"]["f1@10"]),
    ("f1@25", lambda r, p: r["segmentation"]["f1@25"]),
    ("f1@50", lambda r, p: r["segmentation"]["f1@50"]),
    ("edit_score", lambda r, p: r["segmentation"]["edit_score"]),
    ("event_precision", lambda r, p: r["events"]["by_tiou"][p]["precision"]),
    ("event_recall", lambda r, p: r["events"]["by_tiou"][p]["recall"]),
    ("event_f1", lambda r, p: r["events"]["by_tiou"][p]["f1"]),
    ("event_recall_tiou10", lambda r, p: r["events"]["by_tiou"]["0.10"]["recall"]),
    ("event_recall_tiou50", lambda r, p: r["events"]["by_tiou"]["0.50"]["recall"]),
    ("onset_mae_s", lambda r, p: r["events"]["by_tiou"][p]["onset_mae_s"]),
    ("offset_mae_s", lambda r, p: r["events"]["by_tiou"][p]["offset_mae_s"]),
    ("duration_mae_s", lambda r, p: r["events"]["by_tiou"][p]["duration_mae_s"]),
    ("detection_delay_s", lambda r, p: r["events"]["by_tiou"][p]["detection_delay_s"]),
    ("false_alerts_per_hour", lambda r, p: r["events"]["by_tiou"][p]["false_alerts_per_hour"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--tag", default="gold_dedup", choices=["gold_dedup", "gold_raw"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs = [json.loads(Path(p).read_text()) for p in args.inputs]
    prim = f"{docs[0]['primary_iou']:.2f}"
    n_gold = docs[0]["results"][args.tag]["n_gold_events"]
    n_tracks = docs[0]["n_tracks"]

    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for d in docs:
        for sysname, r in d["results"][args.tag]["per_system"].items():
            for mname, fn in ROWS:
                try:
                    acc[sysname][mname].append(float(fn(r, prim)))
                except (KeyError, TypeError):
                    acc[sysname][mname].append(float("nan"))

    def ms(v):
        a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        if a.size == 0:
            return float("nan"), float("nan")
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

    order = [k for k in acc if k not in ("teacher", "majority")] + \
            [k for k in ("teacher", "majority") if k in acc]
    summary = {"evaluator_version": EVALUATOR_VERSION, "tag": args.tag,
               "primary_iou": prim, "n_gold_events": n_gold, "n_tracks": n_tracks,
               "n_seeds": len(docs), "inputs": list(args.inputs), "systems": {}}
    for s in order:
        summary["systems"][s] = {m: {"mean": ms(acc[s][m])[0], "std": ms(acc[s][m])[1],
                                     "values": acc[s][m]} for m, _ in ROWS}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"table_b_{args.tag}.json").write_text(json.dumps(summary, indent=2))

    hdr = ["system"] + [m for m, _ in ROWS]
    with open(out / f"table_b_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for s in order:
            w.writerow([s] + [f"{summary['systems'][s][m]['mean']:.4f}" for m, _ in ROWS])

    md = [f"### Table B — human-gold temporal events ({args.tag})", "",
          f"**{n_gold} distinct gold episodes over {n_tracks} tracks; "
          f"mean ± sd over {len(docs)} seeds; episode matching at tIoU {prim}.**",
          "", "Detector and tracker are excluded by design — track identity comes from the "
          "manifest — so these numbers isolate the cue + event layers. This is a "
          "**diagnostic** human-gold set (two densely annotated segments), not a "
          "classroom-wide estimate: one episode is 6 percentage points of recall.", "",
          "| system | frame acc | frame macro-F1 | F1@10 | F1@25 | F1@50 | edit | "
          "event P | event R | event F1 | onset MAE (s) | offset MAE (s) | dur MAE (s) | FA/h |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in order:
        d = summary["systems"][s]
        g = lambda m, n=3: (f"{d[m]['mean']:.{n}f} ± {d[m]['std']:.{n}f}"
                            if np.isfinite(d[m]["mean"]) else "n/a")
        md.append(f"| {s} | {g('frame_accuracy')} | {g('frame_macro_f1')} | {g('f1@10')} | "
                  f"{g('f1@25')} | {g('f1@50')} | {g('edit_score', 1)} | "
                  f"{g('event_precision')} | {g('event_recall')} | {g('event_f1')} | "
                  f"{g('onset_mae_s', 2)} | {g('offset_mae_s', 2)} | "
                  f"{g('duration_mae_s', 2)} | {g('false_alerts_per_hour', 2)} |")
    md += ["", "Notes:",
           "- `teacher` is the Qwen3-VL-family pseudo-labeller measured against the same "
           "human gold. It is an **empirical teacher benchmark**, not a theoretical ceiling.",
           "- `majority` is the constant-dominant-class control.",
           "- Boundary errors (onset/offset/duration) are computed over each system's own "
           "matched events and are therefore **not** directly comparable across systems "
           "with different recall; see the common-matched-subset block in "
           "`summary_s*.json` for the like-for-like comparison."]
    (out / f"table_b_{args.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
