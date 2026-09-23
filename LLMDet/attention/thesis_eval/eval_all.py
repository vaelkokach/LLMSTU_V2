"""Evaluate every finished run under a sweep root with the unified evaluator.

Runs are evaluated **sequentially on a single GPU** — the models are small and
the evaluator is the authoritative measurement, so there is no reason to add
concurrency (and the project caps GPU occupancy at 4 regardless).

    python -m attention.thesis_eval.eval_all --root work_dirs/thesis/ladder --split val
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ckpt-name", default="best.pth")
    ap.add_argument("--refine-asrf", action="store_true",
                    help="use boundary-refined labels for ASRF runs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sequence-root",
                    default="../grounding_data/llmstu_sequences_full")
    args = ap.parse_args()

    root = Path(args.root)
    todo = sorted(d for d in root.iterdir()
                  if (d / "run_record.json").exists()
                  and (d / "checkpoints" / args.ckpt_name).exists())
    if not todo:
        raise SystemExit(f"no completed runs under {root}")

    for d in todo:
        out = d / f"eval_{args.split}"
        if (out / "metrics.json").exists() and not args.force:
            print(f"skip (done): {d.name}")
            continue
        cmd = [sys.executable, "-m", "attention.thesis_eval.run_eval",
               "--ckpt", str(d / "checkpoints" / args.ckpt_name),
               "--split", args.split, "--out", str(out),
               "--device", args.device, "--n-boot", str(args.n_boot),
               "--sequence-root", args.sequence_root]
        if args.refine_asrf and "asrf" in d.name:
            cmd.append("--refine")
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED: {d.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
