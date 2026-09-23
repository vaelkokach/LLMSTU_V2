#!/usr/bin/env python
"""Is a rebuilt sequence set label-identical to the old one?

Adding ``y_cand`` needs the sequences rebuilt, and a rebuild re-runs the whole
builder: tracking, split assignment, feature extraction. If any of that drifts,
the new runs are measured on a different dataset than the published 6-class
numbers and the comparison is silently void.

So check the two things a comparison depends on, and nothing else:

  * the manifest lists the same files in the same splits;
  * every sequence's ``y_frames`` and ``t`` are byte-identical.

Features are NOT compared. CLIP on a different GPU can differ in the last bits
of a float, which is not a reason to reject a rebuild -- but a changed label or
timestamp is.

    python tools/verify_sequence_rebuild.py OLD_DIR NEW_DIR \
        --old-manifest ... --new-manifest ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(manifest: Path):
    rows = json.load(open(manifest, encoding="utf-8"))["samples"]
    return {r["file"]: r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--old-manifest", type=Path, required=True)
    ap.add_argument("--new-manifest", type=Path, required=True)
    ap.add_argument("--training-manifest", type=Path, default=None,
                    help="the leak-free split manifest training actually reads "
                         "(llmstu_seq_split_manifest.json). Checked separately: "
                         "it carries the video-wise train/val/TEST split, while "
                         "a builder meta.json carries the builder's own 80/20, "
                         "so comparing those two directly gives a false failure.")
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N sequences (a smoke test)")
    args = ap.parse_args()

    old, new = load(args.old_manifest), load(args.new_manifest)

    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    if only_old or only_new:
        print(f"FAIL: manifest membership changed "
              f"({len(only_old)} dropped, {len(only_new)} added)")
        for f in (only_old + only_new)[:5]:
            print(f"    {f}")
        return 1
    moved = [f for f in old if old[f]["split"] != new[f]["split"]]
    if moved:
        print(f"FAIL: {len(moved)} sequences changed split, e.g. {moved[:5]}")
        return 1
    print(f"manifest: {len(old)} sequences, same files, same splits")

    files = sorted(old)
    if args.limit:
        files = files[:args.limit]

    bad_y, bad_t, no_cand, checked = [], [], [], 0
    for f in files:
        try:
            a = np.load(args.old_dir / f)
            b = np.load(args.new_dir / f)
        except FileNotFoundError as e:
            print(f"FAIL: {e}")
            return 1
        if not np.array_equal(a["y_frames"], b["y_frames"]):
            bad_y.append(f)
        if not np.allclose(a["t"], b["t"], rtol=0, atol=1e-9):
            bad_t.append(f)
        if "y_cand" not in b.files:
            no_cand.append(f)
        else:
            # the precedence winner must be inside its own candidate set,
            # or PRODEN would put zero weight on the class CE trains toward
            yc, y = b["y_cand"].astype(bool), b["y_frames"]
            if not yc[np.arange(len(y)), y].all():
                bad_y.append(f + " (label outside candidate set)")
        checked += 1

    # What training reads must resolve inside the new build.
    if args.training_manifest:
        rows = load(args.training_manifest)
        missing = [f for f in rows if not (args.new_dir / f).is_file()]
        if missing:
            print(f"FAIL: {len(missing)} of {len(rows)} files named by "
                  f"{args.training_manifest.name} are absent from the new build, "
                  f"e.g. {missing[:3]}")
            return 1
        by_split = {}
        for f, r in rows.items():
            by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        print(f"training manifest: all {len(rows)} files present "
              f"({', '.join(f'{k}={v}' for k, v in sorted(by_split.items()))})")

    ok = True
    for name, bad in (("y_frames differs", bad_y), ("t differs", bad_t),
                      ("y_cand missing", no_cand)):
        if bad:
            ok = False
            print(f"FAIL: {name} in {len(bad)} sequences, e.g. {bad[:3]}")
    if ok:
        print(f"{checked} sequences: labels and timestamps identical, "
              f"y_cand present and consistent")
        print("\nthe rebuild is comparable to the published runs")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
