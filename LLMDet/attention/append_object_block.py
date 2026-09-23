#!/usr/bin/env python
"""Append the 6 object columns to an existing v1074_head build.

Why extend instead of rebuild
-----------------------------
The object features do not depend on any column already in the build, so a full
rebuild would spend ~40 GPU-minutes re-extracting CLIP over 284k crops to write
byte-identical numbers next to six new ones. Worse, it would re-run tracking and
chunking, and any drift in either silently makes the new runs incomparable to
the published ones -- the same argument ``build_cue_labels`` makes for labels.

It also avoids a live hazard: three head-pose caches exist
(``head_pose_cache``, ``_bbox_person``, ``_fullrange``), the build meta does not
record which one a given sequence tree used, and they differ. Picking the wrong
one would misalign 4 of 1080 columns with nothing to notice. Extending sidesteps
the question entirely: columns [0, 1074) are copied, not recomputed.

So this reads each sample, concatenates the object block at [1074, 1080), and
writes a new tree with ``layout="v1080_obj"``. Every other array -- labels,
candidates, timestamps -- is copied through unchanged.

Correctness is checked, not assumed:

  * the source must declare ``v1074_head`` and be 1074 wide;
  * every sample's frame count must match the number of crops resolved for it;
  * the output is re-read and its first rows compared against the source, so a
    write that silently reordered or truncated is caught here rather than in a
    training run.

    python -m attention.append_object_block \\
        --source ../grounding_data/llmstu_sequences_head \\
        --objects ../grounding_data/object_features.npz \\
        --manifest ../grounding_data/llmstu_seq_split_manifest.json \\
        --out ../grounding_data/llmstu_sequences_obj
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from attention.object_features import OBJECT_DIM
from attention.thesis_eval.data import LAYOUT_WIDTH


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--objects", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path,
                    default=Path("../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl"))
    ap.add_argument("--frame-to-video", type=Path,
                    default=Path("../grounding_data/llmstu_tools/outputs/frame_to_video.json"))
    args = ap.parse_args()

    d = np.load(args.objects, allow_pickle=False)
    omap = {str(n): i for i, n in enumerate(d["names"])}
    ovecs = d["vecs"].astype(np.float32)
    if ovecs.shape[1] != OBJECT_DIM:
        raise SystemExit(f"object cache is {ovecs.shape[1]}-wide, expected {OBJECT_DIM}")
    print(f"object cache: {len(omap)} crops x {OBJECT_DIM}")

    rows = json.load(open(args.manifest))["samples"]
    print(f"manifest: {len(rows)} samples")

    # Crop names per sample, recovered by replaying the builder's own grouping.
    # The samples do not store them, and guessing the order is exactly the
    # silent misalignment this tool exists to avoid. `replay_chunks` is the same
    # mechanism `build_cue_labels` uses, and it is verified the same way: the
    # replayed timestamps must equal the stored `t` EXACTLY, which is the
    # strongest available check that these are the same frames in the same
    # order.
    from attention.thesis_eval.patch_pose_columns import replay_chunks
    crop_names = {}
    n_checked = 0
    for rec in replay_chunks(args.labels, args.frame_to_video):
        key = f"{rec['split']}/sample_{rec['sample_idx']:06d}.npz"
        fp = args.source / key
        if not fp.exists():
            continue
        t = np.load(fp, allow_pickle=False)["t"].astype(np.float64)
        if len(t) != len(rec["times"]) or not np.array_equal(t, rec["times"]):
            raise SystemExit(
                f"{key}: replay/stored timestamp mismatch "
                f"({len(rec['times'])} replayed vs {len(t)} stored). Abort -- "
                f"the object rows would be appended against the wrong frames.")
        crop_names[key] = [m.get("file_name", "") for m in rec["metas"]]
        n_checked += len(t)
    print(f"replay: {len(crop_names)} samples, {n_checked} frames, "
          f"timestamps verified exactly")

    n_missing = n_frames = 0
    for split in ("train", "val"):
        (args.out / split).mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(rows, 1):
        src = args.source / r["file"]
        z = np.load(src, allow_pickle=False)
        x = z["x"].astype(np.float32)
        stored = str(z["layout"]) if "layout" in z.files else "v570"
        if stored != "v1074_head" or x.shape[1] != LAYOUT_WIDTH["v1074_head"]:
            raise SystemExit(
                f"{r['file']} is a {stored!r} build of width {x.shape[1]}; this "
                f"appends to v1074_head only. Columns are absolute -- writing "
                f"into the wrong layout would misalign every config.")

        names = crop_names.get(r["file"])
        if names is None:
            raise SystemExit(
                f"the replay produced no crops for {r['file']}. The builder's "
                f"grouping has drifted -- abort rather than append a block "
                f"against the wrong rows.")
        if len(names) != len(x):
            raise SystemExit(f"{r['file']}: {len(names)} crop names, {len(x)} frames")

        blk = np.zeros((len(x), OBJECT_DIM), dtype=np.float32)
        for k, nm in enumerate(names):
            j = omap.get(nm)
            if j is None:
                n_missing += 1
            else:
                blk[k] = ovecs[j]
        n_frames += len(x)

        out = args.out / r["file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: z[k] for k in z.files}
        payload["x"] = np.concatenate([x, blk], axis=1).astype(np.float32)
        payload["layout"] = np.array("v1080_obj")
        np.savez_compressed(out, **payload)
        if i % 1000 == 0:
            print(f"  {i}/{len(rows)} samples", flush=True)

    for extra in ("meta.json",):
        s = args.source / extra
        if s.exists():
            m = json.loads(s.read_text())
            m["layout"] = "v1080_obj"
            m["derived_from"] = str(args.source)
            m["object_cache"] = str(args.objects)
            (args.out / extra).write_text(json.dumps(m))

    print(f"\n{len(rows)} samples, {n_frames} frames")
    print(f"  crops with no object-cache entry: {n_missing} "
          f"({100 * n_missing / max(n_frames, 1):.2f}%) -- zeroed")

    # Re-read one sample and prove the first 1074 columns survived untouched.
    probe = rows[0]["file"]
    a = np.load(args.source / probe)["x"].astype(np.float32)
    b = np.load(args.out / probe)["x"].astype(np.float32)
    if b.shape[1] != LAYOUT_WIDTH["v1080_obj"] or not np.array_equal(a, b[:, :1074]):
        raise SystemExit("verification FAILED: the copied columns are not identical")
    print(f"  verified: {probe} columns [0,1074) byte-identical, width {b.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
