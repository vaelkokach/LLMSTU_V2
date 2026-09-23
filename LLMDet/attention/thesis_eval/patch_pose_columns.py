"""Replace columns 552:556 of the built sequences with a different pose cache.

Motivation: the head-pose block was cached from the stored 512x512 crops
(``bbox_crop``) while the deployed pipeline computes pose from the detector's
box (``bbox_person``). 11% of frames disagreed on ``face_found``
[internal notes, not included]. Rebuilding pose the way deployment does is the fix; this
script applies it to the existing sequences.

Why patch instead of rebuilding the sequences: the other 552 columns are CLIP,
geometry, colour and posture, none of which depend on head pose. A full rebuild
would re-run CLIP over 271,485 frames for no change to 552 of 570 columns.

The hard part is that an NPZ stores only ``x``, ``y_frames``, ``y`` and ``t`` —
not which label records produced it. The mapping is recovered by **replaying**
``sequence_builder.build_sequences_llmstu``'s grouping (parse -> assign_seats ->
chunk) with feature extraction removed, which is deterministic. The replay is
then **proved** correct rather than assumed: every recovered chunk's timestamps
must equal the stored ``t`` array exactly, for every sequence. Any mismatch
aborts.

Columns 563:570 (the dynamic block) are recomputed too, because
``compute_dynamic`` derives its gaze-deviation features from the pose columns —
leaving them stale would make the 563_dyn / 570_full rungs internally
inconsistent.

    python -m attention.thesis_eval.patch_pose_columns \
        --src ../grounding_data/llmstu_sequences_full \
        --dst ../grounding_data/llmstu_sequences_full_bp \
        --pose-cache ../grounding_data/llmstu_tools/outputs/head_pose_cache_bbox_person.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from attention.dynamic_features import DYNAMIC_DIM, compute_dynamic
from attention.sequence_builder import assign_seats, parse_llmstu_labels, split_videos

POSE_LO, POSE_HI = 552, 556
DYN_LO, DYN_HI = 563, 570


def replay_chunks(labels: Path, frame_to_video: Path, min_track_len: int = 8,
                  max_track_len: int = 128, max_gap_s: float = 15.0,
                  val_fraction: float = 0.2, seed: int = 42,
                  ruleset: str = "v1"):
    """Reproduce the builder's emission order without extracting features.

    Yields, in the exact order ``build_sequences_llmstu`` assigned
    ``sample_idx``, one dict per emitted sequence.
    """
    f2v = json.load(open(frame_to_video))
    paths = sorted(labels.glob("*.jsonl")) if labels.is_dir() else [labels]
    by_video = parse_llmstu_labels(paths, f2v, allow_filename_fallback=False,
                                   ruleset=ruleset)
    train_vids, _ = split_videos(list(by_video.keys()), val_fraction, seed)

    idx = 0
    for video_id, obs_list in sorted(by_video.items()):
        split = "train" if video_id in train_vids else "val"
        for sid, seat_obs in assign_seats(obs_list).items():
            if len(seat_obs) < min_track_len:
                continue
            chunks = [[]]
            for obs in seat_obs:
                cur = chunks[-1]
                if cur and (obs.time_s - cur[-1].time_s > max_gap_s
                            or len(cur) >= max_track_len):
                    chunks.append([])
                    cur = chunks[-1]
                cur.append(obs)
            for chunk in chunks:
                if len(chunk) < min_track_len:
                    continue
                yield {
                    "sample_idx": idx, "split": split, "video_id": video_id,
                    "seat_id": sid,
                    "file_names": [o.meta.get("file_name", "") for o in chunk],
                    # Full cue-rule fields per frame, so a caller can recompute
                    # labels under a different ruleset without re-parsing.
                    # Additive: existing callers read only the keys they know.
                    "metas": [o.meta for o in chunk],
                    "times": np.array([o.time_s for o in chunk], dtype=np.float64),
                    "boxes": [list(o.bbox_xyxy) for o in chunk],
                }
                idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pose-cache", required=True)
    ap.add_argument("--labels",
                    default="../grounding_data/llmstu_tools/outputs/labels_tracked.jsonl")
    ap.add_argument("--frame-to-video",
                    default="../grounding_data/llmstu_tools/outputs/frame_to_video.json")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    d = np.load(args.pose_cache, allow_pickle=False)
    pose = {str(n): i for i, n in enumerate(d["names"])}
    pvecs = d["vecs"].astype(np.float32)
    print(f"pose cache: {len(pose)} crops, face_found {pvecs[:, 3].mean():.1%}")

    for sp in ("train", "val"):
        (dst / sp).mkdir(parents=True, exist_ok=True)

    n_patched = n_frames = n_missing = 0
    changed_ff = 0
    for rec in replay_chunks(Path(args.labels), Path(args.frame_to_video)):
        name = f"{rec['split']}/sample_{rec['sample_idx']:06d}.npz"
        fp = src / name
        if not fp.exists():
            raise SystemExit(f"replay produced {name}, which does not exist in {src}. "
                             "The builder's grouping has drifted — abort rather than "
                             "patch the wrong rows.")
        z = np.load(fp)
        x, t = z["x"].astype(np.float32), z["t"].astype(np.float64)
        # PROOF the replay lines up: the builder writes `t` straight from the
        # chunk, so exact equality is the strongest available check.
        if x.shape[0] != len(rec["times"]) or not np.array_equal(t, rec["times"]):
            raise SystemExit(
                f"{name}: replay/stored timestamp mismatch "
                f"({x.shape[0]} vs {len(rec['times'])} frames). Abort.")

        new_pose = np.zeros((x.shape[0], 4), dtype=np.float32)
        for i, fn in enumerate(rec["file_names"]):
            j = pose.get(fn)
            if j is None:
                n_missing += 1
                continue
            new_pose[i] = pvecs[j]
        changed_ff += int((new_pose[:, 3] != x[:, POSE_HI - 1]).sum())

        y = x.copy()
        y[:, POSE_LO:POSE_HI] = new_pose
        # dynamic features read pose, so they must be rebuilt from the new pose
        y[:, DYN_LO:DYN_HI] = compute_dynamic(rec["boxes"], new_pose)

        np.savez_compressed(dst / name, x=y, y_frames=z["y_frames"], y=z["y"], t=z["t"])
        n_patched += 1
        n_frames += x.shape[0]
        if n_patched % 1000 == 0:
            print(f"  {n_patched} sequences", flush=True)

    shutil.copy(src / "meta.json", dst / "meta.json")
    print(f"\npatched {n_patched} sequences / {n_frames} frames -> {dst}")
    print(f"  crops missing from the pose cache: {n_missing}")
    print(f"  frames whose face_found CHANGED  : {changed_ff} ({changed_ff / n_frames:.1%})")


if __name__ == "__main__":
    main()
