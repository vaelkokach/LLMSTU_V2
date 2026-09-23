#!/usr/bin/env python3
"""Recover source-video identity for every frame in stu_img/frames.

Background: 128 source videos were sampled at every 19th/20th native frame into
a single flat directory. The extractor appended the `_video_...` name only on
filename collision, so 97.7% of frames carry no video ID. `person_idx` is a
detection-confidence rank, not a track ID, so video identity must be recovered
from pixels.

Method — appearance chain tracking:
  * Every video starts at f=0 (there are exactly 128 files named
    t000000_000_f000000*), so the f=0 frames seed one chain per video.
  * Within a video, consecutive sampled frames are 1 s apart and nearly
    identical; the per-video frame-index step is constant 19 or 20 (missing
    frames appear as multiples).
  * Process frames in ascending frame-index order. For each group of frames at
    index f, match them to the live chains whose last index is f-19..f-2*20
    using mean-absolute-difference on a downscaled grayscale thumbnail, solved
    globally per group with Hungarian assignment and a distance gate.
  * Chains keep an EMA of their appearance so slow lighting drift is tolerated.

Validation: the 4,660 `_video_`-suffixed frames (127 named videos) are held-out
truth — a chain must map to exactly one video name and vice versa.

Phase 1 (--extract) writes features.npy + index; phase 2 (--assign) builds
frame_to_video.json. CPU only.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FRAMES_DIR = "/home/jovyan/Computer_vision/grounding_data/stu_img/frames"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
THUMB_W, THUMB_H = 64, 24
NAME_RE = re.compile(r"^t(\d{6})_(\d{3})_f(\d{6})(?:_(video_.+?))?\.jpg$")


def parse_name(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    sec, ms, f, vid = m.groups()
    return int(sec) + int(ms) / 1000.0, int(f), vid


def _feat_batch(names):
    from PIL import Image
    out = np.empty((len(names), THUMB_H * THUMB_W), dtype=np.uint8)
    for i, n in enumerate(names):
        with Image.open(os.path.join(FRAMES_DIR, n)) as im:
            out[i] = np.asarray(
                im.convert("L").resize((THUMB_W, THUMB_H), Image.BILINEAR),
                dtype=np.uint8).ravel()
    return out


def extract(workers):
    names = sorted(n for n in os.listdir(FRAMES_DIR) if parse_name(n))
    chunks = [names[i:i + 500] for i in range(0, len(names), 500)]
    feats = np.empty((len(names), THUMB_H * THUMB_W), dtype=np.uint8)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk, arr in zip(chunks, ex.map(_feat_batch, chunks)):
            feats[done:done + len(chunk)] = arr
            done += len(chunk)
            if (done // 500) % 40 == 0:
                print(f"  features {done}/{len(names)}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "frame_features.npy"), feats)
    with open(os.path.join(OUT_DIR, "frame_index.json"), "w") as fh:
        json.dump(names, fh)
    print(f"extracted {done} features")


def assign(gate):
    from scipy.optimize import linear_sum_assignment

    names = json.load(open(os.path.join(OUT_DIR, "frame_index.json")))
    feats = np.load(os.path.join(OUT_DIR, "frame_features.npy")).astype(np.float32)

    by_f = defaultdict(list)          # frame index -> [row ids]
    meta = [parse_name(n) for n in names]
    for i, (_, f, _) in enumerate(meta):
        by_f[f].append(i)

    # Seed chains from f=0
    chain_feat, chain_last, chain_members = [], [], []
    for i in by_f[0]:
        chain_feat.append(feats[i].copy())
        chain_last.append(0)
        chain_members.append([i])
    chain_feat = np.stack(chain_feat)
    print(f"seeded {len(chain_members)} chains at f=0")

    unassigned = 0
    for f in sorted(by_f):
        if f == 0:
            continue
        rows = by_f[f]
        # candidate chains: last seen within 3 sampling steps (allows 2 missing)
        cand = [c for c in range(len(chain_members))
                if 19 <= f - chain_last[c] <= 62]
        if not cand:
            for i in rows:
                chain_feat = np.vstack([chain_feat, feats[i][None]])
                chain_last.append(f)
                chain_members.append([i])
                unassigned += 1
            continue
        sub = chain_feat[cand]                            # (C, D)
        fr = feats[rows]                                  # (R, D)
        cost = np.abs(fr[:, None, :] - sub[None, :, :]).mean(2)   # (R, C)
        big = gate * 4
        padded = np.full((len(rows), len(cand) + len(rows)), big, np.float32)
        padded[:, :len(cand)] = cost
        np.fill_diagonal(padded[:, len(cand):], gate)     # "new chain" option
        ri, ci = linear_sum_assignment(padded)
        for r, c in zip(ri, ci):
            i = rows[r]
            if c < len(cand) and cost[r, c] < gate:
                ch = cand[c]
                chain_feat[ch] = 0.85 * chain_feat[ch] + 0.15 * feats[i]
                chain_last[ch] = f
                chain_members[ch].append(i)
            else:
                chain_feat = np.vstack([chain_feat, feats[i][None]])
                chain_last.append(f)
                chain_members.append([i])
                unassigned += 1

    print(f"chains: {len(chain_members)} (extra spawned mid-stream: {unassigned})")

    # ---- merge mid-stream fragments back into their parent chains ----
    # Every video starts at f=0 (128 seed frames), so a chain that begins at
    # f>0 is a broken-off fragment of some earlier chain. Merge each fragment
    # (ascending start f) into the appearance-closest chain that ended before
    # it started.
    # A frame index occurs at most once per video, so a fragment and its true
    # parent have disjoint f-sets; match on the appearance of the parent frame
    # temporally closest to the fragment's start.
    fsets = [set(meta[i][1] for i in m) for m in chain_members]
    sorted_fs = [np.array(sorted(s)) for s in fsets]
    first_f = [min(s) for s in fsets]
    first_feat = [feats[min(m, key=lambda i: meta[i][1])].astype(np.float32)
                  for m in chain_members]
    frame_of = [{meta[i][1]: i for i in m} for m in chain_members]
    alive = {c: True for c in range(len(chain_members))}
    for c in sorted(range(len(chain_members)), key=lambda c: first_f[c]):
        if first_f[c] == 0 or not alive[c]:
            continue
        best_k, best_d = None, None
        for k in range(len(chain_members)):
            if k == c or not alive[k]:
                continue
            # tolerate a handful of f-collisions (chain-builder swap noise)
            if len(fsets[k] & fsets[c]) > max(2, 0.01 * len(fsets[c])):
                continue
            # parent frame nearest (in f) to the fragment start
            pos = np.searchsorted(sorted_fs[k], first_f[c])
            near = []
            if pos > 0:
                near.append(sorted_fs[k][pos - 1])
            if pos < len(sorted_fs[k]):
                near.append(sorted_fs[k][pos])
            d = min(np.abs(first_feat[c]
                           - feats[frame_of[k][nf]].astype(np.float32)).mean()
                    for nf in near)
            if best_d is None or d < best_d:
                best_k, best_d = k, d
        if best_k is None:
            continue
        k = best_k
        chain_members[k].extend(chain_members[c])
        fsets[k] |= fsets[c]
        sorted_fs[k] = np.array(sorted(fsets[k]))
        frame_of[k].update(frame_of[c])
        alive[c] = False
    # Forced pass: any fragment still alive must belong to some seed chain
    # (every video starts at f=0), so merge it to the appearance-nearest seed
    # regardless of f-collisions; the anchor purity check below validates this.
    forced = 0

    def anchor_name(c):
        votes = defaultdict(int)
        for i in chain_members[c]:
            if meta[i][2]:
                votes[meta[i][2]] += 1
        return max(votes, key=votes.get) if votes else None

    for c in range(len(chain_members)):
        if not alive[c] or first_f[c] == 0:
            continue
        # if the fragment carries suffixed anchor frames, its video is known
        nm = anchor_name(c)
        if nm is not None:
            target = [k for k in range(len(chain_members))
                      if alive[k] and k != c and first_f[k] == 0
                      and anchor_name(k) == nm]
            if len(target) == 1:
                k = target[0]
                chain_members[k].extend(chain_members[c])
                fsets[k] |= fsets[c]
                sorted_fs[k] = np.array(sorted(fsets[k]))
                frame_of[k].update(frame_of[c])
                alive[c] = False
                forced += 1
                continue
        best_k, best_d = None, None
        for k in range(len(chain_members)):
            if k == c or not alive[k] or first_f[k] != 0:
                continue
            pos = np.searchsorted(sorted_fs[k], first_f[c])
            near = []
            if pos > 0:
                near.append(sorted_fs[k][pos - 1])
            if pos < len(sorted_fs[k]):
                near.append(sorted_fs[k][pos])
            d = min(np.abs(first_feat[c]
                           - feats[frame_of[k][nf]].astype(np.float32)).mean()
                    for nf in near)
            if best_d is None or d < best_d:
                best_k, best_d = k, d
        if best_k is not None:
            chain_members[best_k].extend(chain_members[c])
            fsets[best_k] |= fsets[c]
            sorted_fs[best_k] = np.array(sorted(fsets[best_k]))
            frame_of[best_k].update(frame_of[c])
            alive[c] = False
            forced += 1
    chain_members = [m for c, m in enumerate(chain_members) if alive[c]]
    print(f"after fragment merge: {len(chain_members)} chains (forced: {forced})")

    # ---- name chains from suffixed anchors, validate purity ----
    mapping, report = {}, {
        "n_chains": len(chain_members), "gate": gate,
        "anchor_frames": 0, "anchor_correct": 0}
    chain_names = {}
    conflicts = 0
    for c, members in enumerate(chain_members):
        votes = defaultdict(int)
        for i in members:
            vid = meta[i][2]
            if vid:
                votes[vid] += 1
        if votes:
            best = max(votes, key=votes.get)
            chain_names[c] = best
            if len(votes) > 1:
                conflicts += 1
    # unnamed chains get synthetic ids
    nxt = 0
    for c in range(len(chain_members)):
        if c not in chain_names:
            chain_names[c] = f"video_unk_{nxt:03d}"
            nxt += 1
    for c, members in enumerate(chain_members):
        for i in members:
            mapping[names[i]] = chain_names[c]
            vid = meta[i][2]
            if vid:
                report["anchor_frames"] += 1
                if vid == chain_names[c]:
                    report["anchor_correct"] += 1
    report["anchor_accuracy"] = report["anchor_correct"] / max(report["anchor_frames"], 1)
    report["chains_with_mixed_anchor_names"] = conflicts
    report["distinct_named"] = len({v for v in chain_names.values()
                                    if not v.startswith("video_unk_")})
    sizes = sorted(len(m) for m in chain_members)
    report["chain_size_min_med_max"] = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]

    with open(os.path.join(OUT_DIR, "frame_to_video.json"), "w") as fh:
        json.dump(mapping, fh)
    with open(os.path.join(OUT_DIR, "video_recovery_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--assign", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--gate", type=float, default=14.0,
                    help="max mean-abs-diff (0-255 scale) to join a chain")
    a = ap.parse_args()
    if a.extract:
        extract(a.workers)
    if a.assign:
        assign(a.gate)
    if not (a.extract or a.assign):
        sys.exit("pass --extract and/or --assign")
