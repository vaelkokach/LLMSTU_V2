#!/usr/bin/env python
"""Merge sharded object-feature caches into one.

`precompute_objects --num-shards N` splits by FRAME, so the shards are disjoint
by construction and a merge is a concatenation. It is still checked: a duplicate
`file_name` across shards would mean the split leaked, and the sequence builder
would then silently take whichever copy it found first.

    python -m attention.merge_object_shards --out obj.npz shard0.npz shard1.npz ...
"""
import argparse, json
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("shards", nargs="+", type=Path)
    args = ap.parse_args()

    names, vecs, prompts = [], [], None
    for p in args.shards:
        d = np.load(p, allow_pickle=False)
        names += [str(n) for n in d["names"]]
        vecs.append(d["vecs"])
        pr = [str(x) for x in d["prompts"]]
        if prompts is None:
            prompts = pr
        elif prompts != pr:
            raise SystemExit(f"{p} has prompts {pr}, expected {prompts}")
        print(f"  {p.name}: {len(d['names'])} crops")

    dup = len(names) - len(set(names))
    if dup:
        raise SystemExit(
            f"{dup} file_name(s) appear in more than one shard. The split is by "
            f"frame and should be disjoint -- merging would hide which copy the "
            f"builder reads.")
    V = np.concatenate(vecs).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), vecs=V,
                        prompts=np.array(prompts),
                        provenance=json.dumps({"n_crops": len(names),
                                               "dim": int(V.shape[1]),
                                               "shards": [str(s) for s in args.shards]}))
    nz = (V != 0).any(axis=1).mean()
    print(f"\nmerged {len(names)} crops, {V.shape[1]} dims -> {args.out}")
    print(f"  crops with any object detected: {100 * nz:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
