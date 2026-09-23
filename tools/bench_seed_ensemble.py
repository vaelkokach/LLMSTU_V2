#!/usr/bin/env python
"""Does averaging the seeds we already trained buy accuracy?

The temporal head is ~18% of the frame budget and, since the frame-rate fix, runs
at 1 Hz. Three of them is still a rounding error against one detector pass, so if
seed averaging helps at all it is close to free accuracy -- the cheapest possible
trade of throughput for quality, needing no new training.

Protocol matches tools/bench_framerate_skew.py: whole-sequence, non-causal, no
calibration. Absolute values are therefore NOT comparable to the published
macro-F1; only the comparison between rows here is the result.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))
IGNORE = -100


def macro_f1(yt, yp, k):
    out = []
    for c in range(k):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        if tp + fn == 0: continue
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn)
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out)) if out else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from attention.thesis_eval.runtime import load_runtime_model
    from attention.thesis_eval import data as D

    bundles = [load_runtime_model(str(REPO / "LLMDet" / c), device=a.device)
               for c in a.ckpts]
    b0 = bundles[0]
    cols = D.column_index(b0.feature_config)
    files = sorted((REPO / "grounding_data" / a.root / "val").glob("*.npz"))[:a.limit]
    print(f"{len(bundles)} seeds | {b0.feature_config} | {len(files)} val sequences")

    per_seed = [[] for _ in bundles]
    ens, truth = [], []
    for f in files:
        z = np.load(f)
        x = np.ascontiguousarray(np.asarray(z["x"], np.float32)[:, cols])
        y = np.asarray(z["y_frames"]).ravel().astype(np.int64)
        keep = y != IGNORE
        if not keep.any(): continue
        t = torch.from_numpy(x[None]).float().to(a.device)
        probs = []
        with torch.inference_mode():
            for i, b in enumerate(bundles):
                lg = b.model(t)["logits"][0].float()
                p = torch.softmax(lg, dim=-1).cpu().numpy()
                probs.append(p)
                per_seed[i].append(p.argmax(-1)[keep])
        ens.append(np.mean(probs, axis=0).argmax(-1)[keep])
        truth.append(y[keep])

    yt = np.concatenate(truth); k = len(b0.class_names)
    print()
    for i, c in enumerate(a.ckpts):
        f1 = macro_f1(yt, np.concatenate(per_seed[i]), k)
        print(f"  seed {Path(c).parts[-3][-3:]:>4}            macro-F1 {f1:.4f}")
    singles = [macro_f1(yt, np.concatenate(p), k) for p in per_seed]
    e = macro_f1(yt, np.concatenate(ens), k)
    print(f"  ---")
    print(f"  best single seed      macro-F1 {max(singles):.4f}")
    print(f"  mean of seeds         macro-F1 {np.mean(singles):.4f}")
    print(f"  ENSEMBLE of {len(bundles)}         macro-F1 {e:.4f}"
          f"   ({e-max(singles):+.4f} vs best, {e-np.mean(singles):+.4f} vs mean)")


if __name__ == "__main__":
    main()
