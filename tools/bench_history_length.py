#!/usr/bin/env python
"""How many seconds of history does a student need before the first cue?

`inference.min_frames_for_pred` is 10. It was chosen when the dashboard fed the
model every frame, so 10 frames took 0.3 s. Since the frame-rate fix
[internal notes, not included] history is admitted at 1 Hz, and the same 10 now costs ~10 s of
"warming up" at the start of a session and after every tracker ID switch. The
value was never measured at 1 Hz.

This measures it the way deployment predicts: at position p, a track that is
only `k` samples old has exactly the last `k` frames and nothing before them, and
the cue shown is the model's output for p. Every k is scored on the SAME
positions -- those with at least `window` frames behind them -- so the only
thing that changes between rows is how much history the model was given. The
`window` row is steady state, the number the thesis quotes the deployment at.

Validation split only [internal notes, not included]. No calibration, so absolute
values are not comparable to the published macro-F1; the drop against the
steady-state row is the result.

    python tools/bench_history_length.py \
        --ckpt work_dirs/thesis/ff_det/mstcn_553_ff_s42/checkpoints/best.pth \
        --root llmstu_sequences_full_det
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))
IGNORE = -100


def per_class(yt, yp, k):
    """(macro-F1, macro-precision, macro-recall) over classes present in yt."""
    f1s, ps, rs = [], [], []
    for c in range(k):
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        ps.append(p)
        rs.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    if not f1s:
        return 0.0, 0.0, 0.0
    return float(np.mean(f1s)), float(np.mean(ps)), float(np.mean(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="relative to LLMDet/")
    ap.add_argument("--root", default="llmstu_sequences_full_det")
    ap.add_argument("--history", default="1,2,3,4,5,6,8,10,16,32",
                    help="history lengths to score, in samples (= seconds at 1 Hz)")
    ap.add_argument("--window", type=int, default=48,
                    help="deployed window_size; the steady-state reference row")
    ap.add_argument("--stride", type=int, default=5, help="score every Nth position")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from attention.thesis_eval import data as D
    from attention.thesis_eval.runtime import load_runtime_model

    b = load_runtime_model(str(REPO / "LLMDet" / a.ckpt), device=a.device)
    cols = D.column_index(b.feature_config)
    ks = sorted({int(x) for x in a.history.split(",")} | {a.window})
    files = sorted((REPO / "grounding_data" / a.root / "val").glob("*.npz"))[:a.limit]
    if not files:
        sys.exit(f"no val sequences under grounding_data/{a.root}/val")
    print(f"{b.experiment_id} ({b.feature_config}) | {len(files)} val sequences | "
          f"reference = {a.window} samples")

    acc = {k: {"t": [], "p": []} for k in ks}
    for f in files:
        z = np.load(f)
        x = np.ascontiguousarray(np.asarray(z["x"], np.float32)[:, cols])
        y = np.asarray(z["y_frames"]).ravel().astype(np.int64)
        common = [p for p in range(max(ks) - 1, len(y), a.stride) if y[p] != IGNORE]
        if not common:
            continue
        for k in ks:
            w = np.stack([x[p - k + 1:p + 1] for p in common])
            with torch.inference_mode():
                lg = b.model(torch.from_numpy(w).float().to(a.device))["logits"]
            acc[k]["t"].append(y[common])
            acc[k]["p"].append(lg[:, -1].float().argmax(-1).cpu().numpy())

    n_cls = len(b.class_names)
    res = {}
    for k in ks:
        yt, yp = np.concatenate(acc[k]["t"]), np.concatenate(acc[k]["p"])
        res[k] = per_class(yt, yp, n_cls) + (len(yt),)
    ref = res[a.window][0]
    print(f"\n{'history':>8}  {'macro-F1':>9}  {'precision':>9}  {'recall':>7}  "
          f"{'vs steady':>9}")
    for k in ks:
        f1, p, r, _ = res[k]
        tag = "(ref)" if k == a.window else f"{f1 - ref:+.4f}"
        print(f"{k:>6} s  {f1:>9.4f}  {p:>9.4f}  {r:>7.4f}  {tag:>9}")
    print(f"\nAll rows scored on the same {res[ks[0]][3]} positions.")


if __name__ == "__main__":
    main()
