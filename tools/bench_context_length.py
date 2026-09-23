#!/usr/bin/env python
"""How much context should the deployed window carry?

`inference.window_size` is 32 frames, which since the frame-rate fix is ~32 s of
real time. The training sequences run to a median of 47 frames and p75 of 60, so
a longer window is within the distribution the model saw -- but longer is not
automatically better: a TCN's receptive field is fixed, and extra context beyond
it only adds frames the model averages over.

Measured the way deployment actually predicts: take the last `N` frames ending at
position p and read the prediction FOR p (causal, exactly `predict_window`), for
every 5th position that has N frames behind it. No calibration, so absolute
values are not comparable to the published macro-F1 -- the comparison across N is
the result.
"""
import argparse, sys
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="llmstu_sequences_full_det")
    ap.add_argument("--windows", default="8,16,24,32,48,64")
    ap.add_argument("--stride", type=int, default=5, help="evaluate every Nth position")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from attention.thesis_eval.runtime import load_runtime_model
    from attention.thesis_eval import data as D

    b = load_runtime_model(str(REPO / "LLMDet" / a.ckpt), device=a.device)
    cols = D.column_index(b.feature_config)
    Ns = [int(x) for x in a.windows.split(",")]
    files = sorted((REPO / "grounding_data" / a.root / "val").glob("*.npz"))[:a.limit]
    print(f"{b.experiment_id} ({b.feature_config}) | {len(files)} val sequences")

    acc = {n: {"t": [], "p": []} for n in Ns}
    for f in files:
        z = np.load(f)
        x = np.ascontiguousarray(np.asarray(z["x"], np.float32)[:, cols])
        y = np.asarray(z["y_frames"]).ravel().astype(np.int64)
        T = len(y)
        # EVERY window is scored on the SAME positions: those with enough
        # frames behind them for the LONGEST window. Letting each N use the
        # positions it can reach compares different frame sets -- longer windows
        # then score only late positions in long sequences, which is a different
        # and probably easier population. The first run of this looked like a
        # huge win for N=64 and was scoring three frames.
        floor = max(Ns) - 1
        common = [p for p in range(floor, T, a.stride) if y[p] != IGNORE]
        if not common:
            continue
        for n in Ns:
            pos = common
            # One batch of equal-length causal windows, each ending at its p.
            w = np.stack([x[p - n + 1:p + 1] for p in pos])
            with torch.inference_mode():
                lg = b.model(torch.from_numpy(w).float().to(a.device))["logits"]
                pred = lg[:, -1].float().argmax(-1).cpu().numpy()
            acc[n]["t"].append(y[pos]); acc[n]["p"].append(pred)

    k = len(b.class_names)
    print(f"\n{'window':>8}  {'real time @1Hz':>14}  {'frames scored':>13}  {'macro-F1':>9}  {'vs 32':>8}")
    base = None
    rows = []
    for n in Ns:
        yt = np.concatenate(acc[n]["t"]); yp = np.concatenate(acc[n]["p"])
        f1 = macro_f1(yt, yp, k)
        rows.append((n, f1, len(yt)))
        if n == 32:
            base = f1
    for n, f1, cnt in rows:
        d = "" if base is None else (" (ref)" if n == 32 else f"{f1-base:+8.4f}")
        print(f"{n:>8}  {n:>11} s  {cnt:>13}  {f1:>9.4f}  {d:>8}")
    print(f"\nAll rows scored on the SAME {rows[0][2]} positions -- those with")
    print(f"{max(Ns)} frames of history available -- so the only difference is how")
    print("much of that history each window was allowed to see.")


if __name__ == "__main__":
    main()
