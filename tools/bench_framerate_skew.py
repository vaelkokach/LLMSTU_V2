#!/usr/bin/env python
"""What does serving faster than the training frame rate cost?

The temporal sequences were built from the LLMSTU annotations at **1.01 fps**
(median inter-frame gap [value removed] s over 300 sampled sequences; 92% within
0.9-1.0 s). The deployed window is 32 frames, so in training a window spans
**31.8 s of real time**.

The live path appends every analysed frame to each student's history, so the
history runs at whatever rate the pipeline achieves -- 2.29 fps on the Space,
4.33 on an A100. A 32-frame window then spans 14.0 s and 7.4 s. The model's
receptive field is defined in FRAMES, so raising the frame rate shrinks the real
time it can see. That is a train/serve skew, and it points the opposite way to
the intuition that more frames must be better.

This measures it. A classroom at 4 fps gives consecutive frames that are nearly
identical, so oversampling is simulated by **repeating each frame k times** --
not interpolation, which would invent motion that a 4 fps camera does not see.
Predictions are then read back at the original frame positions and scored against
the same labels, so the only thing that changed is how much real time the
window covers.

    python tools/bench_framerate_skew.py --model epochs240/mstcn_556_hp/checkpoints/best.pth
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))

IGNORE = -100


def macro_f1(y_true, y_pred, k):
    out = []
    for c in range(k):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out)) if out else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to best.pth, relative to LLMDet/")
    ap.add_argument("--split", default="val", choices=["val"],
                    help="val only: the test split is closed [internal notes, not included]")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--rates", default="1,2,4",
                    help="factors to simulate. >1 oversamples (serve faster than "
                         "trained); 0.5 subsamples by 2 (serve slower), which is "
                         "the control for whether the effect is about real-time "
                         "span or about duplicate frames")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="stddev of Gaussian noise added to DUPLICATED frames "
                         "only, as a fraction of each column's own std. A real "
                         "4 fps camera gives near-identical, not identical, "
                         "frames; if the loss is an artefact of exact repeats it "
                         "should shrink here.")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from attention.thesis_eval.runtime import load_runtime_model
    from attention.thesis_eval import data as D

    ck = (REPO / "LLMDet" / a.model) if not Path(a.model).is_absolute() else Path(a.model)
    bundle = load_runtime_model(str(ck), device=a.device)
    print(f"model    : {bundle.experiment_id} ({bundle.feature_config}, "
          f"{bundle.input_dim}-dim, {len(bundle.class_names)} classes)")

    import json
    rec = json.loads((ck.parent.parent / "run_record.json").read_text())
    root = REPO / "grounding_data" / Path(rec["spec"]["sequence_root"]).name \
        if "sequence_root" in rec.get("spec", {}) else \
        REPO / "grounding_data" / "llmstu_sequences_full_det"
    files = sorted((root / a.split).glob("*.npz"))[:a.limit]
    print(f"sequences: {len(files)} from {root.name}/{a.split}")

    # NOT bundle.live_columns: that selects from the 556-wide LIVE vector. The
    # stored sequences are the full v570 layout, so the columns to take are the
    # ones the feature config names in THAT layout.
    cols = D.column_index(bundle.feature_config)
    rates = [float(r) for r in a.rates.split(",")]
    acc = {r: {"true": [], "pred": []} for r in rates}
    rng = np.random.default_rng(0)
    dts = []

    for f in files:
        z = np.load(f)
        x = np.asarray(z["x"], dtype=np.float32)
        y = np.asarray(z["y_frames"]).ravel().astype(np.int64)
        t = np.asarray(z["t"]).ravel().astype(float)
        if len(t) > 1:
            dts.append(float(np.median(np.diff(t))))
        x = np.ascontiguousarray(x[:, cols])
        keep = y != IGNORE
        if not keep.any():
            continue
        for r in rates:
            # Repeat each frame r times: the same real seconds, sampled r times
            # denser, which is exactly what a faster pipeline produces.
            if r >= 1:
                n = int(round(r))
                xr = np.repeat(x, n, axis=0)      # axis 0 is TIME here
                if a.jitter and n > 1:
                    sd = x.std(axis=0, keepdims=True) * a.jitter
                    noise = rng.normal(0.0, 1.0, xr.shape).astype(np.float32) * sd
                    noise[::n] = 0.0              # leave the real frames alone
                    xr = xr + noise
                take, yk, kk = slice(None, None, n), y, keep
            else:
                # Serve SLOWER than trained: every m-th frame, so a 32-frame
                # window covers m x 31.8 s. Scored on the frames it kept.
                m = int(round(1.0 / r))
                xr, yk, kk = x[::m], y[::m], keep[::m]
                take = slice(None)
            with torch.inference_mode():
                out = bundle.model(
                    torch.from_numpy(np.ascontiguousarray(xr)[None])
                    .float().to(bundle.device))
                pred = out["logits"][0].float().argmax(-1).cpu().numpy()
            # Read back at the ORIGINAL frame positions, so every rate is scored
            # on real frames against the same labels.
            pred = pred[take][:len(yk)]
            acc[r]["true"].append(yk[kk])
            acc[r]["pred"].append(pred[kk])

    k = len(bundle.class_names)
    print(f"\ntraining frame rate: {1/np.median(dts):.2f} fps "
          f"(median dt {np.median(dts):.3f} s)")
    print(f"window_size 32 frames -> {32*np.median(dts):.1f} s of real time in training\n")
    print(f"{'served at':>12}  {'window spans':>13}  {'macro-F1':>9}  {'vs trained rate':>16}")
    # The reference is the TRAINED rate (r == 1), not whichever rate was listed
    # first: the claim being tested is "how much does departing from the trained
    # rate cost", and 1.0 is the only row that answers to it.
    base = None
    rows = []
    # Reference row first, so the deltas are always printable.
    for r in sorted(rates, key=lambda v: abs(v - 1.0)):
        yt = np.concatenate(acc[r]["true"])
        yp = np.concatenate(acc[r]["pred"])
        f1 = macro_f1(yt, yp, k)
        if abs(r - 1.0) < 1e-9:
            base = f1
        fps = r / np.median(dts)
        rows.append((fps, f1, r))
        print(f"{fps:9.2f} fps  {32/fps:10.1f} s  {f1:9.4f}  "
              + ("     (reference)" if base is not None and abs(r-1.0) < 1e-9
                 else f"{f1-base:+16.4f}" if base is not None else " "*16))
    print("\n(1x is the rate the sequences were built at; 2x ~ the HF Space, "
          "4x ~ one A100 at full speed)")


if __name__ == "__main__":
    main()
