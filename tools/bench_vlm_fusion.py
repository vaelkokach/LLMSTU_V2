#!/usr/bin/env python
"""Does the VLM second opinion actually improve the decision?

The `+vlm` entries are deployed and marked `recommended`, and nothing has ever
measured whether fusing the two opinions beats the temporal model alone. What IS
known [internal notes, not included] is that the VLM is WORSE overall on human gold ([value removed] vs the
temporal model) but better on `phone_use` ([value removed] vs [value removed]) -- so fusion has a
plausible mechanism to help one class and hurt others, and the policy could go
either way. That is a guess, and it is currently shipping as a recommendation.

Evaluated on the val split, with the crop images recovered per frame through
`replay_chunks`, which reproduces the sequence builder's emission order exactly.
Timestamps are verified against each npz's own `t`, so a frame is never paired
with the wrong image.

    python tools/bench_vlm_fusion.py --ckpt work_dirs/.../best.pth --limit 200
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "LLMDet"))
IGNORE = -100
CROPS = REPO / "grounding_data" / "LLMSTU" / "crops"


def macro_f1(yt, yp, k):
    out = []
    for c in range(k):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        if tp + fn == 0: continue
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn)
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out)) if out else 0.0


def per_class_f1(yt, yp, classes):
    out = {}
    for c, name in enumerate(classes):
        tp = int(((yt == c) & (yp == c)).sum()); fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        if tp + fn == 0: continue
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn)
        out[name] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/thesis/epochs240/mstcn_556_cue6_s42/checkpoints/best.pth")
    ap.add_argument("--root", default="llmstu_sequences_full_det")
    ap.add_argument("--limit", type=int, default=200, help="val sequences")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", default="agreement")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from attention.thesis_eval.runtime import load_runtime_model
    from attention.thesis_eval import data as D
    from attention.thesis_eval.patch_pose_columns import replay_chunks
    from attention.vlm_grounder import QwenGrounder
    from attention.fusion import fuse_student, Policy

    bundle = load_runtime_model(str(REPO / "LLMDet" / a.ckpt), device=a.device)
    classes = list(bundle.class_names)
    cols = D.column_index(bundle.feature_config)
    print(f"model    : {bundle.experiment_id} ({bundle.feature_config}, {classes})")

    # Map val sample_idx -> crop file names, in the builder's own order.
    print("recovering crop paths via replay_chunks ...", flush=True)
    chunks = {}
    for ch in replay_chunks(REPO / "grounding_data/llmstu_tools/outputs/labels_tracked.jsonl",
                            REPO / "grounding_data/llmstu_tools/outputs/frame_to_video.json"):
        if ch["split"] == "val":
            chunks[ch["sample_idx"]] = ch
    print(f"  {len(chunks)} val chunks", flush=True)

    files = sorted((REPO / "grounding_data" / a.root / "val").glob("*.npz"))
    grounder = QwenGrounder(device=a.device, classes=classes)
    prompt_classes = classes

    POLICIES = ("agreement", "pool", "product")
    ABSTAIN = -1
    yt, y_tmp, y_vlm = [], [], []
    y_fus = {p: [] for p in POLICIES}
    n_seq = 0
    t0 = time.time()
    import cv2
    from PIL import Image
    for f in files:
        idx = int(f.stem.split("_")[-1])
        ch = chunks.get(idx)
        if ch is None:
            continue
        z = np.load(f)
        t = np.asarray(z["t"]).ravel().astype(float)
        if len(t) != len(ch["times"]) or not np.allclose(t, ch["times"], atol=1e-6):
            continue                      # never pair a frame with the wrong crop
        x = np.ascontiguousarray(np.asarray(z["x"], np.float32)[:, cols])
        y = np.asarray(z["y_frames"]).ravel().astype(np.int64)
        keep = y != IGNORE
        if not keep.any():
            continue
        with torch.inference_mode():
            lg = bundle.model(torch.from_numpy(x[None]).float().to(a.device))["logits"][0]
            tprob = torch.softmax(lg.float(), dim=-1).cpu().numpy()

        # VLM over this chunk's crops, batched.
        names = ch["file_names"]
        vprob = np.full((len(names), len(classes)), np.nan, dtype=np.float64)
        imgs, at = [], []
        for i, nm in enumerate(names):
            p = CROPS / nm
            if not p.exists():
                continue
            imgs.append(Image.open(p).convert("RGB")); at.append(i)
        for s in range(0, len(imgs), a.batch):
            batch, where = imgs[s:s + a.batch], at[s:s + a.batch]
            frames = [cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) for im in batch]
            # score_students wants one frame + boxes; each crop IS the student,
            # so the box is the whole image.
            for fr, w in zip(frames, where):
                h, wd = fr.shape[:2]
                vprob[w] = grounder.score_students(fr, [[0, 0, wd, h]])[0]

        for i in np.flatnonzero(keep):
            yt.append(int(y[i])); y_tmp.append(int(tprob[i].argmax()))
            if np.isnan(vprob[i]).any():
                for pol in POLICIES:
                    y_fus[pol].append(int(tprob[i].argmax()))
                y_vlm.append(-1)
                continue
            y_vlm.append(int(vprob[i].argmax()))
            for pol in POLICIES:
                r = fuse_student(tprob[i], vprob[i], policy=Policy(pol),
                                 classes=classes)
                # AGREEMENT deliberately declines to pick when the two disagree.
                # The dashboard renders that as the abstain label, so that is
                # what it is scored as -- and the coverage is reported beside it,
                # because an abstaining number without its coverage is the one
                # thing this project refuses to quote.
                y_fus[pol].append(ABSTAIN if r.cue is None
                                  else classes.index(r.cue))
        n_seq += 1
        if n_seq % 10 == 0:
            print(f"  {n_seq} sequences, {len(yt)} frames, {time.time()-t0:.0f}s", flush=True)
        if n_seq >= a.limit:
            break

    yt = np.array(yt); k = len(classes); tmp = np.array(y_tmp)
    got = np.array(y_vlm) >= 0
    print(f"\n== {n_seq} val sequences, {len(yt)} frames "
          f"({got.mean():.0%} with a VLM opinion) ==")
    f_t = macro_f1(yt, tmp, k)
    f_v = macro_f1(yt[got], np.array(y_vlm)[got], k) if got.any() else float("nan")
    print(f"  temporal alone       macro-F1 {f_t:.4f}   coverage 100%")
    print(f"  VLM alone            macro-F1 {f_v:.4f}")
    res = {}
    for pol in POLICIES:
        fp = np.array(y_fus[pol])
        cov = float((fp >= 0).mean())
        sel = fp >= 0
        f_all = macro_f1(yt[sel], fp[sel], k)          # on what it answered
        f_base = macro_f1(yt[sel], tmp[sel], k)        # temporal on the SAME frames
        res[pol] = {"macro_f1_selective": f_all, "coverage": cov,
                    "temporal_same_frames": f_base}
        print(f"  fused: {pol:<10}    macro-F1 {f_all:.4f}   coverage {cov:5.1%}"
              f"   (temporal on those same frames {f_base:.4f}, {f_all-f_base:+.4f})")
    print("\n  per class, best policy by selective macro-F1:")
    best = max(POLICIES, key=lambda p: res[p]["macro_f1_selective"])
    fp = np.array(y_fus[best]); sel = fp >= 0
    pt = per_class_f1(yt[sel], tmp[sel], classes)
    pf = per_class_f1(yt[sel], fp[sel], classes)
    for c in classes:
        if c in pt:
            print(f"    {c:18} {pt[c]:.3f} -> {pf.get(c,0):.3f}   {pf.get(c,0)-pt[c]:+.3f}")
    if a.out:
        import json
        Path(a.out).write_text(json.dumps(
            {"ckpt": a.ckpt, "n_frames": int(len(yt)), "temporal": f_t,
             "vlm": f_v, "policies": res, "classes": classes}, indent=1))


if __name__ == "__main__":
    main()
