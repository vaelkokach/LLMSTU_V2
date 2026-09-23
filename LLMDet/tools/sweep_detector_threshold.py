#!/usr/bin/env python3
"""Re-tune the runtime detection threshold (and prompt) for a fine-tuned checkpoint.

WHY: `configs/attention_temporal.yaml` ships `text_prompt: "student"` with
`score_thr: 0.45`, inherited from the PRE-fine-tuning model. But the word
"student" appears in 0 of 36,339 fine-tuning captions -- training captions are
cue phrases ("using laptop. listening attentively.") -- so that prompt is out of
distribution for the fine-tuned head and its scores collapse. Probing
iter_5000 showed localization recall [value removed] at thr 0.35 vs [value removed] at thr 0.05:
the students ARE found, they just score low. Deployed as configured the runtime
would drop over half the students, and Branch B only sees what the detector
returns, so the loss propagates silently into the cue output.

This sweeps threshold x prompt and reports the precision/recall knee.

Efficiency: inference runs ONCE per (frame, prompt); every threshold is then
evaluated offline from the cached scores. Sweeping by re-running inference
would be ~20x more expensive for identical numbers.

Usage:
  python tools/sweep_detector_threshold.py \
      --ckpt work_dirs/student_llmstu_exact/iter_25000.pth \
      --n 120 --device cuda:0 --out work_dirs/probe/threshold_sweep.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

CFG = "configs/student_llmstu_exact.py"
VAL = "../grounding_data/llmstu_tools/outputs/odvg_val.jsonl"
IMG_ROOT = "../grounding_data/stu_img/frames"

# "student" is the current runtime prompt (out-of-vocabulary for the fine-tune).
# The rest ARE in the trained caption vocabulary (regen_odvg.ACTIVITY_PHRASE),
# so they test whether an in-vocabulary person-anchor scores better.
# Generic person-anchors: safe as a runtime detector prompt because they do not
# name a behaviour, so they cannot bias detection toward one activity.
GENERIC_PROMPTS = ["student", "person", "a student sitting"]
# Content prompts: included to measure the vocabulary effect, NOT to be used as
# the runtime anchor (see the warning in the recommendation block).
CONTENT_PROMPTS = ["sitting at desk", "listening attentively", "using laptop"]
PROMPTS = GENERIC_PROMPTS + CONTENT_PROMPTS
THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.60]


def iou_mat(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)[:, None, :]
    g = np.asarray(gt, dtype=np.float32)[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0]); y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2]); y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / np.maximum(ap + ag - inter, 1e-6)


def score_at(cached, thr, iou_thr=0.5):
    """Greedy one-to-one TP/FP/FN across all frames at one threshold."""
    tp = fp = fn = 0
    for boxes, scores, gt in cached:
        keep = scores >= thr
        pb = boxes[keep]
        order = np.argsort(-scores[keep])
        pb = pb[order]
        M = iou_mat(pb, gt)
        used = set()
        for i in range(len(pb)):
            j = -1; best = iou_thr
            for k in range(len(gt)):
                if k in used:
                    continue
                if M[i, k] >= best:
                    best = M[i, k]; j = k
            if j >= 0:
                used.add(j); tp += 1
            else:
                fp += 1
        fn += len(gt) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="work_dirs/probe/threshold_sweep.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from mmdet.apis import init_detector, inference_detector

    rows = [json.loads(l) for l in open(VAL)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    model = init_detector(CFG, args.ckpt, device=args.device)

    results = {}
    for prompt in PROMPTS:
        cached = []
        for r in rows:
            im = cv2.imread(str(Path(IMG_ROOT) / r["filename"]))
            if im is None:
                continue
            o = inference_detector(model, im, text_prompt=prompt,
                                   custom_entities=True)
            p = o.pred_instances
            cached.append((p.bboxes.cpu().numpy(), p.scores.cpu().numpy(),
                           [rg["bbox"] for rg in r["grounding"]["regions"]]))
        curve = {f"{t:.2f}": score_at(cached, t) for t in THRESHOLDS}
        best_t = max(curve, key=lambda k: curve[k]["f1"])
        results[prompt] = {"curve": curve, "best_threshold": float(best_t),
                           "best": curve[best_t]}
        print(f"\n=== prompt: {prompt!r}  (frames={len(cached)})")
        print(f"{'thr':>6} {'prec':>7} {'recall':>7} {'F1':>7}   TP/FP/FN")
        for t, s in curve.items():
            mark = "  <== best F1" if t == best_t else ""
            print(f"{t:>6} {s['precision']:7.3f} {s['recall']:7.3f} "
                  f"{s['f1']:7.3f}   {s['tp']}/{s['fp']}/{s['fn']}{mark}")

    ranked = sorted(results.items(), key=lambda kv: -kv[1]["best"]["f1"])
    print("\n===== RECOMMENDATION =====")
    for name, r in ranked:
        b = r["best"]
        tag = "" if name in GENERIC_PROMPTS else "   [content prompt - see warning]"
        print(f"  {name:22s} best F1 {b['f1']:.3f} at thr {r['best_threshold']:.2f} "
              f"(P {b['precision']:.3f} R {b['recall']:.3f}){tag}")

    # Content prompts ("using laptop") can top aggregate F1 simply by naming the
    # MAJORITY activity -- using_laptop is 42,984 of the deduped records. Such a
    # prompt retrieves most students because most students are doing that thing,
    # while systematically missing the minority behaviours (phone, head-down,
    # talking). For attention-loss detection those minorities are the entire
    # point, so a content prompt is the wrong runtime person-anchor no matter
    # how good its aggregate F1 looks.
    generic = [(n, r) for n, r in ranked if n in GENERIC_PROMPTS]
    top, tr = generic[0] if generic else ranked[0]
    if ranked[0][0] not in GENERIC_PROMPTS:
        print(f"\n  NOTE: {ranked[0][0]!r} scores highest overall but is a CONTENT "
              f"prompt naming a majority activity.\n  Using it as the runtime "
              f"person-anchor would bias detection toward that behaviour and miss "
              f"the\n  off-task students the system exists to find. Recommending the "
              f"best GENERIC prompt instead.")
    print(f"\n  -> set attention_temporal.yaml: text_prompt: {top!r}, "
          f"score_thr: {tr['best_threshold']:.2f}")
    print("  Current shipped values are text_prompt: 'student', score_thr: 0.45 —")
    cur = results.get("student", {}).get("curve", {}).get("0.45")
    if cur:
        print(f"  which measure P {cur['precision']:.3f} R {cur['recall']:.3f} "
              f"F1 {cur['f1']:.3f} on this checkpoint.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"ckpt": args.ckpt, "frames": len(rows), "results": results,
         "recommended_prompt": top,
         "recommended_threshold": tr["best_threshold"],
         "generic_prompts": GENERIC_PROMPTS,
         "highest_f1_prompt_overall": ranked[0][0],
         "note": ("Recommendation is restricted to generic person-anchors. A "
                  "content prompt may top aggregate F1 by naming the majority "
                  "activity while missing the minority off-task behaviours the "
                  "system exists to detect.")}, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
