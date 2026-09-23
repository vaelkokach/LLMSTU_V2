"""Branch-B baseline-chain evaluation on the val sequences.

Produces the thesis comparison table:
  1. majority-class baseline        (per-frame accuracy / macro-F1)
  2. temporal transformer           (per-frame accuracy / macro-F1)
  3. rule/event layer run on label vs model cue timelines
     (window-state agreement, episode counts, event metrics of model-derived
     events against pseudo-label-derived events — NOT human ground truth).

Usage (from LLMDet/):
    python -m attention.eval_baseline_chain \
        --config configs/attention_temporal.yaml \
        --ckpt work_dirs/attention_temporal_v2/checkpoints/best.pth \
        --val-dir ../grounding_data/llmstu_sequences/val \
        --out work_dirs/attention_temporal_v2/baseline_chain_eval.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from attention.taxonomy import CUE_CLASSES, LEGACY_7CLASS_REMAP
from attention.temporal_model import AttentionTransformer
from attention.rule_baseline import RuleConfig, classify_timeline, STATES
from attention.events import EventConfig, segment_events
from attention.event_metrics import evaluate_events

IGNORE = -100


def load_sequences(val_dir: Path):
    lut = np.arange(7, dtype=np.int64)
    for s, d in LEGACY_7CLASS_REMAP.items():
        lut[s] = d
    seqs = []
    for fp in sorted(val_dir.glob("*.npz")):
        d = np.load(fp)
        y = d["y_frames"].astype(np.int64)
        y = lut[np.clip(y, 0, 6)]
        seqs.append((d["x"].astype(np.float32), y, d["t"].astype(np.float64)))
    return seqs


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    f1s, present = [], []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if (tp + fn) == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        present.append(c)
    return float(np.mean(f1s)), dict(zip([CUE_CLASSES[c] for c in present],
                                         [round(v, 4) for v in f1s]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/attention_temporal.yaml")
    ap.add_argument("--ckpt", default="work_dirs/attention_temporal_v2/checkpoints/best.pth")
    ap.add_argument("--val-dir", default="../grounding_data/llmstu_sequences/val")
    ap.add_argument("--out", default="work_dirs/attention_temporal_v2/baseline_chain_eval.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_classes = int(cfg["model"]["num_classes"])
    seqs = load_sequences(Path(args.val_dir))
    print(f"val sequences: {len(seqs)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AttentionTransformer(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=n_classes,
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        per_frame=True,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    all_true, all_pred = [], []
    per_seq_pred_cues = []
    with torch.no_grad():
        for x, y, t in seqs:
            xt = torch.from_numpy(x)[None].to(device)
            logits = model(xt)[0]  # [T, C]
            pred = logits.argmax(-1).cpu().numpy()
            per_seq_pred_cues.append(pred)
            all_true.append(y)
            all_pred.append(pred)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    results = {"n_sequences": len(seqs), "n_frames": int(y_true.size),
               "note": ("Rule/event comparisons are measured against "
                        "pseudo-label-derived cue timelines, not human ground truth.")}

    # 1. majority baseline
    maj = int(Counter(y_true.tolist()).most_common(1)[0][0])
    maj_pred = np.full_like(y_true, maj)
    mf1, _ = macro_f1(y_true, maj_pred, n_classes)
    results["majority"] = {"class": CUE_CLASSES[maj],
                           "accuracy": float((maj_pred == y_true).mean()),
                           "macro_f1": round(mf1, 4)}

    # 2. transformer
    mf1, per_cls = macro_f1(y_true, y_pred, n_classes)
    results["transformer"] = {"accuracy": float((y_pred == y_true).mean()),
                              "macro_f1": round(mf1, 4), "per_class_f1": per_cls}

    # 3. rule/event layer on label vs model cue timelines
    rcfg = RuleConfig.from_dict(cfg.get("rule_baseline", {}) or {})
    ecfg = EventConfig.from_dict(cfg.get("events", {}) or {})
    agree = tot = 0
    state_counts_lab, state_counts_mod = Counter(), Counter()
    eps_lab, eps_mod = [], []
    offset, total_dur = 0.0, 0.0
    for (x, y, t), pred in zip(seqs, per_seq_pred_cues):
        tt = t - t[0]
        s_lab = classify_timeline(tt, y.tolist(), rcfg)
        s_mod = classify_timeline(tt, pred.tolist(), rcfg)
        agree += sum(a == b for a, b in zip(s_lab, s_mod))
        tot += len(s_lab)
        state_counts_lab.update(s_lab)
        state_counts_mod.update(s_mod)
        shifted = tt + offset
        eps_lab += segment_events(shifted, y.tolist(), ecfg)
        eps_mod += segment_events(shifted, pred.tolist(), ecfg)
        dur = float(tt[-1]) if len(tt) > 1 else 1.0
        total_dur += dur
        offset += dur + 1000.0  # disjoint timelines: no cross-sequence matching
    results["rule_layer"] = {
        "window_state_agreement": round(agree / tot, 4),
        "state_distribution_labels": {s: state_counts_lab.get(s, 0) for s in STATES},
        "state_distribution_model": {s: state_counts_mod.get(s, 0) for s in STATES},
    }
    ev = evaluate_events(eps_mod, eps_lab, observed_duration_s=total_dur)
    results["event_layer_model_vs_pseudolabel"] = {
        ch: {k: (round(v, 3) if v == v else None) for k, v in st.items()}
        for ch, st in ev.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
