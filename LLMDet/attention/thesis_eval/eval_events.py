"""Temporal-event evaluation of any number of systems against the human gold.

Scores every system against the *same* 984-crop dense human-gold segments, with
the detector and tracker excluded by design (track identity comes from the
manifest), so the number isolates the cue + event layers.

What this adds over ``attention/eval_events_vs_human.py``:

* the standard action-segmentation family — segmental F1@10/25/50 and edit
  score — which is what the temporal literature reports and what exposes
  over-segmentation that frame accuracy hides;
* event precision / recall / F1 at several tIoU thresholds instead of a single
  match count, separating under-firing from false alerting;
* onset **and offset** MAE (previously only onset and duration), plus detection
  delay and false alerts per hour;
* boundary errors on the **common matched subset** of gold events, so a
  higher-recall model is not charged for the harder episodes only it found;
* de-duplicated gold. ``attention.events`` aliases ``inactivity`` onto
  ``head_down``, and the gold carries one ``return_to_task`` marker twice, so
  the historic "24 episodes" is 16 distinct ones. Both are reported.
* one shared feature cache, so all systems see byte-identical inputs.

Usage (from LLMDet/):
    python -m attention.thesis_eval.eval_events \
        --cache ../grounding_data/llmstu_tools/outputs/gold_event_features.npz \
        --gold-events ../grounding_data/llmstu_tools/outputs/gold_events.jsonl \
        --ckpt name=work_dirs/thesis/ladder/transformer_556_hp_s42/checkpoints/best.pth \
        --ckpt name2=... --out work_dirs/thesis/events/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from attention.events import Episode, EventConfig, segment_events
from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import data as D
from attention.thesis_eval import metrics as M
from attention.thesis_eval import segmentation as S
from attention.thesis_eval.models import build_model


def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["track_keys"]]
    return {k: {"x": z[f"x::{k}"], "t": z[f"t::{k}"],
                "teacher": z[f"teacher::{k}"], "human": z[f"human::{k}"]}
            for k in keys}


def load_gold(path: Path) -> Dict:
    by = defaultdict(list)
    for line in open(path):
        g = json.loads(line)
        by[f"{g['video_id']}|{g['seat_id']}"].append(
            Episode(g["channel"], float(g["t_start"]), float(g["t_end"])))
    return dict(by)


@torch.no_grad()
def predict_tracks(ckpt_path: Path, cache: Dict, device) -> Dict[str, np.ndarray]:
    ck = torch.load(ckpt_path, map_location="cpu")
    spec = ck["spec"]
    cols = D.column_index(spec["feature_config"])
    kw = dict(spec.get("model_kwargs") or {})
    if spec["model"] == "transformer":
        kw.setdefault("dropout", spec.get("dropout", 0.1))
    model = build_model(spec["model"], len(cols), len(CUE_CLASSES), **kw).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    out = {}
    for k, tr in cache.items():
        x = torch.from_numpy(tr["x"][:, cols]).float()[None].to(device)
        res = model(x)
        if "boundary" in res:
            pred = model.refine(res["logits"].float(), res["boundary"].float(),
                                torch.tensor([x.shape[1]]))[0]
        else:
            pred = res["logits"][0].argmax(-1)
        pred = pred.cpu().numpy()
        if pred.shape[0] != tr["x"].shape[0]:
            raise RuntimeError(f"{ckpt_path}: {pred.shape[0]} preds for "
                               f"{tr['x'].shape[0]} frames on track {k}")
        out[k] = pred
    return out, spec


def evaluate_system(name: str, cues_by_track: Dict[str, np.ndarray], cache: Dict,
                    gold: Dict, ecfg: EventConfig, iou_thresholds, primary_iou,
                    dedup: bool) -> Dict:
    """Frame + segmental + event metrics for one predicted cue timeline."""
    # --- frame level, restricted to human-verified frames ---
    fy, fp = [], []
    seg_gt, seg_pred = {}, {}
    for k, tr in cache.items():
        hum = tr["human"]
        m = hum >= 0
        fy.append(hum[m])
        fp.append(cues_by_track[k][m])
        if m.sum() > 1:
            # segmentation metrics run on the human-verified subsequence; the
            # rejected frames are gaps the annotator could not verify, so no
            # label exists to score against there.
            seg_gt[k] = hum[m].tolist()
            seg_pred[k] = cues_by_track[k][m].tolist()
    y = np.concatenate(fy)
    p = np.concatenate(fp)
    cm = M.confusion_matrix(y, p)

    # --- events ---
    per_track_pred, per_track_gold = {}, {}
    durations = []
    for k, tr in cache.items():
        t = tr["t"] - tr["t"][0]
        eps = segment_events(t, cues_by_track[k].tolist(), ecfg)
        per_track_pred[k] = S.dedup_episodes(eps) if dedup else eps
        g = gold.get(k, [])
        t0 = tr["t"][0]
        g = [Episode(e.channel, e.t_start - t0, e.t_end - t0) for e in g]
        per_track_gold[k] = S.dedup_episodes(g) if dedup else g
        durations.append(float(t[-1]) if len(t) > 1 else 1.0)
    span = max(durations) if durations else 1.0
    total_observed = float(sum(durations))
    pred_flat = S.offset_flatten(per_track_pred, span)
    gold_flat = S.offset_flatten(per_track_gold, span)

    ev = S.event_metrics(pred_flat, gold_flat, total_observed,
                         iou_thresholds=iou_thresholds, primary_iou=primary_iou)
    seg = S.segmentation_metrics_over_tracks(seg_gt, seg_pred, (0.1, 0.25, 0.5))

    return {
        "system": name,
        "frame": {
            "n_frames": int(len(y)),
            "accuracy": M.accuracy(cm),
            "balanced_accuracy": M.balanced_accuracy(cm),
            "macro_f1": M.macro_f1(cm),
            "per_class_f1": {CUE_CLASSES[c]: float(v) for c, v in
                             enumerate(M.per_class_prf(cm)["f1"])},
            "confusion_matrix": cm.tolist(),
        },
        "segmentation": seg,
        "events": ev,
        "_episodes": pred_flat,
        "_gold": gold_flat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gold-events", required=True)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="NAME=path/to/checkpoint.pth (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--primary-iou", type=float, default=0.3)
    ap.add_argument("--tiou", default="0.1,0.25,0.5")
    ap.add_argument("--min-duration-s", type=float, default=3.0)
    ap.add_argument("--max-gap-s", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = load_cache(Path(args.cache))
    gold = load_gold(Path(args.gold_events))
    ecfg = EventConfig(min_duration_s=args.min_duration_s, max_gap_s=args.max_gap_s,
                       per_channel_min_duration={"phone_use": 2.0, "head_down": 5.0})
    tious = tuple(float(v) for v in args.tiou.split(","))

    systems: Dict[str, Dict[str, np.ndarray]] = {}
    specs: Dict[str, Dict] = {}
    for entry in args.ckpt:
        name, path = entry.split("=", 1)
        cues, spec = predict_tracks(Path(path), cache, device)
        systems[name] = cues
        specs[name] = {"checkpoint": path, "model": spec["model"],
                       "feature_config": spec["feature_config"], "seed": spec.get("seed")}

    # reference systems on identical frames
    systems["teacher"] = {k: tr["teacher"] for k, tr in cache.items()}
    specs["teacher"] = {"note": "Qwen3.5-VL pseudo-labels — empirical teacher "
                                "benchmark, NOT a theoretical ceiling"}
    major = Counter(int(v) for tr in cache.values() for v in tr["teacher"]).most_common(1)[0][0]
    systems["majority"] = {k: np.full_like(tr["teacher"], major) for k, tr in cache.items()}
    specs["majority"] = {"note": f"constant {CUE_CLASSES[major]} control"}

    out: Dict = {"evaluator_version": EVALUATOR_VERSION,
                 "cache": str(args.cache), "gold_events": str(args.gold_events),
                 "primary_iou": args.primary_iou, "tiou_thresholds": list(tious),
                 "event_config": {"min_duration_s": args.min_duration_s,
                                  "max_gap_s": args.max_gap_s},
                 "n_tracks": len(cache), "systems": specs, "results": {}}

    # After the 2026-08-01 removal of the `inactivity` alias from
    # attention/events.py, predictions no longer emit that channel while gold
    # files written earlier still contain it. `gold_raw` is then ASYMMETRIC —
    # gold has 7 episodes no system can possibly match — and its recall is
    # meaningless. Detect that rather than let it read as a regression.
    from attention.events import EVENT_CHANNELS, LEGACY_ALIAS_CHANNELS
    gold_channels = {e.channel for eps in gold.values() for e in eps}
    stale_alias = sorted(gold_channels & set(LEGACY_ALIAS_CHANNELS)
                         - set(EVENT_CHANNELS))

    for dedup in (False, True):
        tag = "gold_dedup" if dedup else "gold_raw"
        res = {name: evaluate_system(name, cues, cache, gold, ecfg, tious,
                                     args.primary_iou, dedup)
               for name, cues in systems.items()}
        # common matched subset across the model systems (references excluded:
        # including the majority control, which matches nothing, would empty it)
        model_names = [n for n in res if n not in ("majority",)]
        common = S.common_subset_boundaries(
            {n: res[n]["_episodes"] for n in model_names},
            res[model_names[0]]["_gold"], args.primary_iou)
        for r in res.values():
            r.pop("_episodes", None)
            r.pop("_gold", None)
        block = {"n_gold_events": sum(
            len(v) for v in ({k: S.dedup_episodes(v) for k, v in gold.items()}
                             if dedup else gold).values()),
            "per_system": res, "common_matched_subset": common}
        if not dedup and stale_alias:
            block["VALID"] = False
            block["invalid_reason"] = (
                f"ASYMMETRIC — the gold file still contains the retired alias "
                f"channel(s) {stale_alias}, which attention/events.py no longer "
                f"emits, so those gold episodes are unmatchable by construction "
                f"and recall here is meaningless. Use gold_dedup. This block is "
                f"retained only to explain the difference from pre-2026-08-01 "
                f"numbers.")
        else:
            block["VALID"] = True
        out["results"][tag] = block

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    for tag in ("gold_raw", "gold_dedup"):
        block = out["results"][tag]
        print(f"\n=== {tag}: {block['n_gold_events']} gold episodes, "
              f"{len(cache)} tracks ===")
        if not block.get("VALID", True):
            print(f"    !! NOT VALID: {block['invalid_reason']}")
        print(f"{'system':22s} {'frameAcc':>8} {'macroF1':>8} {'F1@10':>7} {'F1@25':>7} "
              f"{'F1@50':>7} {'edit':>6} {'ev_rec':>7} {'ev_prec':>8} {'ev_F1':>6} {'FA/h':>6}")
        for name, r in block["per_system"].items():
            e = r["events"]["by_tiou"][f"{args.primary_iou:.2f}"]
            s = r["segmentation"]
            print(f"{name:22s} {r['frame']['accuracy']:8.3f} {r['frame']['macro_f1']:8.3f} "
                  f"{s['f1@10']:7.3f} {s['f1@25']:7.3f} {s['f1@50']:7.3f} "
                  f"{s['edit_score']:6.1f} {e['recall']:7.3f} {e['precision']:8.3f} "
                  f"{e['f1']:6.3f} {e['false_alerts_per_hour']:6.2f}")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
