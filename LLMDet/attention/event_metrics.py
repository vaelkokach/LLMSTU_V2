"""Event-level evaluation: predicted vs ground-truth episodes.

Matching is per channel, greedy on temporal IoU (highest IoU pair first,
one-to-one), with a configurable IoU threshold. Reported metrics per channel
and micro-averaged overall:

- ``onset_error_s``       mean |pred.t_start - gt.t_start| over matched pairs
- ``duration_error_s``    mean |pred.duration - gt.duration| over matched pairs
- ``alert_delay_s``       mean max(0, pred.t_start - gt.t_start) (late alerts)
- ``missed_event_rate``   unmatched GT / total GT
- ``false_alerts_per_hour``  unmatched predictions / observed hours
- ``num_gt`` / ``num_pred`` / ``num_matched``
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from attention.events import Episode


def temporal_iou(a: Episode, b: Episode) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    if union <= 0:
        # both zero-length markers: match iff simultaneous
        return 1.0 if abs(a.t_start - b.t_start) < 1e-9 else 0.0
    return inter / union


def match_episodes(
    pred: Sequence[Episode], gt: Sequence[Episode], iou_thr: float = 0.3
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU. Returns (pred_idx, gt_idx, iou)."""
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p.channel != g.channel:
                continue
            iou = temporal_iou(p, g)
            if iou >= iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    matches = []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, iou))
    return matches


def evaluate_events(
    pred: Sequence[Episode],
    gt: Sequence[Episode],
    observed_duration_s: float,
    iou_thr: float = 0.3,
) -> Dict[str, Dict[str, float]]:
    """Per-channel + overall event metrics. ``observed_duration_s`` is the
    total observed timeline duration (for false-alerts-per-hour)."""
    if observed_duration_s <= 0:
        raise ValueError("observed_duration_s must be positive")
    channels = sorted({e.channel for e in list(pred) + list(gt)})
    out: Dict[str, Dict[str, float]] = {}
    hours = observed_duration_s / 3600.0

    def _stats(p: List[Episode], g: List[Episode]) -> Dict[str, float]:
        matches = match_episodes(p, g, iou_thr)
        onset = [abs(p[i].t_start - g[j].t_start) for i, j, _ in matches]
        dur = [abs(p[i].duration - g[j].duration) for i, j, _ in matches]
        delay = [max(0.0, p[i].t_start - g[j].t_start) for i, j, _ in matches]
        n_missed = len(g) - len(matches)
        n_false = len(p) - len(matches)
        return {
            "num_gt": float(len(g)),
            "num_pred": float(len(p)),
            "num_matched": float(len(matches)),
            "onset_error_s": float(np.mean(onset)) if onset else float("nan"),
            "duration_error_s": float(np.mean(dur)) if dur else float("nan"),
            "alert_delay_s": float(np.mean(delay)) if delay else float("nan"),
            "missed_event_rate": n_missed / len(g) if g else float("nan"),
            "false_alerts_per_hour": n_false / hours,
        }

    for ch in channels:
        out[ch] = _stats([e for e in pred if e.channel == ch], [e for e in gt if e.channel == ch])
    out["overall"] = _stats(list(pred), list(gt))
    return out
