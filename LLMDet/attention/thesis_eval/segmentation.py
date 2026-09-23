"""Temporal-action-segmentation and event metrics.

Implements the standard family used by the action-segmentation literature
(MS-TCN / ASRF / ASFormer report exactly these), computed on the project's own
cue timelines:

    segmental F1@{10,25,50}   overlap-thresholded segment F1
    segmental edit score      normalised Levenshtein over the segment label
                              sequence — the fragmentation / over-segmentation
                              penalty that frame accuracy is blind to
    event P / R / F1 @ tIoU   one-to-one matched episode detection
    onset / offset / duration MAE, detection delay, false alerts per hour

Two project-specific corrections are made here and documented rather than
inherited silently:

**1. ``inactivity`` duplicates ``head_down``.** ``attention.events``
maps both channels to the single cue ``head_down``, so every head-down episode
is emitted twice. The human gold therefore contains 7 ``head_down`` + 7
identical ``inactivity`` episodes among its "24". Reported both ways:
``dedup=False`` reproduces the historic counts, ``dedup=True`` drops the
duplicate channel and is the honest denominator.

**2. Duplicate zero-length ``return_to_task`` markers.** The gold carries the
marker at t=51.9 twice on one track. Greedy matching is one-to-one, so one of
the pair is *unmatchable by construction* and scores as a guaranteed miss.
De-duplication removes it.

Boundary errors are additionally reported over the **common matched subset** —
the gold events that *both* compared systems matched. Without this, a
higher-recall model is penalised for the extra, harder episodes it alone found,
which is exactly how the 556 model's onset error appeared to explode from 1.6 s
to 10.4 s [internal notes, not included] note 3).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from attention.events import Episode

#: ``inactivity`` is an exact alias of ``head_down`` in ``attention.events``.
DUPLICATE_CHANNELS = ("inactivity",)


# --------------------------------------------------------------------------
# segment extraction
# --------------------------------------------------------------------------

def label_segments(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Run-length encode a frame label sequence into (label, start, end_excl)."""
    labels = list(labels)
    if not labels:
        return []
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i))
            start = i
    return out


def segment_iou(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Frame-index IoU of two (label, start, end_excl) segments."""
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    union = max(a[2], b[2]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def segmental_f1(gt: Sequence[int], pred: Sequence[int], overlap: float,
                 background: Optional[int] = None) -> Dict[str, float]:
    """Segment-level F1 at an IoU threshold (Lea et al., the MS-TCN protocol).

    Each predicted segment matches at most one ground-truth segment of the same
    class with IoU >= ``overlap``; greedy in prediction order, which is the
    reference implementation's behaviour.
    """
    g = [s for s in label_segments(gt) if background is None or s[0] != background]
    p = [s for s in label_segments(pred) if background is None or s[0] != background]
    used: Set[int] = set()
    tp = 0
    for ps in p:
        best, best_iou = -1, 0.0
        for j, gs in enumerate(g):
            if j in used or gs[0] != ps[0]:
                continue
            iou = segment_iou(ps, gs)
            if iou > best_iou:
                best, best_iou = j, iou
        if best >= 0 and best_iou >= overlap:
            used.add(best)
            tp += 1
    fp = len(p) - tp
    fn = len(g) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "n_gt_segments": len(g), "n_pred_segments": len(p)}


def edit_score(gt: Sequence[int], pred: Sequence[int],
               background: Optional[int] = None) -> float:
    """Normalised segmental edit score in [0, 100].

    Levenshtein distance between the two *segment label* sequences, normalised
    by the longer one. Penalises fragmentation, which frame accuracy rewards.
    """
    g = [s[0] for s in label_segments(gt) if background is None or s[0] != background]
    p = [s[0] for s in label_segments(pred) if background is None or s[0] != background]
    n, m = len(p), len(g)
    if max(n, m) == 0:
        return 100.0
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if p[i - 1] == g[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float((1.0 - d[n, m] / max(n, m)) * 100.0)


# --------------------------------------------------------------------------
# episode-level detection
# --------------------------------------------------------------------------

def temporal_iou(a: Episode, b: Episode) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    if union <= 0:  # two zero-length markers match iff simultaneous
        return 1.0 if abs(a.t_start - b.t_start) < 1e-9 else 0.0
    return inter / union


def dedup_episodes(eps: Iterable[Episode],
                   drop_channels: Sequence[str] = DUPLICATE_CHANNELS) -> List[Episode]:
    """Drop alias channels and exact duplicate episodes."""
    seen: Set[Tuple[str, float, float]] = set()
    out = []
    for e in eps:
        if e.channel in drop_channels:
            continue
        k = (e.channel, round(e.t_start, 6), round(e.t_end, 6))
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def match_episodes(pred: Sequence[Episode], gt: Sequence[Episode],
                   iou_thr: float) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by descending tIoU, within channel.

    Ties are broken by (pred index, gt index) so the result does not depend on
    input ordering noise — the historic implementation sorted raw tuples, which
    made the tie-break depend on the float IoU value alone.
    """
    cand = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p.channel != g.channel:
                continue
            iou = temporal_iou(p, g)
            if iou >= iou_thr:
                cand.append((-iou, i, j))
    cand.sort()
    used_p: Set[int] = set()
    used_g: Set[int] = set()
    out = []
    for neg_iou, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, -neg_iou))
    return out


def _boundary_stats(pred: Sequence[Episode], gt: Sequence[Episode],
                    matches: Sequence[Tuple[int, int, float]],
                    restrict_gt: Optional[Set[int]] = None) -> Dict[str, float]:
    onset, offset, dur, delay = [], [], [], []
    for i, j, _ in matches:
        if restrict_gt is not None and j not in restrict_gt:
            continue
        onset.append(abs(pred[i].t_start - gt[j].t_start))
        offset.append(abs(pred[i].t_end - gt[j].t_end))
        dur.append(abs(pred[i].duration - gt[j].duration))
        delay.append(max(0.0, pred[i].t_start - gt[j].t_start))
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return {"n": len(onset), "onset_mae_s": f(onset), "offset_mae_s": f(offset),
            "duration_mae_s": f(dur), "detection_delay_s": f(delay)}


def event_metrics(pred: Sequence[Episode], gt: Sequence[Episode],
                  observed_duration_s: float,
                  iou_thresholds: Sequence[float] = (0.1, 0.25, 0.5),
                  primary_iou: float = 0.3) -> Dict:
    """Episode detection metrics at several tIoU thresholds, plus per-channel
    breakdown and temporal AP at the primary threshold."""
    if observed_duration_s <= 0:
        raise ValueError("observed_duration_s must be positive")
    hours = observed_duration_s / 3600.0
    out: Dict = {"n_gt": len(gt), "n_pred": len(pred),
                 "observed_duration_s": observed_duration_s, "by_tiou": {}}

    thresholds = sorted(set(list(iou_thresholds) + [primary_iou]))
    for thr in thresholds:
        m = match_episodes(pred, gt, thr)
        tp, fp, fn = len(m), len(pred) - len(m), len(gt) - len(m)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        row = {
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "matched": tp, "missed": fn, "false_alerts": fp,
            "missed_event_rate": fn / len(gt) if gt else float("nan"),
            "false_alerts_per_hour": fp / hours,
            "matched_gt_indices": sorted(j for _, j, _ in m),
        }
        row.update(_boundary_stats(pred, gt, m))
        out["by_tiou"][f"{thr:.2f}"] = row

    out["mean_recall_over_tiou"] = float(np.mean(
        [out["by_tiou"][f"{t:.2f}"]["recall"] for t in thresholds]))
    out["mean_f1_over_tiou"] = float(np.mean(
        [out["by_tiou"][f"{t:.2f}"]["f1"] for t in thresholds]))

    channels = sorted({e.channel for e in list(pred) + list(gt)})
    out["by_channel"] = {}
    for ch in channels:
        p = [e for e in pred if e.channel == ch]
        g = [e for e in gt if e.channel == ch]
        m = match_episodes(p, g, primary_iou)
        row = {"n_gt": len(g), "n_pred": len(p), "matched": len(m),
               "missed_event_rate": (len(g) - len(m)) / len(g) if g else float("nan"),
               "false_alerts_per_hour": (len(p) - len(m)) / hours}
        row.update(_boundary_stats(p, g, m))
        out["by_channel"][ch] = row
    out["primary_iou"] = primary_iou
    return out


def common_subset_boundaries(
    systems: Dict[str, Sequence[Episode]], gt: Sequence[Episode],
    iou_thr: float,
) -> Dict:
    """Boundary errors restricted to gold events matched by **every** system.

    This is the like-for-like comparison: a model that finds four extra, harder
    episodes must not be charged for their looser boundaries when the baseline
    never found them at all.
    """
    matches = {k: match_episodes(v, gt, iou_thr) for k, v in systems.items()}
    common: Optional[Set[int]] = None
    for m in matches.values():
        s = {j for _, j, _ in m}
        common = s if common is None else (common & s)
    common = common or set()
    return {
        "iou_thr": iou_thr,
        "n_common_gold_events": len(common),
        "common_gold_indices": sorted(common),
        "per_system": {
            k: _boundary_stats(systems[k], gt, matches[k], restrict_gt=common)
            for k in systems
        },
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

@dataclass
class TrackTimeline:
    key: Tuple[str, str]
    t: np.ndarray
    labels: np.ndarray


def offset_flatten(per_track: Dict, duration: float) -> List[Episode]:
    """Lay per-track episodes on disjoint slices of one timeline so that greedy
    global matching cannot pair student A's prediction with student B's gold."""
    flat: List[Episode] = []
    for i, (_, eps) in enumerate(sorted(per_track.items())):
        off = i * (duration + 1000.0)
        flat += [Episode(e.channel, e.t_start + off, e.t_end + off) for e in eps]
    return flat


def segmentation_metrics_over_tracks(
    gt_by_track: Dict, pred_by_track: Dict,
    overlaps: Sequence[float] = (0.1, 0.25, 0.5),
    background: Optional[int] = None,
) -> Dict:
    """Segmental F1 / edit score aggregated over tracks.

    F1 is micro-aggregated (tp/fp/fn summed across tracks, as MS-TCN does);
    edit score is macro-averaged over tracks, weighted by track length.
    """
    keys = sorted(set(gt_by_track) & set(pred_by_track))
    agg = {f"f1@{int(o * 100)}": {"tp": 0, "fp": 0, "fn": 0} for o in overlaps}
    edits, weights = [], []
    for k in keys:
        g, p = list(gt_by_track[k]), list(pred_by_track[k])
        if len(g) != len(p):
            raise ValueError(f"track {k}: {len(g)} gt frames vs {len(p)} pred frames")
        for o in overlaps:
            r = segmental_f1(g, p, o, background)
            a = agg[f"f1@{int(o * 100)}"]
            a["tp"] += r["tp"]; a["fp"] += r["fp"]; a["fn"] += r["fn"]
        edits.append(edit_score(g, p, background))
        weights.append(len(g))
    out: Dict = {"n_tracks": len(keys)}
    for name, a in agg.items():
        prec = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0.0
        rec = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0
        out[name] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name + "_detail"] = dict(a, precision=prec, recall=rec)
    out["edit_score"] = float(np.average(edits, weights=weights)) if edits else 0.0
    out["edit_score_unweighted"] = float(np.mean(edits)) if edits else 0.0
    return out
