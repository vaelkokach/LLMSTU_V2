"""Evaluation of matching strategies against exact LLMSTU correspondences.

Ground-truth interface (produced by grounding_data/llmstu_tools, W1):
    correspondence_gt.jsonl — one record per source frame:
    {"src_frame": "...jpg", "students": [
        {"bbox_person": [x1,y1,x2,y2], "det_conf": f, "person_idx": i,
         "crop_file": "shard_XXX/part_YYY/....jpg", "caption": "...",
         "labels": {"activity": ..., "gaze_direction": ..., ...}}, ...]}

If that file does not exist yet, `build_standin_gt` derives an equivalent one
directly from a LLMSTU label shard (the correspondence is exact either way);
swap in the real file by passing --gt when W1 lands.

Evaluation protocol ("reconstruction"): units are built from each student's
own LLMSTU caption, presented to the matcher in a chosen order
(--unit-order lr|conf) while boxes are presented in detector-confidence order
(the order the real pipeline sees, per the audit). The true unit<->box
bijection is known, so assignment accuracy is exact.

Metrics: assignment accuracy, wrong-assignment rate, per-crowding breakdown
(2 / 3 / 4+ students), and region-correctness (IoU>=0.5 of the assigned box
against the true box, which credits near-misses onto overlapping neighbours).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

LLMSTU_ROOT = Path("/home/jovyan/Computer_vision/grounding_data/LLMSTU")
STRUCT_KEYS = ["activity", "gaze_direction", "attention_target",
               "engagement_level", "posture", "hand_state",
               "phone_visible", "laptop_visible", "talking", "occluded"]


def build_standin_gt(shard_jsonl: Path, min_students: int = 2,
                     max_frames: Optional[int] = None) -> List[Dict]:
    """Group a LLMSTU label shard by src_frame into the GT interface."""
    frames = defaultdict(list)
    with open(shard_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            frames[rec["src_frame"]].append(rec)
    out = []
    for src, recs in frames.items():
        if len(recs) < min_students:
            continue
        recs.sort(key=lambda r: r["person_idx"])
        out.append({
            "src_frame": src,
            "students": [{
                "bbox_person": r["bbox_person"],
                "det_conf": r["det_conf"],
                "person_idx": r["person_idx"],
                "crop_file": r["file_name"],
                "caption": r["caption"],
                "labels": {k: r[k] for k in STRUCT_KEYS},
            } for r in recs],
        })
        if max_frames and len(out) >= max_frames:
            break
    return out


def _normalize_student(s: Dict, fallback_idx: int) -> Dict:
    """Map W1's correspondence_gt field names onto the canonical interface.

    W1 emits flat records with `caption_neutral`/`file_name`/`seat_id`;
    the stand-in builder emits `caption`/`crop_file`/`person_idx` with a
    nested `labels` dict. Accept both.
    """
    if "labels" in s and "caption" in s and "crop_file" in s:
        return s
    return {
        "bbox_person": s["bbox_person"],
        "det_conf": s["det_conf"],
        "person_idx": s.get("person_idx", s.get("seat_id", fallback_idx)),
        "crop_file": s.get("crop_file", s.get("file_name")),
        "caption": s.get("caption", s.get("caption_neutral")),
        "labels": s.get("labels", {k: s[k] for k in STRUCT_KEYS if k in s}),
    }


def load_gt(gt_path: Optional[Path], standin_shard: Path,
            min_students: int = 2,
            max_frames: Optional[int] = None) -> List[Dict]:
    if gt_path and gt_path.exists():
        records = []
        with open(gt_path) as fh:
            for line in fh:
                rec = json.loads(line)
                if len(rec["students"]) >= min_students:
                    rec["students"] = [_normalize_student(s, i)
                                       for i, s in enumerate(rec["students"])]
                    records.append(rec)
                    if max_frames and len(records) >= max_frames:
                        break
        return records
    return build_standin_gt(standin_shard, min_students, max_frames)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


class MatchingEvaluator:
    """Accumulates exact-assignment metrics across frames."""

    def __init__(self):
        self.per_frame = []

    def add_frame(self, pairs, true_box_of_unit: np.ndarray,
                  boxes: np.ndarray, n_students: int):
        """pairs: matcher output (unit_idx, box_idx) over presented order;
        true_box_of_unit[k] = presented box index truly belonging to unit k."""
        n = len(true_box_of_unit)
        assigned = {u: b for u, b in pairs}
        correct = sum(1 for u in range(n) if assigned.get(u) == true_box_of_unit[u])
        region_ok = 0
        for u in range(n):
            if u not in assigned:
                continue
            if iou(boxes[assigned[u]], boxes[true_box_of_unit[u]]) >= 0.5:
                region_ok += 1
        self.per_frame.append({
            "n": n_students,
            "correct": correct,
            "assigned": len(assigned),
            "region_ok": region_ok,
        })

    def summary(self) -> Dict:
        def agg(frames):
            total = sum(f["n"] for f in frames)
            correct = sum(f["correct"] for f in frames)
            assigned = sum(f["assigned"] for f in frames)
            region = sum(f["region_ok"] for f in frames)
            wrong = assigned - correct
            return {
                "frames": len(frames),
                "units": total,
                "assignment_acc": correct / total if total else float("nan"),
                "wrong_assignment_rate": wrong / assigned if assigned else float("nan"),
                "region_correct_iou50": region / total if total else float("nan"),
            }
        out = {"overall": agg(self.per_frame)}
        buckets = {"n=2": lambda n: n == 2, "n=3": lambda n: n == 3,
                   "n>=4": lambda n: n >= 4}
        for name, pred in buckets.items():
            out[name] = agg([f for f in self.per_frame if pred(f["n"])])
        return out


def format_results_table(results: Dict[str, Dict]) -> str:
    """results: {strategy: summary-dict from MatchingEvaluator}."""
    cols = ["overall", "n=2", "n=3", "n>=4"]
    lines = []
    header = f"{'strategy':<14}" + "".join(
        f"{c + ' acc':>12}" for c in cols) + f"{'wrong-rate':>12}{'IoU50':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for strat, res in results.items():
        row = f"{strat:<14}"
        for c in cols:
            row += f"{res[c]['assignment_acc']:>12.4f}"
        row += f"{res['overall']['wrong_assignment_rate']:>12.4f}"
        row += f"{res['overall']['region_correct_iou50']:>10.4f}"
        lines.append(row)
    return "\n".join(lines)
