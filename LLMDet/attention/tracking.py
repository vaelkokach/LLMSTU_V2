from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from attention.detector_adapter import DetectionResult


@dataclass
class Track:
    track_id: int
    bbox_xyxy: List[float]
    score: float


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class IoUTracker:
    """
    Robust online tracker with IoU + appearance association.
    """

    def __init__(
        self,
        iou_match_thr: float = 0.35,
        max_age: int = 30,
        min_hits: int = 3,
        appearance_weight: float = 0.35,
        min_match_score: float = 0.25,
    ):
        self.iou_match_thr = iou_match_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.appearance_weight = appearance_weight
        self.min_match_score = min_match_score
        self.next_id = 1
        self.tracks: Dict[int, List[float]] = {}
        self.ages: Dict[int, int] = {}
        self.hits: Dict[int, int] = {}
        self.track_feat: Dict[int, np.ndarray] = {}

    @staticmethod
    def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        da = float(np.linalg.norm(a))
        db = float(np.linalg.norm(b))
        if da <= 1e-6 or db <= 1e-6:
            return 0.0
        return float(np.dot(a, b) / (da * db))

    def update(self, detections: List[DetectionResult], det_features: Optional[List[np.ndarray]] = None) -> List[Track]:
        det_used = set()
        matched: List[Tuple[int, int]] = []
        active_ids = list(self.tracks.keys())
        if det_features is None:
            det_features = [None for _ in detections]

        for tid in active_ids:
            best_det = -1
            best_score = -1.0
            for di, det in enumerate(detections):
                if di in det_used:
                    continue
                ov = iou_xyxy(self.tracks[tid], det.bbox_xyxy)
                if ov < self.iou_match_thr * 0.5:
                    continue
                app = self._cosine_sim(self.track_feat.get(tid), det_features[di])
                match_score = (1.0 - self.appearance_weight) * ov + self.appearance_weight * max(0.0, app)
                if match_score > best_score:
                    best_score = match_score
                    best_det = di
            if best_det >= 0 and best_score >= self.min_match_score:
                matched.append((tid, best_det))
                det_used.add(best_det)

        for tid, di in matched:
            self.tracks[tid] = detections[di].bbox_xyxy
            self.ages[tid] = 0
            self.hits[tid] = self.hits.get(tid, 0) + 1
            if det_features[di] is not None:
                prev = self.track_feat.get(tid, det_features[di])
                self.track_feat[tid] = 0.8 * prev + 0.2 * det_features[di]

        for di, det in enumerate(detections):
            if di in det_used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = det.bbox_xyxy
            self.ages[tid] = 0
            self.hits[tid] = 1
            if det_features[di] is not None:
                self.track_feat[tid] = det_features[di]

        remove = []
        matched_ids = {tid for tid, _ in matched}
        for tid in self.tracks:
            if tid not in matched_ids:
                self.ages[tid] = self.ages.get(tid, 0) + 1
            if self.ages[tid] > self.max_age:
                remove.append(tid)
        for tid in remove:
            self.tracks.pop(tid, None)
            self.ages.pop(tid, None)
            self.hits.pop(tid, None)
            self.track_feat.pop(tid, None)

        output = []
        for tid, bbox in self.tracks.items():
            if self.hits.get(tid, 0) < self.min_hits:
                continue
            output.append(Track(track_id=tid, bbox_xyxy=bbox, score=1.0))
        return output

    def coast(self) -> List[Track]:
        """Confirmed tracks at their last known boxes, **without ageing them**.

        Used on frames where the detector was skipped. Calling ``update([])``
        would also work but would age every track, and ``max_age`` is meant to
        count frames where we looked and found nothing — not frames where we
        never looked. At a detector stride of 5 that distinction is the
        difference between a track surviving a 45-frame gap and surviving a
        9-frame one.

        Boxes are held fixed rather than extrapolated: measured over 115 LLMSTU
        tracks, a seated student's centre jitters by 4.4% of their box diagonal
        (median; 13% at p90), so constant-position is a better model than
        constant-velocity here.
        """
        return [Track(track_id=tid, bbox_xyxy=bbox, score=1.0)
                for tid, bbox in self.tracks.items()
                if self.hits.get(tid, 0) >= self.min_hits]

