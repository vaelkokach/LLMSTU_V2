from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from mmdet.apis.inference import inference_detector, init_detector


@dataclass
class DetectionResult:
    bbox_xyxy: List[float]
    score: float
    label: int


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    w, h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class FrozenLLMDetAdapter:
    """
    Frozen student detector wrapper around trained LLMDet checkpoint.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompt: str = "student",
        score_thr: float = 0.35,
        device: str = "cuda:0",
        max_det: int = 100,
        min_rel_area: float = 0.01,
        max_rel_area: float = 0.60,
        min_aspect_ratio: float = 0.22,
        max_aspect_ratio: float = 1.25,
        nms_iou_thr: float = 0.5,
    ):
        cfg_path = Path(config_path)
        ckpt_path = Path(checkpoint_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.text_prompt = text_prompt
        self.score_thr = score_thr
        self.max_det = max_det
        self.min_rel_area = min_rel_area
        self.max_rel_area = max_rel_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.nms_iou_thr = nms_iou_thr
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = init_detector(str(cfg_path), str(ckpt_path), device=self.device)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectionResult]:
        frame_h, frame_w = frame_bgr.shape[:2]
        out = inference_detector(
            self.model,
            frame_bgr,
            text_prompt=self.text_prompt,
            custom_entities=True,
        )
        pred = out.pred_instances
        if pred is None or len(pred) == 0:
            return []

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()

        keep = scores >= self.score_thr
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if scores.size == 0:
            return []

        # Student-only geometric filtering to suppress desk/object false positives.
        geom_keep = []
        frame_area = float(frame_w * frame_h) + 1e-6
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box.tolist()
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            rel_area = (bw * bh) / frame_area
            ar = bw / bh
            if rel_area < self.min_rel_area or rel_area > self.max_rel_area:
                continue
            if ar < self.min_aspect_ratio or ar > self.max_aspect_ratio:
                continue
            geom_keep.append(i)
        if not geom_keep:
            return []
        bboxes = bboxes[geom_keep]
        scores = scores[geom_keep]
        labels = labels[geom_keep]

        keep_nms = self._nms(bboxes, scores, self.nms_iou_thr)
        bboxes = bboxes[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]
        order = np.argsort(-scores)[: self.max_det]
        results: List[DetectionResult] = []
        for i in order:
            results.append(
                DetectionResult(
                    bbox_xyxy=[float(x) for x in bboxes[i].tolist()],
                    score=float(scores[i]),
                    label=int(labels[i]),
                )
            )
        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep


class MMDetDetectorAdapter:
    """
    Generic detector wrapper for primary person detection or verification.
    Supports both closed-set MMDet and text-prompted MM-GroundingDINO models.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        score_thr: float = 0.3,
        device: str = "cuda:0",
        max_det: int = 100,
        text_prompt: Optional[str] = None,
        class_whitelist: Optional[List[int]] = None,
    ):
        cfg_path = Path(config_path)
        ckpt_path = Path(checkpoint_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.score_thr = score_thr
        self.max_det = max_det
        self.text_prompt = text_prompt
        self.class_whitelist = set(class_whitelist or [])
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = init_detector(str(cfg_path), str(ckpt_path), device=self.device)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectionResult]:
        if self.text_prompt:
            out = inference_detector(
                self.model,
                frame_bgr,
                text_prompt=self.text_prompt,
                custom_entities=True,
            )
        else:
            out = inference_detector(self.model, frame_bgr)

        pred = out.pred_instances
        if pred is None or len(pred) == 0:
            return []

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()

        keep = scores >= self.score_thr
        if self.class_whitelist:
            keep = keep & np.isin(labels, np.array(list(self.class_whitelist), dtype=labels.dtype))
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if scores.size == 0:
            return []

        order = np.argsort(-scores)[: self.max_det]
        return [
            DetectionResult(
                bbox_xyxy=[float(x) for x in bboxes[i].tolist()],
                score=float(scores[i]),
                label=int(labels[i]),
            )
            for i in order
        ]

