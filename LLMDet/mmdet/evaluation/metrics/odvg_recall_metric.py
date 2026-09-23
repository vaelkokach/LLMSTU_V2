# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from ..functional import bbox_overlaps


@METRICS.register_module()
class ODVGRecallMetric(BaseMetric):
    """Recall@K for OD/VG-style datasets with label indices per phrase.

    This metric evaluates whether each GT box is recovered by the top-K
    predictions of the same label (phrase index) at a given IoU threshold.
    """

    default_prefix: Optional[str] = 'odvg'

    def __init__(
        self,
        topk: Sequence[int] = (1, 5, 10),
        iou_thrs: float = 0.5,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topk = tuple(topk)
        self.iou_thrs = iou_thrs

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            dataset_mode = None
            if hasattr(data_sample, 'metainfo'):
                dataset_mode = data_sample.metainfo.get('dataset_mode')
            if dataset_mode is not None and dataset_mode != 'VG':
                continue
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            gt_bboxes = gt['bboxes'].cpu().numpy()
            gt_labels = gt['labels'].cpu().numpy()

            self.results.append(
                (pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels)
            )

    def _recall_for_sample(
        self,
        pred_bboxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_bboxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Tuple[Dict[int, int], int]:
        hits_by_k = {k: 0 for k in self.topk}
        total_gts = len(gt_bboxes)

        if total_gts == 0:
            return hits_by_k, 0

        for gt_box, gt_label in zip(gt_bboxes, gt_labels):
            label_mask = pred_labels == gt_label
            if not np.any(label_mask):
                continue
            label_bboxes = pred_bboxes[label_mask]
            label_scores = pred_scores[label_mask]

            order = np.argsort(-label_scores)
            label_bboxes = label_bboxes[order]

            ious = bbox_overlaps(
                label_bboxes, np.array(gt_box, dtype=np.float32).reshape(1, 4)
            ).reshape(-1)

            for k in self.topk:
                k = int(k)
                if k <= 0:
                    continue
                topk_ious = ious[:k] if len(ious) >= k else ious
                if topk_ious.size and topk_ious.max() >= self.iou_thrs:
                    hits_by_k[k] += 1

        return hits_by_k, total_gts

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        total_hits = {k: 0 for k in self.topk}
        total_gts = 0

        for pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels in results:
            hits_by_k, gt_count = self._recall_for_sample(
                pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels
            )
            total_gts += gt_count
            for k in self.topk:
                total_hits[k] += hits_by_k[k]

        metrics = {}
        for k in self.topk:
            recall = 0.0 if total_gts == 0 else total_hits[k] / total_gts
            metrics[f'recall@{k}'] = recall

        logger.info(metrics)
        return metrics