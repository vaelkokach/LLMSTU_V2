# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ODVGPhraseEvalPrep(BaseTransform):
    """Align ODVG ``phrases`` dict with Flickr30k-style phrase grounding eval.

    Run **after** :class:`LoadTextAnnotations` (which needs the original
    ``phrases`` dict to build ``tokens_positive``). This transform replaces
    ``results['phrases']`` with an ordered list of phrase strings and sets
    ``phrase_ids`` to match ``gt_bboxes_labels`` for :class:`Flickr30kMetric`.
    """

    def transform(self, results: dict) -> dict:
        def _key_order(k):
            if isinstance(k, int):
                return k
            try:
                return int(k)
            except (TypeError, ValueError):
                return str(k)

        phrases = results.get('phrases')
        phrase_list = []
        if isinstance(phrases, dict) and phrases:
            keys = sorted(phrases.keys(), key=_key_order)
            for k in keys:
                p = phrases[k].get('phrase', '')
                if isinstance(p, list):
                    p = p[0] if p else ''
                phrase_list.append(str(p))
            results['phrases'] = phrase_list
        elif isinstance(phrases, list):
            results['phrases'] = [str(p) for p in phrases]
            phrase_list = results['phrases']
        else:
            results['phrases'] = []

        # Flickr30kMetric requires phrase_ids + phrases on the DetDataSample.
        # Always set phrase_ids from labels when present (do not return early
        # when phrases dict is empty — that skipped phrase_ids and caused KeyError).
        if 'gt_bboxes_labels' in results:
            results['phrase_ids'] = np.asarray(
                results['gt_bboxes_labels'], dtype=np.int64)
            # Metric indexes predictions by phrase id i; ensure len(phrases) > max id
            if len(results['gt_bboxes_labels']) > 0:
                need = int(results['gt_bboxes_labels'].max()) + 1
                pl = results['phrases']
                if len(pl) < need:
                    results['phrases'] = list(pl) + [''] * (need - len(pl))

        return results
