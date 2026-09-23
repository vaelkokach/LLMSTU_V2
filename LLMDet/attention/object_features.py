"""Per-student object-presence features, from a SECOND detector.

Why a second detector
---------------------
The deployed student detector **ignores its text prompt**. Fine-tuning on one
category collapsed the text conditioning, and measurably so: over 12 frames the
nonsense string ``qwertyuiop`` retrieves students as well as ``a student
sitting`` does, 92% of its boxes matching a student box at IoU >= 0.9
[internal notes, not included]. It cannot be re-prompted to find phones.

A *pretrained* GroundingDINO can. The same test on
``grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det`` returns **0**
detections for the nonsense prompt, and ``cell phone`` boxes an order of
magnitude smaller than person boxes with 0% overlap [internal notes, not included]. Obj365
contains both ``cell phone`` and ``laptop``.

Which features, and why only three
----------------------------------
Validated on 250 **matched pairs** -- two students from the *same source frame*,
one with ``phone_visible`` and one without, so room, camera, lighting and
detector behaviour are identical and the only difference is the student
[internal notes, not included]:

===================  =======  ==============================================
feature              AUROC    
===================  =======  ==============================================
``score * y_frac``   [value removed]    score weighted by how far down the box it sits
``y_frac``           [value removed]    a held phone is at hand/desk height, not head
``score_contained``  [value removed]    the detector's confidence, contained in the box
-------------------  -------  ----------------------------------------------
``rel_area``         [value removed]    no signal -- dropped
``n_contained``      [value removed]    INVERTED: counts how busy the desk is -- dropped
===================  =======  ==============================================

The two dropped features are dropped deliberately. A column that does not
discriminate still costs width, still gets a name, and still invites a story
about what the model "learned" from it.

An earlier version of this measurement gave ``y_frac`` an AUROC of [value removed], which
was an artefact of defaulting it to 0.0 when nothing was detected, on an
unmatched sample drawn from different rooms. ``y_frac`` is NaN-then-zero here for
the same reason it is reported as NaN there: absence is not position.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

#: The objects asked for, in the order their columns appear.
OBJECT_PROMPTS: Tuple[str, ...] = ("cell phone", "laptop")
#: Per object: score_contained, y_frac, score * y_frac.
DIMS_PER_OBJECT = 3
OBJECT_DIM = len(OBJECT_PROMPTS) * DIMS_PER_OBJECT      # 6, == the v1080 block

#: Below this the detector's output is noise at this resolution. Chosen as the
#: floor used in the validation, not tuned against the outcome.
MIN_SCORE = 0.05


def student_object_features(
    boxes_by_object: Sequence[Tuple[np.ndarray, np.ndarray]],
    person_box: Sequence[float],
) -> np.ndarray:
    """[OBJECT_DIM] for one student.

    ``boxes_by_object`` is one ``(boxes[n,4], scores[n])`` pair per entry of
    :data:`OBJECT_PROMPTS`, in that order, already thresholded.

    Containment is by the object's CENTRE inside the student's box, with no
    padding. The padded version was measured and does not work: a 15% pad around
    a seated student reaches their neighbour, and the feature then fired on 90.8%
    of students who had no phone [internal notes, not included].
    """
    px1, py1, px2, py2 = (float(v) for v in person_box)
    ph = max(py2 - py1, 1e-6)
    out = np.zeros(OBJECT_DIM, dtype=np.float32)

    for oi, (boxes, scores) in enumerate(boxes_by_object):
        best_s, best_y = 0.0, 0.0
        for b, s in zip(np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                        np.asarray(scores, dtype=np.float64).ravel()):
            if s < MIN_SCORE:
                continue
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            if not (px1 <= cx <= px2 and py1 <= cy <= py2):
                continue
            if s > best_s:
                best_s = float(s)
                # 0 at the top of the student's box, 1 at the bottom. Measured:
                # (measured value removed from the handover copy)
                # (measured value removed from the handover copy)
                best_y = float(np.clip((cy - py1) / ph, 0.0, 1.0))
        o = oi * DIMS_PER_OBJECT
        out[o] = best_s
        # y_frac is 0 when nothing was found, and that is the ONE place absence
        # and "at the very top" collide. They are distinguishable through
        # out[o] == 0, which is why the score column comes first and is never
        # (measured value removed from the handover copy)
        out[o + 1] = best_y
        out[o + 2] = best_s * best_y
    return out


class ObjectDetector:
    """A pretrained open-vocabulary detector, prompted for the objects.

    Kept separate from ``detector_adapter.FrozenLLMDetAdapter`` on purpose: that
    one applies student-shaped geometry filters (min_rel_area 0.01, aspect ratio
    0.20-1.20) which would reject a phone outright, and it wraps a checkpoint
    whose prompt does nothing.
    """

    def __init__(self, config_path: str, checkpoint_path: str,
                 device: str = "cuda:0",
                 prompts: Sequence[str] = OBJECT_PROMPTS,
                 min_score: float = MIN_SCORE):
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.prompts = tuple(prompts)
        self.min_score = float(min_score)
        self._model = None

    def _ensure(self):
        if self._model is None:
            from mmdet.apis import init_detector
            self._model = init_detector(self.config_path, self.checkpoint_path,
                                        device=self.device)

    def detect(self, frame_bgr: np.ndarray):
        """-> [(boxes, scores)] per prompt, in OBJECT_PROMPTS order.

        ONE forward pass for all prompts. GroundingDINO takes a multi-phrase
        caption (``"cell phone. laptop."``) and `pred_instances.labels` indexes
        the phrase each box matched, so asking separately costs a full pass per
        object for no extra information. Over a corpus of ~91k frames that is
        the difference between one pass and N.

        Falls back to per-prompt passes if the labels do not come back — better
        to be slow than to mis-assign every box to the wrong object.
        """
        from mmdet.apis import inference_detector
        self._ensure()
        caption = ". ".join(self.prompts) + "."
        r = inference_detector(self._model, frame_bgr,
                               text_prompt=caption, custom_entities=True)
        p = r.pred_instances
        sc = p.scores.detach().cpu().numpy()
        bb = p.bboxes.detach().cpu().numpy()
        lb = (p.labels.detach().cpu().numpy()
              if hasattr(p, "labels") and p.labels is not None else None)
        if lb is None or (len(sc) and lb.max() >= len(self.prompts)):
            return self._detect_separately(frame_bgr)
        keep = sc >= self.min_score
        return [(bb[keep & (lb == i)], sc[keep & (lb == i)])
                for i in range(len(self.prompts))]

    def _detect_separately(self, frame_bgr: np.ndarray):
        from mmdet.apis import inference_detector
        out = []
        for prompt in self.prompts:
            r = inference_detector(self._model, frame_bgr,
                                   text_prompt=prompt, custom_entities=True)
            p = r.pred_instances
            sc = p.scores.detach().cpu().numpy()
            bb = p.bboxes.detach().cpu().numpy()
            k = sc >= self.min_score
            out.append((bb[k], sc[k]))
        return out

    def features(self, frame_bgr: np.ndarray,
                 person_boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """[n_students, OBJECT_DIM]. One detector pass per prompt per frame."""
        if not len(person_boxes):
            return np.zeros((0, OBJECT_DIM), dtype=np.float32)
        per_obj = self.detect(frame_bgr)
        return np.stack([student_object_features(per_obj, b)
                         for b in person_boxes])
