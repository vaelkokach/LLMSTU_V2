"""Temporal body-language and gaze features derived per track.

Closes two `Thesis_Topic.md` requirements the pipeline had no channel for:

  * **Body language (fidgeting, leaning forward/backward)** — motion statistics
    from the student's own box across consecutive frames. Fidgeting is by
    definition temporal: a single frame cannot show it, which is why the
    per-frame feature block could never represent it.
  * **Gaze direction (at the instructor vs at distractions)** — head pose alone
    is not gaze relative to a task. A student whose head rests at yaw = -15 deg
    because that is where their monitor sits is ON task; the same angle for a
    student facing forward is off task.

The gaze features use a **personalised baseline**: each student's own median
head pose over their track is treated as their task-oriented direction, and the
features measure deviation from it. This needs no monitor calibration, no room
geometry, and no extra labels, and it is robust to seating position — which a
fixed "screen direction" prior is not.

All features are computed from data already present (bbox + cached head pose),
so they cost nothing at build time and require no new annotation.

Dims produced (7):
    0 motion_now        |Δ centre| this frame, normalised by box diagonal
    1 motion_mean       mean |Δ centre| over a trailing window
    2 motion_std        std of |Δ centre| over the window  (fidget signal)
    3 scale_change      Δ box area / area  (leaning forward/back)
    4 gaze_dev_yaw      yaw − track-median yaw
    5 gaze_dev_pitch    pitch − track-median pitch
    6 gaze_dev_mag      Euclidean magnitude of the deviation
"""
from typing import List, Optional, Sequence

import numpy as np

DYNAMIC_DIM = 7


def compute_dynamic(bboxes: Sequence[Sequence[float]],
                    poses: Optional[np.ndarray] = None,
                    window: int = 5) -> np.ndarray:
    """[T, 7] features for one track.

    ``bboxes`` are per-frame [x1, y1, x2, y2] in frame coordinates.
    ``poses``  is [T, >=4] from the affect cache: yaw, pitch, roll, face_found.
                Rows with face_found == 0 are excluded from the personalised
                baseline and yield zero gaze-deviation — a missing face must not
                masquerade as "no deviation from baseline".
    """
    b = np.asarray(bboxes, dtype=np.float32)
    T = len(b)
    out = np.zeros((T, DYNAMIC_DIM), dtype=np.float32)
    if T == 0:
        return out

    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5
    w = np.maximum(b[:, 2] - b[:, 0], 1.0)
    h = np.maximum(b[:, 3] - b[:, 1], 1.0)
    diag = np.sqrt(w ** 2 + h ** 2)
    area = w * h

    d = np.zeros(T, dtype=np.float32)
    d[1:] = np.sqrt((cx[1:] - cx[:-1]) ** 2 + (cy[1:] - cy[:-1]) ** 2)
    d = d / np.maximum(diag, 1.0)          # scale-invariant: near/far students comparable
    out[:, 0] = d

    for t in range(T):
        s = max(0, t - window + 1)
        seg = d[s:t + 1]
        out[t, 1] = seg.mean()
        out[t, 2] = seg.std()

    da = np.zeros(T, dtype=np.float32)
    da[1:] = (area[1:] - area[:-1]) / np.maximum(area[:-1], 1.0)
    out[:, 3] = np.clip(da, -1.0, 1.0)

    if poses is not None and len(poses) == T:
        p = np.asarray(poses, dtype=np.float32)
        valid = p[:, 3] > 0.5
        if valid.sum() >= 3:
            # Median, not mean: robust to the transient look-aways that are
            # precisely the events we want to measure deviation FROM.
            base_y = float(np.median(p[valid, 0]))
            base_p = float(np.median(p[valid, 1]))
            dy = np.where(valid, p[:, 0] - base_y, 0.0)
            dp = np.where(valid, p[:, 1] - base_p, 0.0)
            out[:, 4] = dy
            out[:, 5] = dp
            out[:, 6] = np.sqrt(dy ** 2 + dp ** 2)
    return out
