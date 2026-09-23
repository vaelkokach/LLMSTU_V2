"""Steady the live overlay without lying about what the model said.

Two things flicker in a live run, for two different reasons, and both were
unsmoothed in the dashboard path.

**Boxes jump.** The detector runs every ``detector_stride`` frames and its box
regression is noisy frame to frame, so a stationary student's box snaps to a
slightly different place at each detection and holds there while the tracker
coasts. The motion is the detector's noise, not the student's.

**Labels flicker.** The temporal model re-decides every ``temporal_stride``
frames and two cues are often near-tied, so the chip alternates several times a
second. ``attention_runtime.yaml`` has declared ``label_smooth_window: 3`` and
``label_switch_margin: 0`` since the config was written, and
``realtime_infer.py`` implements exactly that -- but ``pipeline_bridge.run_live``
never read either. The dashboard has been running with no smoothing at all while
its config said otherwise.

What this does NOT do
---------------------
It does not touch the prediction. ``raw_cue``, ``conf``, ``abstained`` and
``alert_allowed`` are the model's own outputs and are passed through untouched,
so an alert still fires on what the model actually decided and the recorded
evidence is still the unsmoothed stream. Only what is DRAWN is steadied.

That distinction is the whole design. A smoother that also smoothed the alert
path would make the system look calmer by making it slower to report, which is
the wrong trade for an instructor-facing display.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence


class BoxSmoother:
    """Exponential moving average per track, on the DISPLAYED box only.

    ``alpha`` is the weight of the new observation. 0.35 keeps most of the
    history: a stationary student stops twitching, and a real move still lands
    within a few frames because the tracker's box moves consistently rather than
    noisily in that case.

    A box that jumps further than ``reset_px`` is taken as a real move (or a
    track id reused for a different student) and snaps rather than gliding, so
    the smoother never drags a box across the room.
    """

    def __init__(self, alpha: float = 0.35, reset_px: float = 120.0):
        self.alpha = float(alpha)
        self.reset_px = float(reset_px)
        self._last: Dict[int, List[float]] = {}

    def __call__(self, track_id: int, box: Sequence[float]) -> List[float]:
        b = [float(v) for v in box]
        prev = self._last.get(track_id)
        if prev is None or max(abs(p - c) for p, c in zip(prev, b)) > self.reset_px:
            self._last[track_id] = b
            return list(b)
        a = self.alpha
        out = [a * c + (1.0 - a) * p for p, c in zip(prev, b)]
        self._last[track_id] = out
        return list(out)

    def drop(self, keep: Sequence[int]) -> None:
        """Forget tracks the tracker dropped, so a reused id does not inherit."""
        k = set(int(i) for i in keep)
        for t in [t for t in self._last if t not in k]:
            del self._last[t]


class LabelSmoother:
    """Majority vote over a window, with hysteresis, per track.

    Mirrors ``realtime_infer.py`` so the dashboard and the CLI agree on what
    they display: a window of the last ``window`` decisions, the mode wins, and
    a switch away from the currently shown label needs to beat it by
    ``margin`` votes. ``margin = 0`` reproduces plain majority voting, which is
    what the shipped config asks for.
    """

    def __init__(self, window: int = 3, margin: int = 0):
        self.window = max(1, int(window))
        self.margin = max(0, int(margin))
        self._hist: Dict[int, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.window))
        self._shown: Dict[int, str] = {}

    def __call__(self, track_id: int, cue: str) -> str:
        h = self._hist[track_id]
        h.append(cue)
        counts: Dict[str, int] = {}
        for c in h:
            counts[c] = counts.get(c, 0) + 1
        best = max(counts, key=lambda c: (counts[c], c == self._shown.get(track_id)))
        prev = self._shown.get(track_id)
        if prev is not None and best != prev:
            if counts[best] - counts.get(prev, 0) < self.margin:
                best = prev
        self._shown[track_id] = best
        return best

    def drop(self, keep: Sequence[int]) -> None:
        k = set(int(i) for i in keep)
        for t in [t for t in self._hist if t not in k]:
            del self._hist[t]
            self._shown.pop(t, None)
