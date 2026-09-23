"""Rule-based baseline over per-frame cue timelines.

Interpretable windowed rules (supervisor's item 18). Operates on one
student's timeline: parallel arrays of timestamps (seconds) and cue class
ids from :mod:`attention.taxonomy`. Emits a per-frame window state:

- ``on_task``            — majority of the trailing window is screen_oriented
- ``possible_distraction`` — continuous looking_away/turned_to_peer streak
                           longer than ``looking_away_s``
- ``off_task_cue``       — continuous head_down or off-screen orientation
                           longer than ``head_down_s``
- ``phone_use_alert``    — phone_use cue fires (immediate; phone visibility
                           is a strong cue on its own)
- ``unknown``            — not enough history, or the window is dominated by
                           uncertain frames

The temporal model must beat this baseline to justify its existence.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

from attention.taxonomy import CUE_TO_ID

STATES = ["on_task", "possible_distraction", "off_task_cue", "phone_use_alert", "unknown"]

_DISTRACT_CUES = {CUE_TO_ID["looking_away"], CUE_TO_ID["turned_to_peer"]}
_OFFTASK_CUES = {CUE_TO_ID["head_down"]}
_ONTASK_CUE = CUE_TO_ID["screen_oriented"]
_PHONE_CUE = CUE_TO_ID["phone_use"]
_UNCERTAIN_CUE = CUE_TO_ID["uncertain"]


@dataclass
class RuleConfig:
    window_s: float = 10.0          # trailing window for the on-task majority vote
    on_task_majority: float = 0.5   # fraction of window frames that must be screen_oriented
    looking_away_s: float = 5.0     # continuous distraction streak -> possible_distraction
    head_down_s: float = 20.0       # continuous head_down/off-screen -> off_task_cue
    min_history_s: float = 3.0      # emit `unknown` before this much history
    uncertain_max_frac: float = 0.5 # window with more uncertain than this -> unknown

    @classmethod
    def from_dict(cls, d: Dict) -> "RuleConfig":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


def _streak_duration(times: Sequence[float], cues: Sequence[int], idx: int, cue_set) -> float:
    """Duration of the continuous run of cues from ``cue_set`` ending at idx."""
    if cues[idx] not in cue_set:
        return 0.0
    j = idx
    while j > 0 and cues[j - 1] in cue_set:
        j -= 1
    return times[idx] - times[j]


def classify_timeline(times: Sequence[float], cues: Sequence[int], cfg: RuleConfig = None) -> List[str]:
    """Per-frame window state for one student's cue timeline."""
    cfg = cfg or RuleConfig()
    if len(times) != len(cues):
        raise ValueError("times and cues must have equal length")
    out: List[str] = []
    for i, t in enumerate(times):
        if t - times[0] < cfg.min_history_s:
            out.append("unknown")
            continue

        # phone first: strongest, immediate cue
        if cues[i] == _PHONE_CUE:
            out.append("phone_use_alert")
            continue

        # sustained head-down / off-screen
        offtask_run = _streak_duration(times, cues, i, _OFFTASK_CUES | _DISTRACT_CUES)
        headdown_run = _streak_duration(times, cues, i, _OFFTASK_CUES)
        if headdown_run >= cfg.head_down_s or offtask_run >= cfg.head_down_s:
            out.append("off_task_cue")
            continue

        # sustained looking-away / peer
        if _streak_duration(times, cues, i, _DISTRACT_CUES) >= cfg.looking_away_s:
            out.append("possible_distraction")
            continue

        # trailing-window majority
        w0 = t - cfg.window_s
        win = [c for tt, c in zip(times, cues) if w0 < tt <= t]
        if not win:
            out.append("unknown")
            continue
        n_unc = sum(1 for c in win if c == _UNCERTAIN_CUE)
        if n_unc / len(win) > cfg.uncertain_max_frac:
            out.append("unknown")
            continue
        n_on = sum(1 for c in win if c == _ONTASK_CUE)
        n_eff = len(win) - n_unc
        if n_eff > 0 and n_on / n_eff >= cfg.on_task_majority:
            out.append("on_task")
        else:
            out.append("possible_distraction")
    return out
