"""Event-level segmentation of per-student cue timelines.

Converts frame-level cues into episodes with start/end times, per channel:

- ``off_screen``       — looking_away
- ``head_down``        — head_down
- ``phone_use``        — phone_use
- ``peer_interaction`` — turned_to_peer
- ``inactivity``       — idle_other sustained
- ``return_to_task``   — instantaneous marker (zero-length episode) emitted
                         when any off-task episode ends and screen_oriented
                         holds for ``confirm_s``

Hysteresis: an episode opens after the cue holds for ``min_duration_s`` and
closes only after the cue has been absent for ``max_gap_s`` (short flickers
neither open nor close episodes).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from attention.taxonomy import CUE_TO_ID

# ``inactivity`` was REMOVED on 2026-08-01. It was defined as
# ``{CUE_TO_ID["head_down"]}`` — byte-identical to the ``head_down`` channel — so
# it emitted a duplicate of every head-down episode. The consequences were not
# cosmetic:
#
#   * the human gold's "24 episodes" were 7 head_down + 7 identical inactivity
#     + 5 phone_use + 3 return_to_task (2 distinct) + 2 peer_interaction, i.e.
#     **16 distinct episodes**, and head_down carried double weight in every
#     aggregate recall and in false-alerts/hour;
#   * an instructor would have been shown the same episode twice.
#
# A genuine inactivity channel needs motion, not a cue alias. The `dynamic`
# feature block (motion_now/mean/std, scale_change) now supplies exactly that,
# so this is a concrete piece of future work rather than a missing capability —
# [internal notes, not included] and [internal notes, not included] A0-3.
#
# ``attention/thesis_eval/segmentation.dedup_episodes`` still drops the channel
# from *stored* gold files written before this change.
EVENT_CHANNELS = ["off_screen", "head_down", "phone_use", "peer_interaction"]

#: Retired channels, kept only so old artefacts can be recognised and dropped.
LEGACY_ALIAS_CHANNELS = ("inactivity",)

_CHANNEL_CUES: Dict[str, set] = {
    "off_screen": {CUE_TO_ID["looking_away"]},
    "head_down": {CUE_TO_ID["head_down"]},
    "phone_use": {CUE_TO_ID["phone_use"]},
    "peer_interaction": {CUE_TO_ID["turned_to_peer"]},
}
_ONTASK = CUE_TO_ID["screen_oriented"]


@dataclass
class Episode:
    channel: str
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class EventConfig:
    min_duration_s: float = 3.0
    max_gap_s: float = 2.0
    return_confirm_s: float = 5.0
    per_channel_min_duration: Dict[str, float] = field(default_factory=dict)

    def min_dur(self, channel: str) -> float:
        return float(self.per_channel_min_duration.get(channel, self.min_duration_s))

    @classmethod
    def from_dict(cls, d: Dict) -> "EventConfig":
        d = dict(d or {})
        per = d.pop("per_channel_min_duration", {}) or {}
        base = {k: float(v) for k, v in d.items() if k in ("min_duration_s", "max_gap_s", "return_confirm_s")}
        return cls(per_channel_min_duration={k: float(v) for k, v in per.items()}, **base)


def _segment_channel(
    times: Sequence[float], active: Sequence[bool], min_duration: float, max_gap: float, channel: str
) -> List[Episode]:
    episodes: List[Episode] = []
    start = None
    last_active = None
    for t, a in zip(times, active):
        if a:
            if start is None:
                start = t
            last_active = t
        elif start is not None and t - last_active > max_gap:
            if last_active - start >= min_duration:
                episodes.append(Episode(channel, start, last_active))
            start = None
            last_active = None
    if start is not None and last_active is not None and last_active - start >= min_duration:
        episodes.append(Episode(channel, start, last_active))
    return episodes


def segment_events(times: Sequence[float], cues: Sequence[int], cfg: EventConfig = None) -> List[Episode]:
    """Segment one student's cue timeline into episodes across all channels,
    plus zero-length ``return_to_task`` markers."""
    cfg = cfg or EventConfig()
    if len(times) != len(cues):
        raise ValueError("times and cues must have equal length")
    episodes: List[Episode] = []
    for channel in EVENT_CHANNELS:
        cue_set = _CHANNEL_CUES[channel]
        active = [c in cue_set for c in cues]
        episodes.extend(_segment_channel(times, active, cfg.min_dur(channel), cfg.max_gap_s, channel))

    # return-to-task markers: after each off-task episode, find sustained on-task
    off_task = sorted(
        [e for e in episodes if e.channel in ("off_screen", "head_down", "phone_use", "peer_interaction")],
        key=lambda e: e.t_end,
    )
    for ep in off_task:
        run_start = None
        for t, c in zip(times, cues):
            if t <= ep.t_end:
                continue
            if c == _ONTASK:
                if run_start is None:
                    run_start = t
                if t - run_start >= cfg.return_confirm_s:
                    episodes.append(Episode("return_to_task", run_start, run_start))
                    break
            else:
                run_start = None
    return sorted(episodes, key=lambda e: (e.t_start, e.channel))
