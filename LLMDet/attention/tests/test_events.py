import math

from attention.event_metrics import evaluate_events, match_episodes, temporal_iou
from attention.events import Episode, EventConfig, segment_events
from attention.taxonomy import CUE_TO_ID

S = CUE_TO_ID["screen_oriented"]
L = CUE_TO_ID["looking_away"]
H = CUE_TO_ID["head_down"]
P = CUE_TO_ID["phone_use"]

CFG = EventConfig(min_duration_s=3.0, max_gap_s=2.0, return_confirm_s=5.0)


def _times(n, step=1.0):
    return [i * step for i in range(n)]


def test_simple_off_screen_episode():
    cues = [S] * 5 + [L] * 6 + [S] * 10
    eps = segment_events(_times(len(cues)), cues, CFG)
    off = [e for e in eps if e.channel == "off_screen"]
    assert len(off) == 1
    assert off[0].t_start == 5.0
    assert off[0].t_end == 10.0


def test_short_flicker_not_an_episode():
    cues = [S] * 5 + [L] * 2 + [S] * 5
    eps = segment_events(_times(len(cues)), cues, CFG)
    assert not [e for e in eps if e.channel == "off_screen"]


def test_gap_bridging():
    # two looking_away runs separated by a 1s screen frame merge (gap <= 2s)
    cues = [S] * 5 + [L] * 4 + [S] + [L] * 4 + [S] * 8
    eps = segment_events(_times(len(cues)), cues, CFG)
    off = [e for e in eps if e.channel == "off_screen"]
    assert len(off) == 1
    assert off[0].duration >= 8.0


def test_return_to_task_marker():
    cues = [S] * 5 + [L] * 6 + [S] * 10
    eps = segment_events(_times(len(cues)), cues, CFG)
    ret = [e for e in eps if e.channel == "return_to_task"]
    assert len(ret) == 1
    assert ret[0].t_start == 11.0


def test_per_channel_min_duration():
    cfg = EventConfig(min_duration_s=3.0, max_gap_s=2.0, per_channel_min_duration={"phone_use": 1.0})
    cues = [S] * 5 + [P, P] + [S] * 5
    eps = segment_events(_times(len(cues)), cues, cfg)
    assert [e for e in eps if e.channel == "phone_use"]


def test_temporal_iou():
    a = Episode("x", 0.0, 10.0)
    b = Episode("x", 5.0, 15.0)
    assert abs(temporal_iou(a, b) - (5.0 / 15.0)) < 1e-9


def test_matching_is_one_to_one():
    pred = [Episode("off_screen", 0, 10), Episode("off_screen", 1, 9)]
    gt = [Episode("off_screen", 0, 10)]
    m = match_episodes(pred, gt, iou_thr=0.3)
    assert len(m) == 1
    assert m[0][0] == 0  # exact match wins


def test_evaluate_events_metrics():
    gt = [Episode("off_screen", 10, 20), Episode("phone_use", 30, 40)]
    pred = [Episode("off_screen", 12, 21)]  # matched, late by 2s; phone missed
    out = evaluate_events(pred, gt, observed_duration_s=3600.0, iou_thr=0.3)
    assert out["overall"]["num_matched"] == 1.0
    assert out["overall"]["missed_event_rate"] == 0.5
    assert out["overall"]["false_alerts_per_hour"] == 0.0
    assert abs(out["off_screen"]["onset_error_s"] - 2.0) < 1e-9
    assert abs(out["off_screen"]["alert_delay_s"] - 2.0) < 1e-9
    assert math.isnan(out["phone_use"]["onset_error_s"])
    fp = evaluate_events([Episode("off_screen", 100, 110)], [], 3600.0)
    assert fp["overall"]["false_alerts_per_hour"] == 1.0
