from attention.rule_baseline import RuleConfig, classify_timeline
from attention.taxonomy import CUE_TO_ID

S = CUE_TO_ID["screen_oriented"]
L = CUE_TO_ID["looking_away"]
H = CUE_TO_ID["head_down"]
P = CUE_TO_ID["phone_use"]
U = CUE_TO_ID["uncertain"]

CFG = RuleConfig(window_s=10.0, looking_away_s=5.0, head_down_s=20.0, min_history_s=3.0)


def _times(n, step=1.0):
    return [i * step for i in range(n)]


def test_all_screen_oriented_is_on_task():
    n = 15
    out = classify_timeline(_times(n), [S] * n, CFG)
    assert out[-1] == "on_task"
    # early frames lack history
    assert out[0] == "unknown"


def test_phone_alert_immediate():
    cues = [S] * 10 + [P]
    out = classify_timeline(_times(11), cues, CFG)
    assert out[-1] == "phone_use_alert"


def test_short_glance_away_stays_on_task():
    cues = [S] * 10 + [L, L] + [S] * 3
    out = classify_timeline(_times(len(cues)), cues, CFG)
    assert out[-1] == "on_task"


def test_sustained_looking_away_flags_distraction():
    cues = [S] * 5 + [L] * 7
    out = classify_timeline(_times(len(cues)), cues, CFG)
    assert out[-1] == "possible_distraction"


def test_long_head_down_is_off_task():
    cues = [S] * 5 + [H] * 25
    out = classify_timeline(_times(len(cues)), cues, CFG)
    assert out[-1] == "off_task_cue"


def test_uncertain_dominated_window_is_unknown():
    cues = [S] * 5 + [U] * 10
    out = classify_timeline(_times(len(cues)), cues, CFG)
    assert out[-1] == "unknown"


def test_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        classify_timeline([0.0, 1.0], [S], CFG)
