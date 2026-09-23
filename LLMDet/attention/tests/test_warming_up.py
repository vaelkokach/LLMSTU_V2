"""Boxes before the model's first prediction.

At the trained 1 Hz a track needs ``min_frames_for_pred`` seconds of history
before it can be scored, and the replay used to omit such tracks entirely -- so
the first ~10 s of every session showed video with no boxes at all.

What these pin is the part that would be silent when wrong: a warming-up student
must be drawn from its first frame, must turn into a real cue exactly when the
history allows it, and must never count as on- or off-task or raise an alert.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from attention.taxonomy import CUE_CLASSES                      # noqa: E402
from attention.thesis_eval.runtime import RuntimeBundle         # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

SR = pytest.importorskip("session_replay")

WIDTH = 4
MINF = 3


class _TopScreen(torch.nn.Module):
    """screen_oriented at probability 0.7 on every frame."""
    def forward(self, x):
        p = torch.full((len(CUE_CLASSES),), 0.3 / (len(CUE_CLASSES) - 1))
        p[0] = 0.7
        return {"logits": torch.log(p).expand(1, x.shape[1], -1).clone()}


class _FakeCache:
    """The slice of SessionCache that replay() reads: 1 fps, N tracks."""
    def __init__(self, n_frames=8, n_tracks=2):
        self.meta = {"fps": 1.0, "source_width": 100,
                     "inference": {"window_size": 8, "temporal_stride": 1,
                                   "min_frames_for_pred": MINF,
                                   "temporal_input_fps": 1.0}}
        self.frame_index = np.arange(n_frames)
        tids, boxes, self._by_frame = [], [], {}
        for f in range(n_frames):
            self._by_frame[f] = []
            for k in range(n_tracks):
                self._by_frame[f].append(len(tids))
                tids.append(k)
                boxes.append([10.0 * k, 0.0, 10.0 * k + 8, 20.0])
        self.track_id, self.bbox = np.array(tids), np.array(boxes)

    @property
    def fps(self):
        return 1.0

    def rows_for(self, frame):
        return self._by_frame.get(int(frame), [])

    def jpeg(self, frame):
        return None

    def live_vector(self, rows, backend):
        return np.zeros((len(rows), WIDTH), dtype=np.float32)


def _bundle():
    b = RuntimeBundle(model=_TopScreen(), experiment_id="stub", model_name="stub",
                      feature_config="stub", input_dim=WIDTH, seed=42,
                      checkpoint="stub", device=torch.device("cpu"),
                      live_input_width=WIDTH)
    b.display_threshold, b.alert_threshold = 0.46, 0.66
    return b


def _run(monkeypatch, n_frames=8, n_tracks=2):
    monkeypatch.setattr(SR, "CONTEXT_INFERENCE", {})
    frames = []
    entry = SimpleNamespace(head_pose_backend="mediapipe", taxonomy="cue6",
                            vlm_policy="agreement")
    SR.replay(_FakeCache(n_frames, n_tracks), entry, _bundle(),
              lambda t, jpg, students, names: frames.append(students),
              realtime=False, overlay=False)
    return frames


def test_every_tracked_student_has_a_box_from_the_first_frame(monkeypatch):
    frames = _run(monkeypatch)
    assert all(len(f) == 2 for f in frames), [len(f) for f in frames]


def test_students_warm_up_until_the_history_allows_a_prediction(monkeypatch):
    frames = _run(monkeypatch)
    for i, f in enumerate(frames):
        for s in f.values():
            if i < MINF - 1:            # history holds i + 1 samples at 1 fps
                assert s.get("warming") and s["cue"] == SR.WARMUP_LABEL
                assert s["conf"] is None and s["alert_allowed"] is False
            else:
                assert not s.get("warming"), f"frame {i} still warming"
                assert s["cue"] == "screen_oriented"


def test_a_warming_box_is_the_tracked_box(monkeypatch):
    frames = _run(monkeypatch)
    # Seat numbers are handed out 1, 2, ... in order of appearance, so tracker id
    # 0 (the box at x=0) is seat 1.
    assert frames[0]["1"]["bbox"] == [0.0, 0.0, 8.0, 20.0]
    assert frames[0]["2"]["bbox"] == [10.0, 0.0, 18.0, 20.0]


def test_warming_colour_is_neither_a_cue_nor_the_abstention_grey():
    assert SR.WARMUP_COLOUR != SR.UNKNOWN_COLOUR
    assert SR.WARMUP_COLOUR not in SR.CUE_COLOUR.values()


# ---------------------------------------------------------------- server ---

SV = pytest.importorskip("server")


def _student(cue, dwell=0.0, alert_allowed=True):
    return {"cue": cue, "conf": 0.9, "bbox": [0, 0, 1, 1], "dwell": dwell,
            "alerted": False, "alert_allowed": alert_allowed}


@pytest.fixture
def clean_server():
    SV.reset_state()
    SV.CONTEXT.pop("policy", None)
    yield SV
    SV.reset_state()


def test_warming_students_count_as_present_but_not_as_evidence(clean_server):
    students = {
        "1": SR.warming_record([0, 0, 1, 1]),
        "2": SR.warming_record([2, 0, 3, 1]),
        "3": _student("phone_use"),
        "4": _student("screen_oriented"),
    }
    clean_server.push_frame(1.0, None, students, list(CUE_CLASSES))
    cs = clean_server.STATE["class_summary"]
    assert cs["n_students"] == 4 and cs["warming_up"] == 2
    assert cs["off_task"] == 1 and cs["off_task_pct"] == 50
    assert SR.WARMUP_LABEL not in cs["by_cue"]


def test_a_warming_student_never_alerts(clean_server):
    w = SR.warming_record([0, 0, 1, 1])
    w["dwell"] = 999.0
    clean_server.push_frame(1.0, None, {"1": w, "2": _student("phone_use", dwell=20.0)},
                            list(CUE_CLASSES))
    alerts = list(clean_server.STATE["alerts"])
    assert [a["seat"] for a in alerts] == ["2"]
